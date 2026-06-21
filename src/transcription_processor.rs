use parking_lot::{Mutex, RwLock};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{broadcast, mpsc};

use crate::backend::TranscriptionBackend;
use crate::config::SpeechConfig;
#[cfg(any(
    feature = "backend-ctranslate2",
    feature = "backend-whisper-cpp",
    feature = "backend-moonshine",
    feature = "backend-parakeet"
))]
use crate::post_processor;
use crate::silero_audio_processor::{AudioSegment, SileroVad, VadConfig, VadState};
use crate::state::{AudioVisualizationData, ProcessingState};
use crate::transcription_stats::TranscriptionStats;

/// Extract the last N words from text for use as a prompt
fn extract_prompt_context(text: &str, max_words: usize) -> String {
    let word_count = text.split_whitespace().count();
    if word_count <= max_words {
        text.to_string()
    } else {
        text.split_whitespace()
            .skip(word_count - max_words)
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Find natural pause points in audio using VAD.
/// Returns sample indices where pauses occur (good places to split chunks).
fn find_pause_points(samples: &[f32], sample_rate: usize) -> Vec<usize> {
    let model_path = match crate::download::model_cache_dir() {
        Ok(models_dir) => models_dir.join("silero_vad.onnx"),
        Err(e) => {
            tracing::info!(
                "Could not resolve model cache directory ({e}), falling back to time-based chunking"
            );
            return Vec::new();
        }
    };

    if !model_path.exists() {
        tracing::info!(
            "VAD model not found at {:?}, falling back to time-based chunking",
            model_path
        );
        return Vec::new();
    }

    // Create VAD with config tuned for finding pauses
    let config = VadConfig {
        threshold: 0.3, // Slightly higher threshold for clearer boundaries
        speech_end_threshold: 0.2,
        frame_size: 512,
        sample_rate,
        hangbefore_frames: 3,
        hangover_frames: 15, // Shorter hangover to detect pauses faster
        hop_samples: 160,
        max_buffer_duration: samples.len() + 1024,
        max_segment_count: 1000,
        silence_tolerance_frames: 3,
        speech_prob_smoothing: 0.3,
    };

    let mut vad = match SileroVad::new(config, &model_path) {
        Ok(v) => v,
        Err(e) => {
            tracing::info!(
                "Failed to initialize VAD: {:?}, falling back to time-based chunking",
                e
            );
            return Vec::new();
        }
    };

    // Process audio through VAD to track state transitions
    // Only consider pauses that last at least this long (filters out brief hesitations)
    let min_pause_duration_ms = 300;
    let min_pause_samples = (sample_rate * min_pause_duration_ms) / 1000;

    let mut pause_points = Vec::new();
    let frame_size = 512;
    let hop_samples = 160;
    let mut current_sample = 0;
    let mut was_speaking = false;
    let mut pause_start: Option<usize> = None;

    // Process in frames
    let mut frame = vec![0.0f32; frame_size];
    let mut buffer_pos = 0;

    for &sample in samples {
        frame[buffer_pos] = sample;
        buffer_pos += 1;

        if buffer_pos >= frame_size {
            if let Ok(state) = vad.process_frame(&frame, hop_samples) {
                let is_speaking = matches!(state, VadState::Speech | VadState::PossibleSpeech);

                // Detect transition from speech to silence (potential pause start)
                if was_speaking && !is_speaking {
                    pause_start = Some(current_sample);
                }

                // Detect transition from silence to speech (pause ended)
                // Only record if pause was long enough
                if !was_speaking && is_speaking {
                    if let Some(start) = pause_start {
                        let pause_duration = current_sample.saturating_sub(start);
                        if pause_duration >= min_pause_samples {
                            // Use the midpoint of the pause as the split point
                            pause_points.push(start + pause_duration / 2);
                        }
                    }
                    pause_start = None;
                }

                was_speaking = is_speaking;
            }

            // Slide the frame
            frame.copy_within(hop_samples.., 0);
            buffer_pos = frame_size - hop_samples;
            current_sample += hop_samples;
        }
    }

    // Handle trailing pause (audio ends in silence)
    if let Some(start) = pause_start {
        let pause_duration = current_sample.saturating_sub(start);
        if pause_duration >= min_pause_samples {
            pause_points.push(start + pause_duration / 2);
        }
    }

    pause_points
}

/// Handles the processing of audio segments for transcription
pub struct TranscriptionProcessor {
    backend: Arc<Mutex<Option<Arc<TranscriptionBackend>>>>,
    backend_ready: Arc<AtomicBool>,
    language: String,
    app_config: Arc<SpeechConfig>,
    running: Arc<AtomicBool>,
    transcription_done_tx: mpsc::UnboundedSender<()>,
    transcription_stats: Arc<Mutex<TranscriptionStats>>,
    audio_visualization_data: Arc<RwLock<AudioVisualizationData>>,
}

impl TranscriptionProcessor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        backend: Arc<Mutex<Option<Arc<TranscriptionBackend>>>>,
        backend_ready: Arc<AtomicBool>,
        language: String,
        app_config: Arc<SpeechConfig>,
        running: Arc<AtomicBool>,
        transcription_done_tx: mpsc::UnboundedSender<()>,
        transcription_stats: Arc<Mutex<TranscriptionStats>>,
        audio_visualization_data: Arc<RwLock<AudioVisualizationData>>,
    ) -> Self {
        Self {
            backend,
            backend_ready,
            language,
            app_config,
            running,
            transcription_done_tx,
            transcription_stats,
            audio_visualization_data,
        }
    }

    /// Transcribe an audio segment using the backend.
    /// Optionally accepts an initial prompt for chunk continuity (whisper.cpp only; CT2 ignores it).
    fn transcribe_segment(
        backend: &Arc<Mutex<Option<Arc<TranscriptionBackend>>>>,
        segment: &AudioSegment,
        language: &str,
        app_config: &SpeechConfig,
        stats: &Arc<Mutex<TranscriptionStats>>,
        audio_visualization_data: &Arc<RwLock<AudioVisualizationData>>,
        initial_prompt: Option<&str>,
    ) -> String {
        let log_stats_enabled = app_config.debug_config.log_stats_enabled;

        // Set processing state to transcribing
        {
            let mut audio_data = audio_visualization_data.write();
            audio_data.set_processing_state(ProcessingState::Transcribing);
        }

        if log_stats_enabled {
            tracing::info!(
                "Transcribing segment from {:.2}s to {:.2}s{}",
                segment.start_time,
                segment.end_time,
                if initial_prompt.is_some() {
                    " (with prompt)"
                } else {
                    ""
                }
            );
        }

        let start_time = Instant::now();

        let backend_arc = {
            let lock = backend.lock();
            lock.as_ref().map(Arc::clone)
        }; // lock dropped here

        let Some(backend_ref) = backend_arc.as_ref() else {
            let total_duration = start_time.elapsed();
            if log_stats_enabled {
                tracing::info!(
                    "Backend not available (checked in {:.2}s)",
                    total_duration.as_secs_f32()
                );
            }
            {
                let mut audio_data = audio_visualization_data.write();
                audio_data.set_processing_state(ProcessingState::Idle);
            }
            return "[backend not available]".to_string();
        };

        #[cfg(all(
            not(feature = "backend-ctranslate2"),
            not(feature = "backend-whisper-cpp"),
            not(feature = "backend-moonshine"),
            not(feature = "backend-parakeet")
        ))]
        {
            let _ = backend_ref;
            let _ = (language, stats);
            {
                let mut audio_data = audio_visualization_data.write();
                audio_data.set_processing_state(ProcessingState::Error);
            }
            "[no transcription backend feature enabled]".to_string()
        }

        #[cfg(any(
            feature = "backend-ctranslate2",
            feature = "backend-whisper-cpp",
            feature = "backend-moonshine",
            feature = "backend-parakeet"
        ))]
        {
            #[cfg(feature = "backend-whisper-cpp")]
            let whisper_cpp_options = {
                let mut options = app_config.whisper_cpp_options.clone();
                if let Some(prompt) = initial_prompt.filter(|prompt| !prompt.is_empty()) {
                    options.initial_prompt = Some(prompt.to_string());
                }
                options
            };
            let inference_start = Instant::now();
            let segment_duration = (segment.end_time - segment.start_time) as f32;

            let result = match &**backend_ref {
                #[cfg(feature = "backend-ctranslate2")]
                crate::backend::TranscriptionBackend::CTranslate2(ct2_backend) => ct2_backend
                    .transcribe(
                        &segment.samples,
                        language,
                        &app_config.common_transcription_options,
                        &app_config.ctranslate2_options,
                        segment.sample_rate,
                    ),
                #[cfg(feature = "backend-whisper-cpp")]
                crate::backend::TranscriptionBackend::WhisperCpp(whisper_cpp_backend) => {
                    whisper_cpp_backend.transcribe(
                        &segment.samples,
                        language,
                        &app_config.common_transcription_options,
                        &whisper_cpp_options,
                        segment.sample_rate,
                    )
                }
                #[cfg(feature = "backend-moonshine")]
                crate::backend::TranscriptionBackend::Moonshine(moonshine_backend) => {
                    moonshine_backend.transcribe(
                        &segment.samples,
                        language,
                        &app_config.common_transcription_options,
                        &app_config.moonshine_options,
                        segment.sample_rate,
                    )
                }
                #[cfg(feature = "backend-parakeet")]
                crate::backend::TranscriptionBackend::Parakeet(parakeet_backend) => {
                    parakeet_backend.transcribe(
                        &segment.samples,
                        language,
                        &app_config.common_transcription_options,
                        &app_config.parakeet_options,
                        segment.sample_rate,
                    )
                }
                #[cfg(feature = "backend-nemotron")]
                crate::backend::TranscriptionBackend::Nemotron(nemotron_backend) => {
                    nemotron_backend.transcribe(
                        &segment.samples,
                        language,
                        &app_config.common_transcription_options,
                        &app_config.nemotron_options,
                        segment.sample_rate,
                    )
                }
            };

            match result {
                Ok(transcription) => {
                    let inference_duration = inference_start.elapsed();
                    let total_duration = start_time.elapsed();
                    let inference_secs = inference_duration.as_secs_f32();
                    let total_secs = total_duration.as_secs_f32();

                    if let Some(mut stats_lock) = stats.try_lock() {
                        stats_lock.update(segment_duration, inference_secs, total_secs);
                    }

                    if log_stats_enabled {
                        tracing::info!(
                            "Transcription timing: Segment length: {:.2}s, Inference time: {:.2}s, Total: {:.2}s, RTF: {:.2}",
                            segment_duration, inference_secs, total_secs, inference_secs / segment_duration
                        );
                        tracing::info!("Transcription (raw): '{}'", transcription);
                    }

                    let processed_transcription = post_processor::post_process_text(
                        transcription,
                        &app_config.post_process_config,
                    );

                    if log_stats_enabled {
                        tracing::info!("Transcription (processed): '{}'", processed_transcription);
                    }

                    {
                        let mut audio_data = audio_visualization_data.write();
                        audio_data.set_processing_state(ProcessingState::Idle);
                    }

                    processed_transcription
                }
                Err(e) => {
                    let total_duration = start_time.elapsed();
                    if log_stats_enabled {
                        tracing::info!(
                            "Transcription error after {:.2}s: {}",
                            total_duration.as_secs_f32(),
                            e
                        );
                    }
                    {
                        let mut audio_data = audio_visualization_data.write();
                        audio_data.set_processing_state(ProcessingState::Error);
                    }
                    format!("[transcription error: {}]", e)
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn process_segment(
        segment: AudioSegment,
        backend: Arc<Mutex<Option<Arc<TranscriptionBackend>>>>,
        language: String,
        app_config: Arc<SpeechConfig>,
        stats: Arc<Mutex<TranscriptionStats>>,
        audio_visualization_data: Arc<RwLock<AudioVisualizationData>>,
        transcript_tx: broadcast::Sender<crate::real_time_transcriber::TranscriptionMessage>,
        log_stats_enabled: bool,
    ) {
        let segment_info = format!(
            "Segment {:.2}s-{:.2}s",
            segment.start_time, segment.end_time
        );
        let start_time = Instant::now();

        let processing_result = tokio::task::spawn_blocking(move || {
            let session_id = segment.session_id.clone();

            if segment.is_manual {
                let transcription = Self::process_manual_segment(
                    &backend,
                    &segment,
                    &language,
                    &app_config,
                    &stats,
                    &audio_visualization_data,
                );

                if transcription.is_empty() {
                    tracing::info!("Manual transcription resulted in empty text");
                    None
                } else {
                    Some(crate::real_time_transcriber::TranscriptionMessage {
                        text: transcription,
                        session_id,
                        is_final: true,
                    })
                }
            } else {
                let transcription = Self::transcribe_segment(
                    &backend,
                    &segment,
                    &language,
                    &app_config,
                    &stats,
                    &audio_visualization_data,
                    None,
                );

                if transcription.is_empty() {
                    None
                } else {
                    Some(crate::real_time_transcriber::TranscriptionMessage {
                        text: transcription,
                        session_id,
                        is_final: true,
                    })
                }
            }
        })
        .await;

        match processing_result {
            Ok(Some(message)) => {
                if let Err(e) = transcript_tx.send(message) {
                    tracing::warn!("Failed to send transcription: {}", e);
                }
            }
            Ok(None) => {}
            Err(e) => tracing::warn!("Transcription worker task failed: {}", e),
        }

        if log_stats_enabled {
            tracing::info!(
                "Segment processing finished for {} in {:.2}s",
                segment_info,
                start_time.elapsed().as_secs_f32()
            );
        }
    }

    /// Spawn the streaming worker: consumes live-audio `StreamEvent`s and emits
    /// interim (`is_final = false`) transcripts for streaming-capable backends.
    /// Non-streaming backends drain events and stay silent (the segment path
    /// produces their final result as usual).
    pub fn start_streaming(
        &self,
        mut stream_rx: mpsc::Receiver<crate::real_time_transcriber::StreamEvent>,
        transcript_tx: broadcast::Sender<crate::real_time_transcriber::TranscriptionMessage>,
    ) -> tokio::task::JoinHandle<()> {
        use crate::real_time_transcriber::{StreamEvent, TranscriptionMessage};

        let backend = self.backend.clone();
        let backend_ready = self.backend_ready.clone();
        let running = self.running.clone();
        let language = self.language.clone();
        let app_config = self.app_config.clone();

        tokio::spawn(async move {
            let mut session_id: Option<String> = None;
            // Whether the current utterance's backend supports streaming.
            let mut active = false;

            while let Some(event) = stream_rx.recv().await {
                if !running.load(Ordering::Relaxed) {
                    break;
                }
                // Snapshot the loaded backend (None during (re)load).
                let backend_arc = {
                    let lock = backend.lock();
                    lock.as_ref().map(Arc::clone)
                };
                let Some(b) = backend_arc else {
                    continue;
                };
                if !backend_ready.load(Ordering::Relaxed) {
                    continue;
                }

                match event {
                    StreamEvent::Start { session_id: sid } => {
                        session_id = sid;
                        active = b.supports_streaming();
                        if active {
                            let lang = language.clone();
                            let opts = app_config.nemotron_options.clone();
                            let _ =
                                tokio::task::spawn_blocking(move || b.stream_reset(&lang, &opts))
                                    .await;
                        }
                    }
                    StreamEvent::Chunk(samples) => {
                        if !active {
                            continue;
                        }
                        let partial =
                            tokio::task::spawn_blocking(move || b.stream_push(&samples)).await;
                        if let Ok(Ok(text)) = partial {
                            if !text.is_empty() {
                                let _ = transcript_tx.send(TranscriptionMessage {
                                    text,
                                    session_id: session_id.clone(),
                                    is_final: false,
                                });
                            }
                        }
                    }
                    StreamEvent::End => {
                        if !active {
                            continue;
                        }
                        active = false;
                        let final_partial =
                            tokio::task::spawn_blocking(move || b.stream_finish()).await;
                        if let Ok(Ok(text)) = final_partial {
                            if !text.is_empty() {
                                // Interim only; the segment path emits the committed final.
                                let _ = transcript_tx.send(TranscriptionMessage {
                                    text,
                                    session_id: session_id.clone(),
                                    is_final: false,
                                });
                            }
                        }
                    }
                }
            }
        })
    }

    pub fn start(
        &self,
        mut segment_rx: mpsc::Receiver<AudioSegment>,
        transcript_tx: broadcast::Sender<crate::real_time_transcriber::TranscriptionMessage>,
    ) -> tokio::task::JoinHandle<()> {
        let backend = self.backend.clone();
        let backend_ready = self.backend_ready.clone();
        let language = self.language.clone();
        let app_config = self.app_config.clone();
        let running = self.running.clone();
        let transcription_done_tx = self.transcription_done_tx.clone();
        let transcription_stats = self.transcription_stats.clone();
        let audio_visualization_data = self.audio_visualization_data.clone();

        let log_stats_enabled = app_config.debug_config.log_stats_enabled;

        // Spawn a dedicated task for transcription
        tokio::spawn(async move {
            tracing::info!("Transcription task started");

            // Wait for backend to be ready before processing segments
            tracing::info!("Waiting for transcription backend to initialize...");
            let warn_interval = std::time::Duration::from_secs(10);
            let mut last_warn = std::time::Instant::now();

            while !backend_ready.load(Ordering::Relaxed) {
                if !running.load(Ordering::Relaxed) {
                    tracing::info!(
                        "Transcription task shutting down before backend initialization"
                    );
                    return;
                }

                if last_warn.elapsed() >= warn_interval {
                    tracing::warn!(
                        "Backend is still initializing (>{}s); continuing to wait",
                        warn_interval.as_secs()
                    );
                    last_warn = std::time::Instant::now();
                }

                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }

            tracing::info!("Backend ready, starting transcription processing");

            // When recording is false, no segments are received from AudioProcessor,
            // so this task naturally idles until recording is resumed
            loop {
                // Check if we should shut down
                if !running.load(Ordering::Relaxed) {
                    // Before shutting down, process any remaining segments
                    while let Ok(segment) = segment_rx.try_recv() {
                        Self::process_segment(
                            segment,
                            backend.clone(),
                            language.clone(),
                            app_config.clone(),
                            transcription_stats.clone(),
                            audio_visualization_data.clone(),
                            transcript_tx.clone(),
                            log_stats_enabled,
                        )
                        .await;
                    }
                    break;
                }

                // Block on receiving segments without timeout - this is much more efficient
                match segment_rx.recv().await {
                    Some(segment) => {
                        // Wait for backend to be ready (e.g. during model reload)
                        while !backend_ready.load(Ordering::Relaxed) {
                            if !running.load(Ordering::Relaxed) {
                                break;
                            }
                            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                        }
                        Self::process_segment(
                            segment,
                            backend.clone(),
                            language.clone(),
                            app_config.clone(),
                            transcription_stats.clone(),
                            audio_visualization_data.clone(),
                            transcript_tx.clone(),
                            log_stats_enabled,
                        )
                        .await;
                    }
                    None => {
                        // Channel closed
                        break;
                    }
                }
            }

            tracing::info!("Transcription task shutting down");
            let _ = transcription_done_tx.send(());
        })
    }

    /// Process a manual mode segment with specialized handling for longer audio
    fn process_manual_segment(
        backend: &Arc<Mutex<Option<Arc<TranscriptionBackend>>>>,
        segment: &AudioSegment,
        language: &str,
        app_config: &SpeechConfig,
        stats: &Arc<Mutex<TranscriptionStats>>,
        audio_visualization_data: &Arc<RwLock<AudioVisualizationData>>,
    ) -> String {
        let start_time = Instant::now();
        let duration = segment.end_time - segment.start_time;

        tracing::info!("Processing manual segment: {:.2}s of audio", duration);

        // Check if user wants to disable chunking entirely (experimental mode)
        if app_config.manual_mode_config.disable_chunking {
            tracing::info!(
                "EXPERIMENTAL: Processing entire recording as single segment (chunking disabled)"
            );
            let result = Self::transcribe_segment(
                backend,
                segment,
                language,
                app_config,
                stats,
                audio_visualization_data,
                None,
            );
            let processing_time = start_time.elapsed();
            tracing::info!(
                "Manual segment processing completed in {:.2}s",
                processing_time.as_secs_f32()
            );
            return result;
        }

        // For very long segments, we might want to split them into smaller chunks
        // to avoid memory issues and improve processing reliability
        let chunk_threshold = {
            let backend_max = backend
                .lock()
                .as_ref()
                .and_then(|b| b.capabilities().max_audio_duration);
            match backend_max {
                Some(max) => max as f64,
                None => app_config.manual_mode_config.chunk_duration_seconds as f64,
            }
        };
        if duration >= chunk_threshold {
            tracing::info!("Large manual segment detected, processing in chunks...");
            return Self::process_large_manual_segment(
                backend,
                segment,
                language,
                app_config,
                stats,
                audio_visualization_data,
            );
        }

        // Process normally for smaller manual segments
        let result = Self::transcribe_segment(
            backend,
            segment,
            language,
            app_config,
            stats,
            audio_visualization_data,
            None,
        );

        let processing_time = start_time.elapsed();
        tracing::info!(
            "Manual segment processing completed in {:.2}s",
            processing_time.as_secs_f32()
        );

        result
    }

    /// Process very large manual segments by splitting into chunks using VAD-guided boundaries.
    /// Finds natural pauses in speech to split at, avoiding mid-word cuts.
    /// Falls back to time-based splitting if no pauses found or continuous speech exceeds limits.
    fn process_large_manual_segment(
        backend: &Arc<Mutex<Option<Arc<TranscriptionBackend>>>>,
        segment: &AudioSegment,
        language: &str,
        app_config: &SpeechConfig,
        stats: &Arc<Mutex<TranscriptionStats>>,
        audio_visualization_data: &Arc<RwLock<AudioVisualizationData>>,
    ) -> String {
        let sample_rate = segment.sample_rate;
        let max_chunk_seconds = backend
            .lock()
            .as_ref()
            .and_then(|b| b.capabilities().max_audio_duration)
            .unwrap_or(app_config.manual_mode_config.chunk_duration_seconds);
        let max_chunk_samples = (max_chunk_seconds as f64 * sample_rate as f64).round() as usize;
        let total_len = segment.samples.len();

        tracing::info!(
            "Processing {:.1}s of audio with VAD-guided chunking (max chunk: {:.1}s)",
            total_len as f64 / sample_rate as f64,
            max_chunk_seconds
        );

        // Find natural pause points using VAD
        let pause_points = find_pause_points(&segment.samples, sample_rate);

        if !pause_points.is_empty() {
            tracing::info!(
                "Found {} natural pause point(s) in audio",
                pause_points.len()
            );
        } else {
            tracing::info!("No natural pauses detected, using time-based chunking");
        }

        // Build chunk ranges using pause points as preferred boundaries
        let chunk_ranges =
            Self::build_vad_guided_chunks(total_len, max_chunk_samples, &pause_points, sample_rate);

        tracing::info!("Split into {} chunk(s)", chunk_ranges.len());

        // Transcribe each chunk with prompt conditioning for continuity
        let mut transcriptions: Vec<String> = Vec::new();
        let mut previous_text = String::new();
        const PROMPT_CONTEXT_WORDS: usize = 30;

        for (chunk_idx, (start_idx, end_idx)) in chunk_ranges.iter().enumerate() {
            let chunk_audio = segment.samples[*start_idx..*end_idx].to_vec();
            let chunk_start_time = segment.start_time + (*start_idx as f64 / sample_rate as f64);
            let chunk_end_time = segment.start_time + (*end_idx as f64 / sample_rate as f64);

            let chunk_segment = AudioSegment {
                samples: chunk_audio,
                start_time: chunk_start_time,
                end_time: chunk_end_time,
                sample_rate,
                session_id: segment.session_id.clone(),
                is_manual: segment.is_manual,
            };

            tracing::info!(
                "Processing chunk {}/{} ({:.1}s - {:.1}s, {:.1}s duration)",
                chunk_idx + 1,
                chunk_ranges.len(),
                chunk_start_time,
                chunk_end_time,
                chunk_end_time - chunk_start_time
            );

            // Use previous transcription as prompt for continuity (whisper.cpp only)
            let prompt = if !previous_text.is_empty() {
                Some(extract_prompt_context(&previous_text, PROMPT_CONTEXT_WORDS))
            } else {
                None
            };

            let chunk_transcription = Self::transcribe_segment(
                backend,
                &chunk_segment,
                language,
                app_config,
                stats,
                audio_visualization_data,
                prompt.as_deref(),
            );

            if !chunk_transcription.is_empty() {
                let trimmed = chunk_transcription.trim().to_string();
                if !trimmed.is_empty() {
                    transcriptions.push(trimmed.clone());
                    previous_text = trimmed;
                }
            }
        }

        // Combine all chunk transcriptions
        transcriptions.join(" ")
    }

    /// Build chunk ranges using VAD pause points as preferred split locations.
    /// Tries to split at natural pauses, falls back to time-based if needed.
    fn build_vad_guided_chunks(
        total_len: usize,
        max_chunk_samples: usize,
        pause_points: &[usize],
        sample_rate: usize,
    ) -> Vec<(usize, usize)> {
        let mut chunk_ranges: Vec<(usize, usize)> = Vec::new();
        let mut start_idx = 0;

        // Minimum chunk size (avoid tiny fragments)
        let min_chunk_samples = sample_rate * 2; // 2 seconds minimum

        while start_idx < total_len {
            let remaining = total_len - start_idx;

            // If remaining audio fits in one chunk, take it all
            if remaining <= max_chunk_samples {
                chunk_ranges.push((start_idx, total_len));
                break;
            }

            // Find the best pause point within our max chunk size
            let max_end = start_idx + max_chunk_samples;
            let best_pause = pause_points
                .iter()
                .filter(|&&p| p > start_idx + min_chunk_samples && p <= max_end)
                .max(); // Take the latest pause within range (largest chunk)

            let end_idx = if let Some(&pause) = best_pause {
                // Split at the natural pause
                tracing::info!(
                    "  Splitting at pause at {:.1}s",
                    pause as f64 / sample_rate as f64
                );
                pause
            } else {
                // No pause found, fall back to max chunk size
                max_end.min(total_len)
            };

            chunk_ranges.push((start_idx, end_idx));
            start_idx = end_idx;
        }

        // Merge trailing tiny chunk into previous if too short
        if chunk_ranges.len() > 1 {
            if let Some(&(last_start, last_end)) = chunk_ranges.last() {
                let last_len = last_end - last_start;
                if last_len < min_chunk_samples {
                    let len = chunk_ranges.len();
                    if let Some(prev_range) = chunk_ranges.get_mut(len - 2) {
                        let merged_len = last_end - prev_range.0;
                        // Only merge if result is within reasonable limits (45s)
                        if merged_len <= sample_rate * 45 {
                            prev_range.1 = last_end;
                            chunk_ranges.pop();
                        }
                    }
                }
            }
        }

        chunk_ranges
    }
}

#[cfg(test)]
mod tests {
    use super::TranscriptionProcessor;

    #[test]
    fn build_vad_guided_chunks_prefers_latest_pause_within_limit() {
        let sample_rate = 10;
        let ranges =
            TranscriptionProcessor::build_vad_guided_chunks(100, 40, &[25, 35, 70], sample_rate);

        assert_eq!(ranges, vec![(0, 35), (35, 70), (70, 100)]);
    }

    #[test]
    fn build_vad_guided_chunks_falls_back_to_max_size_without_pause() {
        let sample_rate = 10;
        let ranges = TranscriptionProcessor::build_vad_guided_chunks(105, 40, &[], sample_rate);

        assert_eq!(ranges, vec![(0, 40), (40, 80), (80, 105)]);
    }

    #[test]
    fn build_vad_guided_chunks_merges_tiny_trailing_chunk() {
        let sample_rate = 10;
        let ranges = TranscriptionProcessor::build_vad_guided_chunks(85, 40, &[], sample_rate);

        assert_eq!(ranges, vec![(0, 40), (40, 85)]);
    }
}
