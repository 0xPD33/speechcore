use parking_lot::{Mutex, RwLock};
use std::collections::HashSet;

/// Manual sessions between Stop and their transcript, and the ones the user cancelled.
#[derive(Debug, Default)]
pub struct SessionLedger {
    /// Stopped sessions whose transcript is not out yet, oldest first.
    awaiting: Vec<String>,
    // ponytail: a session cancelled while recording never finishes, so its id
    // stays here; one short string per cancel.
    cancelled: HashSet<String>,
}

impl SessionLedger {
    pub(crate) fn stopped(&mut self, session_id: String) {
        self.awaiting.push(session_id);
    }

    /// Cancels the recording session, or else the newest one still being transcribed.
    pub(crate) fn cancel(&mut self, recording: Option<String>) -> Option<String> {
        let session_id = recording.or_else(|| self.awaiting.pop())?;
        self.cancelled.insert(session_id.clone());
        Some(session_id)
    }

    /// Marks a session's transcript as out. Returns whether it was cancelled.
    fn finish(&mut self, session_id: &str) -> bool {
        self.awaiting.retain(|id| id != session_id);
        self.cancelled.remove(session_id)
    }
}
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
use crate::state::{AudioVisualizationData, BackendStatus, BackendStatusState, ProcessingState};
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
    language: Arc<RwLock<String>>,
    app_config: Arc<SpeechConfig>,
    running: Arc<AtomicBool>,
    transcription_done_tx: mpsc::UnboundedSender<()>,
    transcription_stats: Arc<Mutex<TranscriptionStats>>,
    audio_visualization_data: Arc<RwLock<AudioVisualizationData>>,
    backend_status: Arc<RwLock<BackendStatus>>,
    session_ledger: Arc<Mutex<SessionLedger>>,
}

impl TranscriptionProcessor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        backend: Arc<Mutex<Option<Arc<TranscriptionBackend>>>>,
        backend_ready: Arc<AtomicBool>,
        language: Arc<RwLock<String>>,
        app_config: Arc<SpeechConfig>,
        running: Arc<AtomicBool>,
        transcription_done_tx: mpsc::UnboundedSender<()>,
        transcription_stats: Arc<Mutex<TranscriptionStats>>,
        audio_visualization_data: Arc<RwLock<AudioVisualizationData>>,
        backend_status: Arc<RwLock<BackendStatus>>,
        session_ledger: Arc<Mutex<SessionLedger>>,
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
            backend_status,
            session_ledger,
        }
    }

    /// Transcribe an audio segment using the backend.
    /// Optionally accepts an initial prompt for chunk continuity (whisper.cpp only; CT2 ignores it).
    /// Returns `None` on failure, after reporting the error to `backend_status`.
    #[allow(clippy::too_many_arguments)]
    fn transcribe_segment(
        backend: &Arc<Mutex<Option<Arc<TranscriptionBackend>>>>,
        segment: &AudioSegment,
        language: &str,
        app_config: &SpeechConfig,
        stats: &Arc<Mutex<TranscriptionStats>>,
        audio_visualization_data: &Arc<RwLock<AudioVisualizationData>>,
        backend_status: &RwLock<BackendStatus>,
        initial_prompt: Option<&str>,
    ) -> Option<String> {
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
            backend_status
                .write()
                .report_error("No model loaded; recording not transcribed");
            return None;
        };

        #[cfg(all(
            not(feature = "backend-ctranslate2"),
            not(feature = "backend-whisper-cpp"),
            not(feature = "backend-moonshine"),
            not(feature = "backend-parakeet")
        ))]
        {
            let _ = backend_ref;
            let _ = (language, stats, initial_prompt);
            {
                let mut audio_data = audio_visualization_data.write();
                audio_data.set_processing_state(ProcessingState::Error);
            }
            backend_status
                .write()
                .report_error("No transcription backend feature enabled");
            None
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
                    // Keep the configured vocabulary ahead of the chunk context.
                    options.initial_prompt = Some(
                        match options.initial_prompt.as_deref().filter(|v| !v.is_empty()) {
                            Some(vocabulary) => format!("{vocabulary} {prompt}"),
                            None => prompt.to_string(),
                        },
                    );
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

                    Some(processed_transcription)
                }
                Err(e) => {
                    tracing::warn!(
                        "Transcription error after {:.2}s: {}",
                        start_time.elapsed().as_secs_f32(),
                        e
                    );
                    {
                        let mut audio_data = audio_visualization_data.write();
                        audio_data.set_processing_state(ProcessingState::Error);
                    }
                    backend_status
                        .write()
                        .report_error(format!("Transcription failed: {e}"));
                    None
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
        backend_status: Arc<RwLock<BackendStatus>>,
        session_ledger: Arc<Mutex<SessionLedger>>,
        manual_assembly: Arc<Mutex<Option<ManualAssembly>>>,
        transcript_tx: broadcast::Sender<crate::real_time_transcriber::TranscriptionMessage>,
        log_stats_enabled: bool,
    ) {
        let segment_info = format!(
            "Segment {:.2}s-{:.2}s",
            segment.start_time, segment.end_time
        );
        let start_time = Instant::now();
        let session_id = segment.session_id.clone();
        let partial = segment.partial;
        let ends_manual_session = segment.is_manual && !segment.partial;

        let processing_result = tokio::task::spawn_blocking(move || {
            if segment.is_manual {
                Self::process_manual_part(
                    &manual_assembly,
                    segment,
                    &backend,
                    &language,
                    &app_config,
                    &stats,
                    &audio_visualization_data,
                    &backend_status,
                )
            } else {
                Self::transcribe_segment(
                    &backend,
                    &segment,
                    &language,
                    &app_config,
                    &stats,
                    &audio_visualization_data,
                    &backend_status,
                    None,
                )
            }
        })
        .await;

        // Checked after transcribing: Cancel may arrive while the backend runs.
        let cancelled = ends_manual_session
            && session_id
                .as_deref()
                .is_some_and(|id| session_ledger.lock().finish(id));

        match processing_result {
            Ok(Some(text)) if !text.is_empty() => {
                if cancelled {
                    tracing::info!("Dropping transcript of cancelled session {:?}", session_id);
                } else if let Err(e) =
                    transcript_tx.send(crate::real_time_transcriber::TranscriptionMessage {
                        text,
                        session_id,
                        is_final: true,
                    })
                {
                    tracing::warn!("Failed to send transcription: {}", e);
                }
            }
            Ok(_) if !partial => tracing::info!("Transcription produced no text"),
            Ok(_) => {}
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
                            let lang = language.read().clone();
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
        let backend_status = self.backend_status.clone();
        let session_ledger = self.session_ledger.clone();
        let manual_assembly = Arc::new(Mutex::new(None));

        let log_stats_enabled = app_config.debug_config.log_stats_enabled;

        // Spawn a dedicated task for transcription
        tokio::spawn(async move {
            tracing::info!("Transcription task started");

            // When recording is false, no segments are received from AudioProcessor,
            // so this task naturally idles until recording is resumed
            loop {
                let segment = if running.load(Ordering::Relaxed) {
                    // Wakes up to notice shutdown; the senders outlive this task.
                    match tokio::time::timeout(
                        std::time::Duration::from_millis(100),
                        segment_rx.recv(),
                    )
                    .await
                    {
                        Ok(Some(segment)) => segment,
                        Ok(None) => break,
                        Err(_) => continue,
                    }
                } else {
                    // Before shutting down, process any remaining segments
                    match segment_rx.try_recv() {
                        Ok(segment) => segment,
                        Err(_) => break,
                    }
                };

                // Wait out a (re)load, but drop the segment when no model will come,
                // so a failed load cannot back up the audio pipeline.
                while !backend_ready.load(Ordering::Relaxed)
                    && running.load(Ordering::Relaxed)
                    && backend_status.read().state != BackendStatusState::NoModel
                {
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                }
                if !backend_ready.load(Ordering::Relaxed) {
                    backend_status
                        .write()
                        .report_error("No model loaded; recording not transcribed");
                    continue;
                }

                let segment_language = language.read().clone();
                Self::process_segment(
                    segment,
                    backend.clone(),
                    segment_language,
                    app_config.clone(),
                    transcription_stats.clone(),
                    audio_visualization_data.clone(),
                    backend_status.clone(),
                    session_ledger.clone(),
                    manual_assembly.clone(),
                    transcript_tx.clone(),
                    log_stats_enabled,
                )
                .await;
            }

            tracing::info!("Transcription task shutting down");
            let _ = transcription_done_tx.send(());
        })
    }

    /// Adds a manual segment to its session's assembly. Parts sent while recording
    /// are transcribed chunk by chunk as they fill up; the final segment
    /// transcribes the rest and returns the text of the whole session.
    #[allow(clippy::too_many_arguments)]
    fn process_manual_part(
        assembly: &Mutex<Option<ManualAssembly>>,
        segment: AudioSegment,
        backend: &Arc<Mutex<Option<Arc<TranscriptionBackend>>>>,
        language: &str,
        app_config: &SpeechConfig,
        stats: &Arc<Mutex<TranscriptionStats>>,
        audio_visualization_data: &Arc<RwLock<AudioVisualizationData>>,
        backend_status: &RwLock<BackendStatus>,
    ) -> Option<String> {
        let mut guard = assembly.lock();
        // Sessions arrive in order, so another session's assembly belongs to a
        // session that never finished (it was cancelled).
        if guard
            .as_ref()
            .is_none_or(|current| current.session_id != segment.session_id)
        {
            *guard = Some(ManualAssembly {
                session_id: segment.session_id.clone(),
                carry: Vec::new(),
                texts: Vec::new(),
            });
        }
        let state = guard.as_mut()?;
        state.carry.extend_from_slice(&segment.samples);

        let chunking = !app_config.manual_mode_config.disable_chunking;
        let transcribe = |samples: Vec<f32>, prompt: Option<&str>| {
            let duration = samples.len() as f64 / segment.sample_rate as f64;
            let chunk = AudioSegment {
                samples,
                start_time: 0.0,
                end_time: duration,
                sample_rate: segment.sample_rate,
                session_id: segment.session_id.clone(),
                is_manual: true,
                partial: false,
            };
            tracing::info!("Transcribing manual chunk of {:.1}s", duration);
            Self::transcribe_segment(
                backend,
                &chunk,
                language,
                app_config,
                stats,
                audio_visualization_data,
                backend_status,
                prompt,
            )
        };

        if chunking {
            let max_chunk_samples = (Self::max_chunk_seconds(backend, app_config) as f64
                * segment.sample_rate as f64)
                .round() as usize;
            let min_chunk_samples = segment.sample_rate * 2;
            // Cut only with 2 s to spare, so the rest never becomes a tiny chunk
            // that the model hallucinates on.
            while state.carry.len() >= max_chunk_samples + min_chunk_samples {
                let pauses =
                    find_pause_points(&state.carry[..max_chunk_samples], segment.sample_rate);
                // The latest natural pause within the limit, else a hard cut.
                let split = pauses
                    .into_iter()
                    .filter(|&pause| pause > min_chunk_samples && pause <= max_chunk_samples)
                    .max()
                    .unwrap_or(max_chunk_samples);
                let chunk: Vec<f32> = state.carry.drain(..split).collect();
                let prompt = state
                    .texts
                    .last()
                    .map(|text| extract_prompt_context(text, PROMPT_CONTEXT_WORDS));
                if let Some(text) = transcribe(chunk, prompt.as_deref()) {
                    let text = text.trim();
                    if !text.is_empty() {
                        state.texts.push(text.to_string());
                    }
                }
            }
        }

        if segment.partial {
            return None;
        }

        let mut state = guard.take()?;
        drop(guard);
        if !state.carry.is_empty() {
            let prompt = state
                .texts
                .last()
                .filter(|_| chunking)
                .map(|text| extract_prompt_context(text, PROMPT_CONTEXT_WORDS));
            if let Some(text) = transcribe(std::mem::take(&mut state.carry), prompt.as_deref()) {
                let text = text.trim();
                if !text.is_empty() {
                    state.texts.push(text.to_string());
                }
            }
        }
        Some(state.texts.join(" "))
    }

    /// Longest audio one backend call gets: the backend's own limit, else the configured chunk.
    fn max_chunk_seconds(
        backend: &Arc<Mutex<Option<Arc<TranscriptionBackend>>>>,
        app_config: &SpeechConfig,
    ) -> f32 {
        backend
            .lock()
            .as_ref()
            .and_then(|b| b.capabilities().max_audio_duration)
            .unwrap_or(app_config.manual_mode_config.chunk_duration_seconds)
    }
}

/// Words of the previous chunk given to whisper.cpp as context for the next one.
const PROMPT_CONTEXT_WORDS: usize = 30;

/// Audio and text of the manual session being transcribed while it records.
struct ManualAssembly {
    session_id: Option<String>,
    /// Audio not yet transcribed.
    carry: Vec<f32>,
    texts: Vec<String>,
}
