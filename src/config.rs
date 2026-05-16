use crate::backend::BackendConfig;
#[cfg(feature = "vad-silero")]
use crate::silero_audio_processor::VadConfig as SileroVadConfig;
#[cfg(feature = "backend-ctranslate2")]
use ct2rs::WhisperOptions;
use serde::{Deserialize, Serialize};

/// Audio sample rate in Hz - hardcoded to 16000 (required by Silero VAD)
pub const SAMPLE_RATE: usize = 16000;

/// Audio processor configuration parameters for general audio processing
/// This is separate from the VAD-specific settings
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioProcessorConfig {
    /// The global buffer size used throughout the application
    /// This is the fundamental audio processing block size in samples
    /// Also used for visualization sample count
    pub buffer_size: usize,
}

impl Default for AudioProcessorConfig {
    fn default() -> Self {
        Self { buffer_size: 1024 }
    }
}

/// Configuration for general core settings
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GeneralConfig {
    /// Main model to use for transcription
    pub model: String,
    /// Language for transcription
    pub language: String,
    /// Transcription mode: "realtime" or "manual"
    pub transcription_mode: String,
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            model: "small.en".to_string(),
            language: "en".to_string(),
            transcription_mode: "manual".to_string(),
        }
    }
}

/// Configuration for real-time transcription mode
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RealtimeModeConfig {
    /// Maximum audio buffer duration in seconds for VAD history
    pub max_buffer_duration_sec: f32,

    /// Maximum number of speech segments to keep in buffer
    pub max_segment_count: usize,
}

impl Default for RealtimeModeConfig {
    fn default() -> Self {
        Self {
            max_buffer_duration_sec: 30.0,
            max_segment_count: 20,
        }
    }
}

/// Configuration for manual transcription mode
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ManualModeConfig {
    /// Maximum recording duration in seconds (default: 120)
    /// Buffer size is calculated as: max_recording_duration_secs * sample_rate
    pub max_recording_duration_secs: u32,

    /// Whether to clear previous transcript when starting new session
    pub clear_on_new_session: bool,

    /// Duration of each chunk in seconds (default: 29.0)
    /// Note: 29s avoids edge case where duration == chunk_size hits token limits
    pub chunk_duration_seconds: f32,

    /// Whether to enable chunk overlap for manual mode transcription (default: true)
    /// When enabled, uses small overlap between chunks to catch boundary words
    /// Overlap amount is controlled by chunk_overlap_seconds
    pub enable_chunk_overlap: bool,

    /// Overlap duration in seconds between chunks (default: 0.5)
    /// Only used when enable_chunk_overlap is true
    /// Recommended range: 0.1 to 1.0 seconds (avoid 2+ seconds due to hallucination)
    pub chunk_overlap_seconds: f32,

    /// EXPERIMENTAL: Disable chunking for manual mode transcription (default: false)
    /// When enabled, processes entire recording as single segment (no chunk limit)
    /// Note: May consume more memory for very long recordings
    /// Note: many transcription models are trained on short chunks, so very long audio may have issues
    pub disable_chunking: bool,
}

impl Default for ManualModeConfig {
    fn default() -> Self {
        Self {
            max_recording_duration_secs: 120,
            clear_on_new_session: true,
            chunk_duration_seconds: 29.0, // 29s avoids edge case at exactly 30s boundary
            enable_chunk_overlap: true,   // Enable overlap by default
            chunk_overlap_seconds: 2.0,   // 2.0 second overlap (matches packaged config)
            disable_chunking: false,      // Chunking enabled by default
        }
    }
}

/// Configuration for debugging and development
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DebugConfig {
    /// Whether to log statistics
    pub log_stats_enabled: bool,
    /// Whether to save manual mode audio to WAV files for debugging
    pub save_manual_audio_debug: bool,
    /// Directory to save debug recordings (default: "recordings")
    pub recording_dir: String,
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            log_stats_enabled: false,
            save_manual_audio_debug: false,
            recording_dir: "recordings".to_string(),
        }
    }
}

/// Configuration for transcription post-processing
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PostProcessConfig {
    /// Enable post-processing of transcriptions
    pub enabled: bool,
    /// Remove leading dashes from transcriptions
    pub remove_leading_dashes: bool,
    /// Remove trailing dashes from transcriptions
    pub remove_trailing_dashes: bool,
    /// Normalize whitespace (collapse multiple spaces, remove leading/trailing)
    pub normalize_whitespace: bool,
}

impl Default for PostProcessConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            remove_leading_dashes: true,
            remove_trailing_dashes: true,
            normalize_whitespace: true,
        }
    }
}

/// VAD sensitivity presets for different acoustic environments
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum VadSensitivity {
    /// Less sensitive - reduces false positives in noisy environments
    Low,
    /// Balanced - good for most environments (default)
    #[default]
    Medium,
    /// More sensitive - catches quiet speech, may trigger on background noise
    High,
}

impl VadSensitivity {
    /// Get the speech detection threshold for this sensitivity level
    pub fn threshold(&self) -> f32 {
        match self {
            VadSensitivity::Low => 0.15,
            VadSensitivity::Medium => 0.10,
            VadSensitivity::High => 0.05,
        }
    }

    /// Get the speech end threshold (hysteresis) for this sensitivity level
    pub fn speech_end_threshold(&self) -> f32 {
        match self {
            VadSensitivity::Low => 0.12,
            VadSensitivity::Medium => 0.08,
            VadSensitivity::High => 0.03,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct SpeechConfig {
    /// General core configuration
    pub general_config: GeneralConfig,

    /// Backend configuration (includes backend selection)
    pub backend_config: BackendConfig,

    /// Audio processing configuration
    pub audio_processor_config: AudioProcessorConfig,

    /// Real-time transcription mode configuration
    pub realtime_mode_config: RealtimeModeConfig,

    /// Manual transcription mode configuration
    pub manual_mode_config: ManualModeConfig,

    /// Voice Activity Detection configuration
    pub vad_config: VadConfigSerde,

    /// Common transcription options shared across all backends
    pub common_transcription_options: CommonTranscriptionOptions,

    /// CTranslate2-specific options
    pub ctranslate2_options: CT2Options,

    /// Whisper.cpp-specific options
    pub whisper_cpp_options: WhisperCppOptions,

    /// Moonshine-specific options
    pub moonshine_options: MoonshineOptions,

    /// Parakeet TDT-specific options
    pub parakeet_options: ParakeetOptions,

    /// Debug and development configuration
    pub debug_config: DebugConfig,

    /// Transcription post-processing configuration
    pub post_process_config: PostProcessConfig,

    /// Deprecated legacy field - use backend_config instead
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compute_type: Option<String>,

    /// Deprecated legacy field - use backend_config instead
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
}

/// Common transcription options shared across all backends
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CommonTranscriptionOptions {
    /// Beam search width (1 = greedy/fastest, higher = more accurate but slower)
    pub beam_size: usize,
    /// Beam search patience factor
    pub patience: f32,
}

impl Default for CommonTranscriptionOptions {
    fn default() -> Self {
        Self {
            beam_size: 5,
            patience: 1.0,
        }
    }
}

/// CTranslate2-specific transcription options
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CT2Options {
    /// Penalty for repeated tokens
    pub repetition_penalty: f32,
}

impl Default for CT2Options {
    fn default() -> Self {
        Self {
            repetition_penalty: 1.25,
        }
    }
}

/// Whisper.cpp internal thresholds - hardcoded to whisper.cpp defaults
pub const WHISPER_ENTROPY_THOLD: f32 = 2.4;
pub const WHISPER_LOGPROB_THOLD: f32 = -1.0;
pub const WHISPER_NO_SPEECH_THOLD: f32 = 0.6;

/// Whisper.cpp-specific transcription options
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WhisperCppOptions {
    pub temperature: f32,
    pub suppress_blank: bool,
    pub no_context: bool,
    pub max_tokens: i32,
    /// Initial prompt to condition the model (used internally for chunk continuity)
    #[serde(skip)]
    pub initial_prompt: Option<String>,
}

impl Default for WhisperCppOptions {
    fn default() -> Self {
        Self {
            temperature: 0.2,     // Gentle sampling bump to match packaged config
            suppress_blank: true, // Skip blank segments
            no_context: true,     // Disable context to prevent double transcriptions
            max_tokens: 0,        // No limit
            initial_prompt: None, // Set dynamically for chunk continuity
        }
    }
}

/// Moonshine-specific options
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct MoonshineOptions {
    /// Whether to use cached decoder (prefill + decode steps) for faster inference
    pub enable_cache: bool,
}

/// Parakeet TDT-specific options
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ParakeetOptions {}

/// Configuration for Voice Activity Detection
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VadConfigSerde {
    /// VAD sensitivity preset for different acoustic environments
    /// Low: Reduces false positives in noisy environments
    /// Medium: Balanced for most environments (default)
    /// High: Catches quiet speech, may trigger on background noise
    pub sensitivity: VadSensitivity,
    /// Number of frames before confirming speech
    pub hangbefore_frames: usize,
    /// Number of frames after speech before ending segment
    pub hangover_frames: usize,
    /// Number of non-speech frames to tolerate in PossibleSpeech before giving up
    pub silence_tolerance_frames: usize,
    /// Exponential moving average smoothing factor (0.0-1.0)
    pub speech_prob_smoothing: f32,
}

impl Default for VadConfigSerde {
    fn default() -> Self {
        Self {
            sensitivity: VadSensitivity::default(), // Medium sensitivity (threshold: 0.10, speech_end: 0.08)
            hangbefore_frames: 5,                   // 50ms - capture more lead-in audio
            hangover_frames: 30,                    // 300ms - keep more trailing audio
            silence_tolerance_frames: 8,            // 80ms - tolerate more pauses
            speech_prob_smoothing: 0.3,             // EMA smoothing factor (production standard)
        }
    }
}

#[cfg(feature = "vad-silero")]
impl SileroVadConfig {
    pub fn from_config(
        vad_config: &VadConfigSerde,
        realtime_config: &RealtimeModeConfig,
        _buffer_size: usize,
        sample_rate: usize,
    ) -> Self {
        Self {
            threshold: vad_config.sensitivity.threshold(),
            frame_size: 512,
            sample_rate,
            hangbefore_frames: vad_config.hangbefore_frames,
            hangover_frames: vad_config.hangover_frames,
            hop_samples: (sample_rate as f32 * 0.01) as usize, // 10ms hop calculated from sample_rate
            max_buffer_duration: (realtime_config.max_buffer_duration_sec * sample_rate as f32)
                as usize,
            max_segment_count: realtime_config.max_segment_count,
            silence_tolerance_frames: vad_config.silence_tolerance_frames,
            speech_end_threshold: vad_config.sensitivity.speech_end_threshold(),
            speech_prob_smoothing: vad_config.speech_prob_smoothing,
        }
    }
}

#[cfg(feature = "vad-silero")]
impl From<(VadConfigSerde, RealtimeModeConfig, usize, usize)> for SileroVadConfig {
    fn from(
        (config, realtime_config, _buffer_size, sample_rate): (
            VadConfigSerde,
            RealtimeModeConfig,
            usize,
            usize,
        ),
    ) -> Self {
        Self {
            threshold: config.sensitivity.threshold(),
            frame_size: 512,
            sample_rate,
            hangbefore_frames: config.hangbefore_frames,
            hangover_frames: config.hangover_frames,
            hop_samples: (sample_rate as f32 * 0.01) as usize, // 10ms hop calculated from sample_rate
            max_buffer_duration: (realtime_config.max_buffer_duration_sec * sample_rate as f32)
                as usize,
            max_segment_count: realtime_config.max_segment_count,
            silence_tolerance_frames: config.silence_tolerance_frames,
            speech_end_threshold: config.sensitivity.speech_end_threshold(),
            speech_prob_smoothing: config.speech_prob_smoothing,
        }
    }
}

impl SpeechConfig {
    /// Migrate legacy compute_type/device fields to new backend_config
    pub fn migrate_legacy_config(&mut self) {
        if let (Some(compute_type), Some(device)) = (&self.compute_type, &self.device) {
            let is_default_config = self.backend_config.threads == num_cpus::get().min(4)
                && !self.backend_config.gpu_enabled;

            if is_default_config {
                tracing::info!(
                    "Migrating legacy config fields (compute_type={}, device={}) to backend_config",
                    compute_type,
                    device
                );

                #[cfg(feature = "backend-ctranslate2")]
                {
                    self.backend_config = crate::backend::ctranslate2::migrate_legacy_config(
                        compute_type,
                        device,
                        None,
                    );
                }
                #[cfg(not(feature = "backend-ctranslate2"))]
                {
                    tracing::warn!(
                        "Legacy CTranslate2 config fields are present, but `backend-ctranslate2` is disabled"
                    );
                }
                self.compute_type = None;
                self.device = None;
            }
        }

        // Ensure whisper.cpp does not reuse context across sessions (prevents duplicate transcriptions)
        if !self.whisper_cpp_options.no_context {
            tracing::info!(
                "Enabling whisper_cpp_options.no_context to prevent cross-session duplication"
            );
            self.whisper_cpp_options.no_context = true;
        }

        // Bring legacy configs up to current default temperature if they were using the old default
        if (self.whisper_cpp_options.temperature - 0.0).abs() < f32::EPSILON {
            self.whisper_cpp_options.temperature = 0.2;
        }
    }
}

#[cfg(feature = "backend-ctranslate2")]
impl CT2Options {
    /// Convert to ct2rs::WhisperOptions, combining with common options
    pub fn to_whisper_options(
        &self,
        common_options: &CommonTranscriptionOptions,
    ) -> WhisperOptions {
        WhisperOptions {
            beam_size: common_options.beam_size,
            patience: common_options.patience,
            repetition_penalty: self.repetition_penalty,
            ..Default::default()
        }
    }
}

#[cfg(feature = "backend-ctranslate2")]
pub fn migrate_legacy_ctranslate2_config(
    compute_type: &str,
    device: &str,
    threads: Option<usize>,
) -> crate::backend::BackendConfig {
    crate::backend::ctranslate2::migrate_legacy_config(compute_type, device, threads)
}
