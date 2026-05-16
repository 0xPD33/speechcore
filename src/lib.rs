//! Reusable speech-to-text runtime components for Rust applications.
//!
//! `speechcore` provides audio capture, Silero VAD segmentation, backend
//! selection, model provisioning, and transcript streaming without any UI,
//! clipboard, tray, or application-specific enhancement code.
//!
//! Push-to-talk applications should start with [`SpeechEngine`]. Applications
//! that need full realtime state/control can use [`RealTimeTranscriber`].

#[cfg(feature = "audio-capture")]
mod audio_capture;
#[cfg(feature = "runtime")]
mod audio_processor;
pub mod backend;
#[cfg(feature = "runtime")]
mod backend_manager;
pub mod config;
#[cfg(feature = "model-downloads")]
pub mod download;
#[cfg(feature = "runtime")]
mod engine;
pub mod feedback;
pub mod post_processor;
#[cfg(feature = "runtime")]
mod real_time_transcriber;
#[cfg(feature = "vad-silero")]
pub mod silero_audio_processor;
pub mod state;
#[cfg(feature = "runtime")]
mod stats_reporter;
#[cfg(feature = "runtime")]
mod transcription_processor;
pub mod transcription_stats;

#[cfg(feature = "audio-capture")]
pub use audio_capture::AudioCapture;
#[cfg(feature = "runtime")]
pub use audio_processor::AudioProcessor;
pub use backend::traits::TranscriptionResult;
pub use backend::{
    create_backend, BackendCapabilities, BackendConfig, BackendType, QuantizationLevel,
    TranscriptionBackend, TranscriptionError,
};
#[cfg(feature = "runtime")]
pub use backend_manager::{BackendCommand, BackendManager};
#[cfg(feature = "backend-ctranslate2")]
pub use config::migrate_legacy_ctranslate2_config;
pub use config::{
    AudioProcessorConfig, CT2Options, CommonTranscriptionOptions, DebugConfig, GeneralConfig,
    ManualModeConfig, MoonshineOptions, ParakeetOptions, PostProcessConfig, RealtimeModeConfig,
    SpeechConfig, VadConfigSerde, WhisperCppOptions, SAMPLE_RATE,
};
#[cfg(feature = "model-downloads")]
pub use download::{
    init_all_models, model_cache_dir, resolve_model_path, resolve_model_path_with_progress,
};
#[cfg(feature = "runtime")]
pub use engine::{SpeechEngine, Transcript};
pub use feedback::{FeedbackEvent, FeedbackSink};
#[cfg(feature = "runtime")]
pub use real_time_transcriber::{
    ManualSessionCommand, ManualSessionStatus, RealTimeTranscriber, TranscriptionMessage,
    TranscriptionMode,
};
#[cfg(feature = "vad-silero")]
pub use silero_audio_processor::{AudioSegment, SampleRate, SileroVad, VadConfig, VadState};
pub use state::{AudioVisualizationData, BackendStatus, BackendStatusState, ProcessingState};
#[cfg(feature = "runtime")]
pub use stats_reporter::StatsReporter;
#[cfg(feature = "runtime")]
pub use transcription_processor::TranscriptionProcessor;
pub use transcription_stats::TranscriptionStats;

pub mod prelude {
    pub use crate::{
        AudioVisualizationData, BackendConfig, BackendStatus, BackendStatusState, BackendType,
        FeedbackEvent, FeedbackSink, ProcessingState, QuantizationLevel, SpeechConfig,
        TranscriptionBackend, TranscriptionResult, TranscriptionStats,
    };

    #[cfg(feature = "audio-capture")]
    pub use crate::AudioCapture;

    #[cfg(feature = "runtime")]
    pub use crate::{
        AudioProcessor, ManualSessionCommand, RealTimeTranscriber, SpeechEngine, Transcript,
        TranscriptionMessage, TranscriptionMode,
    };

    #[cfg(feature = "vad-silero")]
    pub use crate::{AudioSegment, SileroVad, VadConfig, VadState};
}
