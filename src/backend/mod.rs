//! Multi-backend transcription abstraction
//!
//! This module provides a unified interface for different transcription backends,
//! allowing runtime selection between CTranslate2, whisper.cpp, and future backends.
//!
//! # Architecture
//! - Enum-based dispatch for zero-cost abstraction
//! - Backend-agnostic configuration with capability negotiation
//! - Single backend loaded at runtime
//! - Maintains all current features (stats, options, error handling)

#[cfg(feature = "backend-ctranslate2")]
pub mod ctranslate2;
pub mod factory;
#[cfg(feature = "backend-moonshine")]
pub mod moonshine;
#[cfg(any(
    feature = "backend-moonshine",
    feature = "backend-parakeet",
    feature = "vad-silero"
))]
pub mod onnx_utils;
#[cfg(feature = "backend-parakeet")]
pub mod parakeet;
pub mod traits;
#[cfg(feature = "backend-whisper-cpp")]
pub mod whisper_cpp;

use serde::{Deserialize, Serialize};
use std::fmt;

/// Identifies which backend implementation to use
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendType {
    /// CTranslate2 backend (current default)
    #[serde(alias = "ct2", alias = "ctranslate2")]
    CTranslate2,

    /// whisper.cpp backend via whisper-rs bindings
    #[serde(alias = "whisper-cpp", alias = "whispercpp")]
    WhisperCpp,

    /// Moonshine ONNX backend
    #[serde(alias = "moonshine")]
    Moonshine,

    /// NVIDIA Parakeet backend (future)
    Parakeet,
}

#[allow(clippy::derivable_impls)]
impl Default for BackendType {
    fn default() -> Self {
        #[cfg(feature = "backend-ctranslate2")]
        {
            Self::CTranslate2
        }
        #[cfg(all(not(feature = "backend-ctranslate2"), feature = "backend-moonshine"))]
        {
            Self::Moonshine
        }
        #[cfg(all(
            not(feature = "backend-ctranslate2"),
            not(feature = "backend-moonshine"),
            feature = "backend-parakeet"
        ))]
        {
            Self::Parakeet
        }
        #[cfg(all(
            not(feature = "backend-ctranslate2"),
            not(feature = "backend-moonshine"),
            not(feature = "backend-parakeet"),
            feature = "backend-whisper-cpp"
        ))]
        {
            Self::WhisperCpp
        }
        #[cfg(all(
            not(feature = "backend-ctranslate2"),
            not(feature = "backend-moonshine"),
            not(feature = "backend-parakeet"),
            not(feature = "backend-whisper-cpp")
        ))]
        {
            Self::CTranslate2
        }
    }
}

impl fmt::Display for BackendType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendType::CTranslate2 => write!(f, "ctranslate2"),
            BackendType::WhisperCpp => write!(f, "whisper_cpp"),
            BackendType::Moonshine => write!(f, "moonshine"),
            BackendType::Parakeet => write!(f, "parakeet"),
        }
    }
}

/// Quantization level for model inference
/// Backends interpret this based on their capabilities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum QuantizationLevel {
    /// Highest precision, slower inference
    /// - CTranslate2: FLOAT32 or FLOAT16
    /// - whisper.cpp: f32 or f16
    High,

    /// Balanced precision and speed (default)
    /// - CTranslate2: FLOAT16 or INT8
    /// - whisper.cpp: q5_1 or q8_0
    #[default]
    Medium,

    /// Lowest precision, fastest inference
    /// - CTranslate2: INT8
    /// - whisper.cpp: q4_0 or q5_0
    Low,
}

/// Backend-agnostic configuration for transcription
/// Each backend interprets these generic options based on its capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BackendConfig {
    /// Backend type to use (ctranslate2, whisper_cpp, moonshine, parakeet)
    pub backend: BackendType,

    /// Number of CPU threads to use for inference
    /// Backends may adjust based on their threading model
    pub threads: usize,

    /// Whether to enable GPU acceleration if available
    /// - CTranslate2: CUDA support
    /// - whisper.cpp: CUDA/Metal/Vulkan support
    pub gpu_enabled: bool,

    /// Quantization level for model weights
    /// Backends translate this to their specific quantization formats
    pub quantization_level: QuantizationLevel,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            backend: BackendType::default(),
            threads: num_cpus::get().min(4), // Cap at 4 threads for efficiency
            gpu_enabled: true,               // Default to GPU for better performance
            quantization_level: QuantizationLevel::Medium,
        }
    }
}

/// Backend capabilities reported by each implementation
/// Allows UI and config validation to adapt to backend features
#[derive(Debug, Clone)]
pub struct BackendCapabilities {
    /// Backend name for display
    pub name: &'static str,

    /// Maximum audio duration this backend can handle in seconds
    pub max_audio_duration: Option<f32>,

    /// Supported languages (None = all languages)
    pub supported_languages: Option<Vec<String>>,

    /// Whether this backend supports streaming transcription
    pub supports_streaming: bool,

    /// Whether GPU acceleration is available on this system
    pub gpu_available: bool,
}

/// The unified transcription backend enum
/// Uses enum dispatch for zero-cost abstraction
pub enum TranscriptionBackend {
    /// CTranslate2 backend
    #[cfg(feature = "backend-ctranslate2")]
    CTranslate2(Box<ctranslate2::CT2Backend>),

    /// whisper.cpp backend
    #[cfg(feature = "backend-whisper-cpp")]
    WhisperCpp(Box<whisper_cpp::WhisperCppBackend>),

    /// Moonshine backend
    #[cfg(feature = "backend-moonshine")]
    Moonshine(Box<moonshine::MoonshineBackend>),

    /// Parakeet TDT backend
    #[cfg(feature = "backend-parakeet")]
    Parakeet(Box<parakeet::ParakeetBackend>),
}

impl TranscriptionBackend {
    /// Get the backend type
    pub fn backend_type(&self) -> BackendType {
        match self {
            #[cfg(feature = "backend-ctranslate2")]
            TranscriptionBackend::CTranslate2(_) => BackendType::CTranslate2,
            #[cfg(feature = "backend-whisper-cpp")]
            TranscriptionBackend::WhisperCpp(_) => BackendType::WhisperCpp,
            #[cfg(feature = "backend-moonshine")]
            TranscriptionBackend::Moonshine(_) => BackendType::Moonshine,
            #[cfg(feature = "backend-parakeet")]
            TranscriptionBackend::Parakeet(_) => BackendType::Parakeet,
            #[cfg(all(
                not(feature = "backend-ctranslate2"),
                not(feature = "backend-whisper-cpp"),
                not(feature = "backend-moonshine"),
                not(feature = "backend-parakeet")
            ))]
            _ => unreachable!("no transcription backend variants are enabled"),
        }
    }

    /// Get backend capabilities
    pub fn capabilities(&self) -> BackendCapabilities {
        match self {
            #[cfg(feature = "backend-ctranslate2")]
            TranscriptionBackend::CTranslate2(backend) => backend.capabilities(),
            #[cfg(feature = "backend-whisper-cpp")]
            TranscriptionBackend::WhisperCpp(backend) => backend.capabilities(),
            #[cfg(feature = "backend-moonshine")]
            TranscriptionBackend::Moonshine(backend) => backend.capabilities(),
            #[cfg(feature = "backend-parakeet")]
            TranscriptionBackend::Parakeet(backend) => backend.capabilities(),
            #[cfg(all(
                not(feature = "backend-ctranslate2"),
                not(feature = "backend-whisper-cpp"),
                not(feature = "backend-moonshine"),
                not(feature = "backend-parakeet")
            ))]
            _ => unreachable!("no transcription backend variants are enabled"),
        }
    }
}

// Re-export commonly used types
pub use factory::create_backend;
pub use traits::TranscriptionError;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "backend-ctranslate2")]
    fn default_backend_prefers_ctranslate2_when_available() {
        assert_eq!(BackendType::default(), BackendType::CTranslate2);
        assert_eq!(BackendConfig::default().backend, BackendType::CTranslate2);
    }
}
