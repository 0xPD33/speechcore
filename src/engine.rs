use crate::config::SpeechConfig;
use crate::feedback::FeedbackSink;
use crate::{init_all_models, RealTimeTranscriber, TranscriptionMessage};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast;

/// Final transcript returned by the high-level manual transcription API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transcript {
    /// Final transcript text.
    pub text: String,
    /// Session that produced this transcript.
    pub session_id: Option<String>,
    /// Model name from the engine configuration.
    pub model: String,
    /// Backend used by the engine configuration.
    pub backend: crate::BackendType,
}

impl Transcript {
    fn from_message(message: TranscriptionMessage, config: &SpeechConfig) -> Self {
        Self {
            text: message.text,
            session_id: message.session_id,
            model: config.general_config.model.clone(),
            backend: config.backend_config.backend,
        }
    }
}

/// High-level push-to-talk speech engine for applications that do not need to
/// wire the lower-level realtime coordinator directly.
///
/// `SpeechEngine` owns model provisioning, transcriber startup, manual session
/// commands, and transcript collection. Use [`RealTimeTranscriber`] directly
/// when an application needs full realtime UI state, backend reload controls,
/// or low-level audio visualization handles.
pub struct SpeechEngine {
    transcriber: RealTimeTranscriber,
    transcript_rx: broadcast::Receiver<TranscriptionMessage>,
    config: SpeechConfig,
    active_session_id: Option<String>,
    transcript_timeout: Duration,
}

impl SpeechEngine {
    /// Create and start a manual speech engine, provisioning the configured
    /// model if necessary.
    pub async fn new(config: SpeechConfig) -> Result<Self, anyhow::Error> {
        Self::with_feedback(config, None).await
    }

    /// Create and start a manual speech engine with an app-owned feedback hook.
    pub async fn with_feedback(
        mut config: SpeechConfig,
        feedback_sink: Option<Arc<dyn FeedbackSink>>,
    ) -> Result<Self, anyhow::Error> {
        config.general_config.transcription_mode = "manual".to_string();
        let (model_path, _) = init_all_models(
            Some(&config.general_config.model),
            config.backend_config.backend,
            &config.backend_config.quantization_level,
        )
        .await?;

        Self::from_model_path(model_path, config, feedback_sink)
    }

    /// Create and start a manual speech engine with an already-resolved model
    /// path.
    pub fn from_model_path(
        model_path: PathBuf,
        mut config: SpeechConfig,
        feedback_sink: Option<Arc<dyn FeedbackSink>>,
    ) -> Result<Self, anyhow::Error> {
        config.general_config.transcription_mode = "manual".to_string();

        let mut transcriber = RealTimeTranscriber::new(model_path, config.clone(), feedback_sink)?;
        let transcript_rx = transcriber.get_transcript_rx();
        transcriber.start()?;

        Ok(Self {
            transcriber,
            transcript_rx,
            config,
            active_session_id: None,
            transcript_timeout: Duration::from_secs(60),
        })
    }

    /// Override how long [`stop_and_transcribe`](Self::stop_and_transcribe)
    /// waits for the final transcript.
    pub fn set_transcript_timeout(&mut self, timeout: Duration) {
        self.transcript_timeout = timeout;
    }

    /// Start recording a new manual session.
    pub async fn start_session(&mut self) -> Result<String, anyhow::Error> {
        let session_id = self.transcriber.start_manual_session().await?;
        self.active_session_id = Some(session_id.clone());
        Ok(session_id)
    }

    /// Stop the active manual session and wait for its final transcript.
    pub async fn stop_and_transcribe(&mut self) -> Result<Transcript, anyhow::Error> {
        let expected_session_id = self.active_session_id.take();
        self.transcriber.stop_manual_session().await?;

        let timeout = self.transcript_timeout;
        let config = self.config.clone();
        let transcript_rx = &mut self.transcript_rx;

        tokio::time::timeout(timeout, async move {
            loop {
                let message = transcript_rx.recv().await.map_err(|e| {
                    anyhow::anyhow!("Transcript channel closed before final transcript: {}", e)
                })?;

                if expected_session_id.is_none() || message.session_id == expected_session_id {
                    return Ok(Transcript::from_message(message, &config));
                }
            }
        })
        .await
        .map_err(|_| anyhow::anyhow!("Timed out waiting for final transcript"))?
    }

    /// Cancel the current manual session without producing a transcript.
    pub async fn cancel_session(&mut self) -> Result<(), anyhow::Error> {
        self.active_session_id = None;
        self.transcriber.cancel_manual_session().await
    }

    /// Subscribe to raw transcript messages for applications that want to
    /// observe the stream directly.
    pub fn transcript_messages(&self) -> broadcast::Receiver<TranscriptionMessage> {
        self.transcriber.get_transcript_rx()
    }

    /// Access the lower-level transcriber for advanced integrations.
    pub fn transcriber(&self) -> &RealTimeTranscriber {
        &self.transcriber
    }

    /// Mutably access the lower-level transcriber for advanced integrations.
    pub fn transcriber_mut(&mut self) -> &mut RealTimeTranscriber {
        &mut self.transcriber
    }

    /// Shut down the engine and release audio/backend resources.
    pub async fn shutdown(&mut self) -> Result<(), anyhow::Error> {
        self.transcriber.shutdown().await
    }
}
