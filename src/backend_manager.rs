use parking_lot::RwLock;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;

use crate::backend::factory::create_backend;
use crate::backend::{BackendConfig, BackendType, TranscriptionBackend};
use crate::state::{AudioVisualizationData, BackendStatus, BackendStatusState, ProcessingState};

pub enum BackendCommand {
    /// Load a backend from a model path that is already on disk.
    Load {
        backend_config: BackendConfig,
        model_path: PathBuf,
    },
    /// Reload the backend with new config.
    /// `model_name` is the user-facing name (e.g. "large-v3-turbo"), not a filesystem path.
    Reload {
        backend_config: BackendConfig,
        model_name: String,
    },
    /// Shutdown the backend manager
    Shutdown,
}

/// Runs every backend load in one queue, so a reload never races the startup
/// load and two models are never in memory at the same time.
pub struct BackendManager {
    backend: Arc<parking_lot::Mutex<Option<Arc<TranscriptionBackend>>>>,
    backend_ready: Arc<AtomicBool>,
    status: Arc<RwLock<BackendStatus>>,
    audio_visualization_data: Arc<RwLock<AudioVisualizationData>>,
    command_tx: mpsc::UnboundedSender<BackendCommand>,
    command_rx: Option<mpsc::UnboundedReceiver<BackendCommand>>,
}

pub(crate) fn backend_display_name(backend: BackendType) -> &'static str {
    match backend {
        BackendType::CTranslate2 => "CTranslate2",
        BackendType::WhisperCpp => "WhisperCpp",
        BackendType::Moonshine => "Moonshine",
        BackendType::Parakeet => "Parakeet",
        BackendType::Nemotron => "Nemotron",
    }
}

impl BackendManager {
    pub fn new(
        backend: Arc<parking_lot::Mutex<Option<Arc<TranscriptionBackend>>>>,
        backend_ready: Arc<AtomicBool>,
        status: Arc<RwLock<BackendStatus>>,
        audio_visualization_data: Arc<RwLock<AudioVisualizationData>>,
    ) -> Self {
        let (command_tx, command_rx) = mpsc::unbounded_channel();

        Self {
            backend,
            backend_ready,
            status,
            audio_visualization_data,
            command_tx,
            command_rx: Some(command_rx),
        }
    }

    pub fn start(&mut self) -> tokio::task::JoinHandle<()> {
        let rx = self.command_rx.take().expect("start called twice");
        let backend = self.backend.clone();
        let backend_ready = self.backend_ready.clone();
        let status = self.status.clone();
        let audio_visualization_data = self.audio_visualization_data.clone();

        tokio::spawn(async move {
            Self::run_command_loop(rx, backend, backend_ready, status, audio_visualization_data)
                .await;
        })
    }

    pub fn command_sender(&self) -> mpsc::UnboundedSender<BackendCommand> {
        self.command_tx.clone()
    }

    pub fn status(&self) -> Arc<RwLock<BackendStatus>> {
        self.status.clone()
    }

    async fn run_command_loop(
        mut rx: mpsc::UnboundedReceiver<BackendCommand>,
        backend: Arc<parking_lot::Mutex<Option<Arc<TranscriptionBackend>>>>,
        backend_ready: Arc<AtomicBool>,
        status: Arc<RwLock<BackendStatus>>,
        audio_visualization_data: Arc<RwLock<AudioVisualizationData>>,
    ) {
        while let Some(command) = rx.recv().await {
            let (backend_config, model_path) = match command {
                BackendCommand::Load {
                    backend_config,
                    model_path,
                } => (backend_config, model_path),
                BackendCommand::Reload {
                    backend_config,
                    model_name,
                } => {
                    let (prev_backend_name, prev_model_name) = {
                        let mut s = status.write();
                        let prev = (s.backend_name.clone(), s.model_name.clone());
                        s.backend_name = backend_display_name(backend_config.backend).to_string();
                        s.model_name = model_name.clone();
                        s.state = BackendStatusState::Loading("Resolving model...".to_string());
                        s.download_progress = None;
                        prev
                    };

                    let status_for_progress = status.clone();
                    let on_progress = move |progress: f64| {
                        let mut s = status_for_progress.write();
                        s.download_progress = Some(progress as f32);
                    };

                    match crate::download::resolve_model_path_with_progress(
                        &model_name,
                        backend_config.backend,
                        &backend_config.quantization_level,
                        Some(&on_progress),
                    )
                    .await
                    {
                        Ok(p) => {
                            status.write().download_progress = None;
                            (backend_config, p)
                        }
                        Err(e) => {
                            // The old backend was not touched, so it still works.
                            let mut s = status.write();
                            s.download_progress = None;
                            s.backend_name = prev_backend_name;
                            s.model_name = prev_model_name;
                            s.state = if backend.lock().is_some() {
                                BackendStatusState::Ready
                            } else {
                                // Ends the loading animation of a first load.
                                audio_visualization_data
                                    .write()
                                    .set_processing_state(ProcessingState::Error);
                                BackendStatusState::NoModel
                            };
                            s.report_error(format!("Model download failed: {}", e));
                            tracing::warn!("BackendManager: Model resolution failed: {}", e);
                            continue;
                        }
                    }
                }
                BackendCommand::Shutdown => {
                    tracing::info!("BackendManager: Shutting down");
                    break;
                }
            };

            Self::load(
                &backend_config,
                &model_path,
                &backend,
                &backend_ready,
                &status,
                &audio_visualization_data,
            )
            .await;
        }
    }

    async fn load(
        backend_config: &BackendConfig,
        model_path: &std::path::Path,
        backend: &parking_lot::Mutex<Option<Arc<TranscriptionBackend>>>,
        backend_ready: &AtomicBool,
        status: &RwLock<BackendStatus>,
        audio_visualization_data: &RwLock<AudioVisualizationData>,
    ) {
        backend_ready.store(false, Ordering::SeqCst);
        *backend.lock() = None;
        status.write().state = BackendStatusState::Loading("Loading backend...".to_string());
        audio_visualization_data
            .write()
            .set_processing_state(ProcessingState::Loading);

        tracing::info!(
            "Loading {} backend with model at {:?} (threads={}, gpu_enabled={}, quantization={:?})",
            backend_config.backend,
            model_path,
            backend_config.threads,
            backend_config.gpu_enabled,
            backend_config.quantization_level
        );

        match create_backend(backend_config.backend, model_path, backend_config).await {
            Ok(new_backend) => {
                let capabilities = new_backend.capabilities();
                tracing::info!(
                    "Backend loaded: name={}, max_audio_duration={:?}, streaming={}",
                    capabilities.name,
                    capabilities.max_audio_duration,
                    capabilities.supports_streaming
                );
                *backend.lock() = Some(Arc::new(new_backend));
                backend_ready.store(true, Ordering::SeqCst);
                status.write().state = BackendStatusState::Ready;
                audio_visualization_data
                    .write()
                    .set_processing_state(ProcessingState::Idle);
            }
            Err(e) => {
                tracing::warn!("BackendManager: Failed to load backend: {}", e);
                let mut s = status.write();
                s.state = BackendStatusState::NoModel;
                s.report_error(format!("Failed to load model: {}", e));
                audio_visualization_data
                    .write()
                    .set_processing_state(ProcessingState::Error);
            }
        }
    }
}
