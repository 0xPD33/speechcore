use anyhow::{Context, Result};
use reqwest;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tokio::io::AsyncWriteExt;

use crate::backend::{BackendType, QuantizationLevel};

/// Default transcription model to download if none specified
const DEFAULT_WHISPER_MODEL: &str = "openai/whisper-base.en";

/// URL for Silero VAD model
// Pinned to a stable release. `master` is a moving target and has shipped an
// ONNX export that returns ~0 probability for all input (breaking VAD).
const SILERO_VAD_URL: &str =
    "https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx";

/// Default filename for the Silero VAD model
const SILERO_MODEL_FILENAME: &str = "silero_vad.onnx";

const MOONSHINE_REPO: &str = "UsefulSensors/moonshine";
const MOONSHINE_ONNX_DIR: &str = "onnx";
const MOONSHINE_MERGED_DIR: &str = "merged";
const MOONSHINE_MERGED_VARIANT: &str = "float";
/// ONNX model files live in the monorepo at onnx/merged/{model_id}/float/
const MOONSHINE_ONNX_FILES: [&str; 2] = ["encoder_model.onnx", "decoder_model_merged.onnx"];
/// Config files live in standalone repos: UsefulSensors/moonshine-{model_id}
const MOONSHINE_CONFIG_FILES: [&str; 2] = ["tokenizer.json", "preprocessor_config.json"];
const MOONSHINE_REQUIRED_FILES: [&str; 4] = [
    "encoder_model.onnx",
    "decoder_model_merged.onnx",
    "tokenizer.json",
    "preprocessor_config.json",
];

fn ensure_backend_available(_backend_type: BackendType) -> Result<()> {
    #[cfg(not(feature = "backend-ctranslate2"))]
    if matches!(_backend_type, BackendType::CTranslate2) {
        anyhow::bail!(
            "CTranslate2 backend is disabled; enable speechcore feature `backend-ctranslate2`"
        );
    }

    #[cfg(not(feature = "backend-moonshine"))]
    if matches!(_backend_type, BackendType::Moonshine) {
        anyhow::bail!(
            "Moonshine backend is disabled; enable speechcore feature `backend-moonshine`"
        );
    }

    #[cfg(not(feature = "backend-parakeet"))]
    if matches!(_backend_type, BackendType::Parakeet) {
        anyhow::bail!("Parakeet backend is disabled; enable speechcore feature `backend-parakeet`");
    }

    #[cfg(not(feature = "backend-nemotron"))]
    if matches!(_backend_type, BackendType::Nemotron) {
        anyhow::bail!("Nemotron backend is disabled; enable speechcore feature `backend-nemotron`");
    }

    #[cfg(not(feature = "backend-whisper-cpp"))]
    if matches!(_backend_type, BackendType::WhisperCpp) {
        anyhow::bail!(
            "Whisper.cpp backend is disabled; enable speechcore feature `backend-whisper-cpp`"
        );
    }

    Ok(())
}

const PARAKEET_MODEL_FILES: [&str; 4] = [
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "joiner.int8.onnx",
    "tokens.txt",
];

fn parakeet_hf_repo(model_name: &str) -> &'static str {
    match model_name {
        "parakeet-tdt-0.6b-v2" => "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
        _ => "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
    }
}

fn parakeet_dir_name(model_name: &str) -> &'static str {
    match model_name {
        "parakeet-tdt-0.6b-v2" => "parakeet-tdt-v2-int8",
        _ => "parakeet-tdt-v3-int8",
    }
}

/// Enum to represent different model types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelType {
    Whisper,
    Silero,
}

/// Common file names that need to be present in converted CT2 models
const REQUIRED_FILES: [&str; 4] = [
    "model.bin",
    "config.json",
    "tokenizer.json",
    "preprocessor_config.json",
];

/// Get the default model cache directory.
///
/// Set `SPEECHCORE_MODEL_DIR` to override the cache location.
pub fn model_cache_dir() -> Result<PathBuf> {
    let models_dir = if let Ok(path) = std::env::var("SPEECHCORE_MODEL_DIR") {
        PathBuf::from(path)
    } else if let Some(cache_home) = std::env::var_os("XDG_CACHE_HOME") {
        PathBuf::from(cache_home).join("speechcore").join("models")
    } else {
        let home_dir = std::env::var("HOME").context("Failed to get HOME directory")?;
        PathBuf::from(home_dir)
            .join(".cache")
            .join("speechcore")
            .join("models")
    };

    // Create models directory if it doesn't exist
    if !models_dir.exists() {
        tracing::info!("Creating models directory: {:?}", models_dir);
        fs::create_dir_all(&models_dir).context("Failed to create models directory")?;
    }

    Ok(models_dir)
}

fn get_models_dir() -> Result<PathBuf> {
    model_cache_dir()
}

/// Detect if running on NixOS
fn is_nixos() -> bool {
    // Check for /etc/nixos directory which is specific to NixOS
    Path::new("/etc/nixos").exists() || 
    // Check for NIX_PATH environment variable as a fallback
    std::env::var("NIX_PATH").is_ok()
}

/// Check if we're in a nix-shell
fn in_nix_shell() -> bool {
    std::env::var("IN_NIX_SHELL").is_ok()
}

/// Checks if all required model files are present
fn is_model_complete(model_dir: &Path) -> Result<bool> {
    tracing::info!(
        "Checking if model is complete in directory: {:?}",
        model_dir
    );

    for file in REQUIRED_FILES.iter() {
        let file_path = model_dir.join(file);
        tracing::info!("  Checking for file: {:?}", file_path);
        if !file_path.exists() {
            tracing::info!("  Missing file: {:?}", file_path);
            return Ok(false);
        }
    }

    tracing::info!("All required files are present");
    Ok(true)
}

fn is_moonshine_model_complete(model_dir: &Path) -> Result<bool> {
    for file in MOONSHINE_REQUIRED_FILES.iter() {
        if !model_dir.join(file).exists() {
            return Ok(false);
        }
    }
    Ok(true)
}

fn moonshine_model_id(model_name: &str) -> String {
    let simple = model_name.split('/').next_back().unwrap_or(model_name);
    simple
        .strip_prefix("moonshine-")
        .unwrap_or(simple)
        .to_string()
}

fn moonshine_model_dir(models_dir: &Path, model_id: &str) -> PathBuf {
    models_dir.join(format!("moonshine-{}-onnx", model_id))
}

/// Checks if Silero model file exists and is valid
fn is_silero_model_valid(model_path: &Path) -> bool {
    if !model_path.exists() {
        return false;
    }

    // Check file size is reasonable (should be > 10KB)
    match fs::metadata(model_path) {
        Ok(metadata) => metadata.len() > 10_000, // Ensuring it's not an empty or corrupted file
        Err(_) => false,
    }
}

/// Convert the model using ct2-transformers-converter
fn convert_model(model_name: &str, output_dir: &Path) -> Result<()> {
    tracing::info!(
        "Converting model {} to {}",
        model_name,
        output_dir.display()
    );

    // Create output directory if it doesn't exist
    if !output_dir.exists() {
        fs::create_dir_all(output_dir)?;
    }

    // Detect system type
    let on_nixos = is_nixos();
    let in_shell = in_nix_shell();

    tracing::info!(
        "System detection: NixOS={}, In nix-shell={}",
        on_nixos,
        in_shell
    );

    let output_dir_str = output_dir.to_str().ok_or_else(|| {
        anyhow::anyhow!("Model output path contains invalid UTF-8: {:?}", output_dir)
    })?;

    let converter_args = [
        "--force",
        "--model",
        model_name,
        "--output_dir",
        output_dir_str,
        "--copy_files",
        "preprocessor_config.json",
        "tokenizer.json",
    ];

    let status = if on_nixos {
        // On NixOS but not in a shell, try to use the provided shell.nix in model-conversion directory
        tracing::info!("On NixOS: Using model-conversion/shell.nix");

        // nix-shell --command runs in a shell context, so quote arguments to prevent injection
        let quoted_args: Vec<String> = converter_args
            .iter()
            .map(|arg| format!("'{}'", arg.replace('\'', "'\\''")))
            .collect();
        let conversion_script = format!("ct2-transformers-converter {}", quoted_args.join(" "));

        // Get the repository root directory to find model-conversion/shell.nix
        let current_dir = std::env::current_dir()?;
        let model_conversion_shell_nix = current_dir.join("model-conversion/shell.nix");

        if model_conversion_shell_nix.exists() {
            tracing::info!("Found shell.nix at {:?}", model_conversion_shell_nix);
            Command::new("nix-shell")
                .arg(model_conversion_shell_nix.to_str().ok_or_else(|| {
                    anyhow::anyhow!(
                        "shell.nix path contains invalid UTF-8: {:?}",
                        model_conversion_shell_nix
                    )
                })?)
                .arg("--command")
                .arg(&conversion_script)
                .status()
        } else {
            tracing::info!(
                "shell.nix not found at {:?}, trying default nix-shell",
                model_conversion_shell_nix
            );
            Command::new("nix-shell")
                .arg("--command")
                .arg(&conversion_script)
                .status()
        }
    } else {
        // Not on NixOS, run directly without shell
        tracing::info!("Not on NixOS: Running conversion directly");
        Command::new("ct2-transformers-converter")
            .args(converter_args)
            .status()
    }
    .context("Failed to run conversion command")?;

    if !status.success() {
        return Err(anyhow::anyhow!(
            "Model conversion failed with status: {}",
            status
        ));
    }

    tracing::info!("Model conversion completed successfully");
    Ok(())
}

/// Download a file from a URL and save it to the specified path
pub async fn download_file(url: &str, output_path: &Path) -> Result<()> {
    download_file_with_progress(url, output_path, None).await
}

/// Download a file from a URL with an optional progress callback (0.0 to 1.0)
pub async fn download_file_with_progress(
    url: &str,
    output_path: &Path,
    on_progress: Option<&(dyn Fn(f64) + Send + Sync)>,
) -> Result<()> {
    tracing::info!("Downloading file from: {}", url);

    // Create parent directories if they don't exist
    if let Some(parent) = output_path.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent)?;
        }
    }

    // Create a temporary file to download to
    let temp_path = output_path.with_extension("downloading");

    // Perform the download
    let response = reqwest::get(url)
        .await
        .context(format!("Failed to download file from {}", url))?;

    if !response.status().is_success() {
        return Err(anyhow::anyhow!(
            "Failed to download file, status: {}",
            response.status()
        ));
    }

    let total_size = response.content_length().unwrap_or(0);
    let mut file = tokio::fs::File::create(&temp_path)
        .await
        .context(format!("Failed to create file at {:?}", temp_path))?;

    let mut stream = response.bytes_stream();
    let mut downloaded: u64 = 0;

    use futures_util::StreamExt;
    while let Some(item) = stream.next().await {
        let chunk = item.context("Error while downloading file")?;
        file.write_all(&chunk).await?;

        downloaded += chunk.len() as u64;
        if total_size > 0 {
            let progress = (downloaded as f64 / total_size as f64) * 100.0;
            if let Some(cb) = &on_progress {
                cb(progress / 100.0);
            }
        }
    }

    tracing::info!(downloaded, total_size, "Download complete");

    // Close the file before renaming
    drop(file);

    // Move the downloaded file to the final location
    fs::rename(&temp_path, output_path).context(format!(
        "Failed to rename downloaded file from {:?} to {:?}",
        temp_path, output_path
    ))?;

    Ok(())
}

pub async fn download_file_optional(url: &str, output_path: &Path) -> Result<bool> {
    download_file_optional_with_progress(url, output_path, None).await
}

pub async fn download_file_optional_with_progress(
    url: &str,
    output_path: &Path,
    on_progress: Option<&(dyn Fn(f64) + Send + Sync)>,
) -> Result<bool> {
    tracing::info!("Downloading file from: {}", url);

    if let Some(parent) = output_path.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent)?;
        }
    }

    let response = reqwest::get(url)
        .await
        .context(format!("Failed to download file from {}", url))?;

    if response.status() == reqwest::StatusCode::NOT_FOUND {
        tracing::info!("Optional file not found (404): {}", url);
        return Ok(false);
    }

    if !response.status().is_success() {
        return Err(anyhow::anyhow!(
            "Failed to download file, status: {}",
            response.status()
        ));
    }

    let temp_path = output_path.with_extension("downloading");
    let total_size = response.content_length().unwrap_or(0);
    let mut file = tokio::fs::File::create(&temp_path)
        .await
        .context(format!("Failed to create file at {:?}", temp_path))?;

    let mut stream = response.bytes_stream();
    let mut downloaded: u64 = 0;

    use futures_util::StreamExt;
    while let Some(item) = stream.next().await {
        let chunk = item.context("Error while downloading file")?;
        file.write_all(&chunk).await?;

        downloaded += chunk.len() as u64;
        if total_size > 0 {
            let progress = (downloaded as f64 / total_size as f64) * 100.0;
            if let Some(cb) = &on_progress {
                cb(progress / 100.0);
            }
        }
    }

    file.flush().await?;
    tokio::fs::rename(&temp_path, output_path).await?;
    tracing::info!("\nDownloaded to {:?}", output_path);
    Ok(true)
}

/// Download and initialize the Silero VAD model
pub async fn init_silero_model() -> Result<PathBuf> {
    tracing::info!("Initializing Silero VAD model...");

    // Get models directory
    let models_dir = get_models_dir()?;
    let silero_model_path = models_dir.join(SILERO_MODEL_FILENAME);

    if is_silero_model_valid(&silero_model_path) {
        tracing::info!("Silero VAD model already exists at {:?}", silero_model_path);
        return Ok(silero_model_path);
    }

    tracing::info!("Downloading Silero VAD model from GitHub...");
    download_file(SILERO_VAD_URL, &silero_model_path).await?;

    // Verify the downloaded model
    if !is_silero_model_valid(&silero_model_path) {
        return Err(anyhow::anyhow!(
            "Downloaded Silero model is invalid or corrupted"
        ));
    }

    tracing::info!("Silero VAD model initialized at: {:?}", silero_model_path);
    Ok(silero_model_path)
}

/// Initialize a model, downloading and converting it if necessary
pub async fn init_model(model_name: Option<&str>) -> Result<PathBuf> {
    let model = model_name.unwrap_or(DEFAULT_WHISPER_MODEL);
    tracing::info!("Initializing transcription model: {}", model);

    // Define paths
    let models_dir = get_models_dir()?;
    let model_name_simple = model.split('/').next_back().unwrap_or(model);
    let ct2_model_dir = models_dir.join(format!("{}-ct2", model_name_simple));

    // Check if converted model already exists
    if ct2_model_dir.exists() && is_model_complete(&ct2_model_dir)? {
        tracing::info!("Converted model already exists at {:?}", ct2_model_dir);
        return Ok(ct2_model_dir);
    }

    // Detect system type
    let on_nixos = is_nixos();
    tracing::info!("System detection: Running on NixOS = {}", on_nixos);

    // Try automatic conversion
    tracing::info!("Converting model {} to CTranslate2 format...", model);
    if let Err(e) = convert_model(model, &ct2_model_dir) {
        tracing::info!("Automatic conversion failed: {}", e);

        if on_nixos {
            tracing::info!("\nManual conversion instructions for NixOS:");
            tracing::info!("1. Enter the nix-shell with: nix-shell model-conversion/shell.nix");
            tracing::info!("2. Run the following command:");
        } else {
            tracing::info!("\nManual conversion instructions:");
            tracing::info!(
                "1. Install required packages: pip install -U ctranslate2 huggingface_hub torch transformers"
            );
            tracing::info!("2. Run the following command:");
        }

        tracing::info!(
            "   ct2-transformers-converter --model {} --output_dir {} --copy_files preprocessor_config.json tokenizer.json",
            model,
            ct2_model_dir.display()
        );
        tracing::info!("3. Then run this application again\n");

        return Err(anyhow::anyhow!(
            "Model conversion failed. Please follow the manual instructions."
        ));
    }

    // Verify the converted model
    if !is_model_complete(&ct2_model_dir)? {
        return Err(anyhow::anyhow!("Model conversion failed or is incomplete"));
    }

    tracing::info!("Model initialized at: {:?}", ct2_model_dir);
    Ok(ct2_model_dir)
}

/// Initialize a model of the specified type
pub async fn init_model_by_type(
    model_type: ModelType,
    model_name: Option<&str>,
) -> Result<PathBuf> {
    match model_type {
        ModelType::Whisper => init_model(model_name).await,
        ModelType::Silero => init_silero_model().await,
    }
}

/// Normalize model name based on backend type
///
/// This allows users to specify simple model names (e.g., "base.en", "small")
/// that work across different backends:
/// - CT2: Maps to distil-whisper models for better performance
/// - WhisperCpp: Uses standard OpenAI Whisper models
///
/// # Arguments
/// * `model_name` - User-specified model name
/// * `backend_type` - Which backend is being used
///
/// # Returns
/// Normalized model name appropriate for the backend
fn normalize_model_name(model_name: &str, backend_type: crate::backend::BackendType) -> String {
    // If model already has an organization prefix, use as-is
    if model_name.contains('/') {
        return model_name.to_string();
    }

    match backend_type {
        crate::backend::BackendType::CTranslate2 => {
            // Map simple names intelligently:
            // - Small models (tiny, base): Use standard OpenAI (already fast enough)
            // - Larger models (small+): Use distil-whisper (need speed optimization)
            let ct2_mapping = match model_name {
                "tiny" | "tiny.en" => "openai/whisper-tiny.en",
                "base" | "base.en" => "openai/whisper-base.en",
                "small" | "small.en" => "distil-whisper/distil-small.en",
                "medium" | "medium.en" => "distil-whisper/distil-medium.en",
                "large" | "large-v1" | "large-v2" | "large-v3" => "distil-whisper/distil-large-v3",
                // If it's already a full model name, use as-is
                other => other,
            };

            ct2_mapping.to_string()
        }
        crate::backend::BackendType::WhisperCpp => {
            // WhisperCpp uses standard OpenAI model names as-is
            // Just strip any "distil-" prefix if present
            model_name
                .strip_prefix("distil-")
                .unwrap_or(model_name)
                .to_string()
        }
        crate::backend::BackendType::Parakeet => {
            // Parakeet will use standard model names
            model_name.to_string()
        }
        crate::backend::BackendType::Moonshine => model_name.to_string(),
        crate::backend::BackendType::Nemotron => model_name.to_string(),
    }
}

/// Initialize all required models (Whisper and Silero)
///
/// # Arguments
/// * `model_name` - Name of the transcription model to use
/// * `backend_type` - Which backend to use (determines model format)
/// * `quantization` - Quantization level (for whisper.cpp models)
///
/// # Returns
/// Tuple of (transcription_model_path, silero_model_path)
pub async fn init_all_models(
    model_name: Option<&str>,
    backend_type: crate::backend::BackendType,
    quantization: &QuantizationLevel,
) -> Result<(PathBuf, PathBuf)> {
    ensure_backend_available(backend_type)?;

    // Initialize Silero VAD model
    let silero_model_path = init_silero_model().await?;

    // Normalize model name based on backend
    let original_model = model_name.unwrap_or("base.en");
    let normalized_model = normalize_model_name(original_model, backend_type);

    // Show model mapping if it changed
    if original_model != normalized_model {
        tracing::info!(
            "Model mapping for {} backend: '{}' → '{}'",
            backend_type,
            original_model,
            normalized_model
        );
    }

    // Initialize the transcription model based on backend type
    let transcription_model_path = match backend_type {
        crate::backend::BackendType::CTranslate2 => {
            // CT2: Download and convert model to CT2 format
            init_model(Some(&normalized_model)).await?
        }
        crate::backend::BackendType::WhisperCpp => {
            // WhisperCpp: Get expected GGML model path
            // The factory will handle auto-download if the file doesn't exist
            get_whisper_cpp_model_path(&normalized_model, quantization)?
        }
        crate::backend::BackendType::Parakeet => {
            init_parakeet_model_with_progress(&normalized_model, None).await?
        }
        crate::backend::BackendType::Moonshine => {
            let model_id = moonshine_model_id(&normalized_model);
            init_moonshine_model(&model_id).await?
        }
        crate::backend::BackendType::Nemotron => init_nemotron_model_with_progress(None).await?,
    };

    Ok((transcription_model_path, silero_model_path))
}

/// Resolve a model name to a filesystem path for a given backend.
/// Downloads/converts the model if necessary.
pub async fn resolve_model_path(
    model_name: &str,
    backend_type: crate::backend::BackendType,
    quantization: &QuantizationLevel,
) -> Result<PathBuf> {
    resolve_model_path_with_progress(model_name, backend_type, quantization, None).await
}

/// Resolve a model name to a filesystem path with an optional download progress callback.
pub async fn resolve_model_path_with_progress(
    model_name: &str,
    backend_type: crate::backend::BackendType,
    quantization: &QuantizationLevel,
    on_progress: Option<&(dyn Fn(f64) + Send + Sync)>,
) -> Result<PathBuf> {
    ensure_backend_available(backend_type)?;

    let normalized = normalize_model_name(model_name, backend_type);

    match backend_type {
        crate::backend::BackendType::CTranslate2 => init_model(Some(&normalized)).await,
        crate::backend::BackendType::WhisperCpp => {
            let path = get_whisper_cpp_model_path(&normalized, quantization)?;
            if !path.exists() {
                download_whisper_cpp_model_with_progress(&normalized, quantization, on_progress)
                    .await?;
            }
            Ok(path)
        }
        crate::backend::BackendType::Moonshine => {
            let model_id = moonshine_model_id(&normalized);
            init_moonshine_model_with_progress(&model_id, on_progress).await
        }
        crate::backend::BackendType::Parakeet => {
            init_parakeet_model_with_progress(&normalized, on_progress).await
        }
        crate::backend::BackendType::Nemotron => {
            init_nemotron_model_with_progress(on_progress).await
        }
    }
}

async fn init_moonshine_model(model_id: &str) -> Result<PathBuf> {
    init_moonshine_model_with_progress(model_id, None).await
}

async fn init_moonshine_model_with_progress(
    model_id: &str,
    on_progress: Option<&(dyn Fn(f64) + Send + Sync)>,
) -> Result<PathBuf> {
    let models_dir = get_models_dir()?;
    let model_dir = moonshine_model_dir(&models_dir, model_id);

    if model_dir.exists() && is_moonshine_model_complete(&model_dir)? {
        tracing::info!("Moonshine model already exists at: {:?}", model_dir);
        return Ok(model_dir);
    }

    if !model_dir.exists() {
        fs::create_dir_all(&model_dir)?;
    }

    // ONNX model files from the monorepo: onnx/merged/{model_id}/float/
    for file in MOONSHINE_ONNX_FILES.iter() {
        let output_path = model_dir.join(file);
        if output_path.exists() {
            continue;
        }

        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}/{}/{}/{}/{}",
            MOONSHINE_REPO,
            MOONSHINE_ONNX_DIR,
            MOONSHINE_MERGED_DIR,
            model_id,
            MOONSHINE_MERGED_VARIANT,
            file
        );
        download_file_with_progress(&url, &output_path, on_progress).await?;
    }

    // Config files from standalone repos: UsefulSensors/moonshine-{model_id}
    for file in MOONSHINE_CONFIG_FILES.iter() {
        let output_path = model_dir.join(file);
        if output_path.exists() {
            continue;
        }

        let url = format!(
            "https://huggingface.co/UsefulSensors/moonshine-{}/resolve/main/{}",
            model_id, file
        );
        download_file_with_progress(&url, &output_path, on_progress).await?;
    }

    Ok(model_dir)
}

fn is_parakeet_model_complete(model_dir: &Path) -> Result<bool> {
    for file in PARAKEET_MODEL_FILES.iter() {
        if !model_dir.join(file).exists() {
            return Ok(false);
        }
    }
    Ok(true)
}

fn parakeet_model_dir(models_dir: &Path, model_name: &str) -> PathBuf {
    models_dir.join(parakeet_dir_name(model_name))
}

async fn init_parakeet_model_with_progress(
    model_name: &str,
    on_progress: Option<&(dyn Fn(f64) + Send + Sync)>,
) -> Result<PathBuf> {
    let models_dir = get_models_dir()?;
    let model_dir = parakeet_model_dir(&models_dir, model_name);

    if model_dir.exists() && is_parakeet_model_complete(&model_dir)? {
        tracing::info!("Parakeet model already exists at: {:?}", model_dir);
        return Ok(model_dir);
    }

    if !model_dir.exists() {
        fs::create_dir_all(&model_dir)?;
    }

    let repo = parakeet_hf_repo(model_name);
    for file in PARAKEET_MODEL_FILES.iter() {
        let output_path = model_dir.join(file);
        if output_path.exists() {
            continue;
        }

        let url = format!("https://huggingface.co/{}/resolve/main/{}", repo, file);
        download_file_with_progress(&url, &output_path, on_progress).await?;
    }

    Ok(model_dir)
}

const NEMOTRON_HF_REPO: &str = "onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4";
const NEMOTRON_DIR_NAME: &str = "nemotron-3.5-asr-streaming-0.6b-int4";
const NEMOTRON_MODEL_FILES: [&str; 7] = [
    "encoder.onnx",
    "encoder.onnx.data",
    "decoder.onnx",
    "decoder.onnx.data",
    "joint.onnx",
    "joint.onnx.data",
    "vocab.txt",
];

fn is_nemotron_model_complete(model_dir: &Path) -> Result<bool> {
    for file in NEMOTRON_MODEL_FILES.iter() {
        if !model_dir.join(file).exists() {
            return Ok(false);
        }
    }
    Ok(true)
}

async fn init_nemotron_model_with_progress(
    on_progress: Option<&(dyn Fn(f64) + Send + Sync)>,
) -> Result<PathBuf> {
    let models_dir = get_models_dir()?;
    let model_dir = models_dir.join(NEMOTRON_DIR_NAME);

    if model_dir.exists() && is_nemotron_model_complete(&model_dir)? {
        tracing::info!("Nemotron model already exists at: {:?}", model_dir);
        return Ok(model_dir);
    }

    if !model_dir.exists() {
        fs::create_dir_all(&model_dir)?;
    }

    for file in NEMOTRON_MODEL_FILES.iter() {
        let output_path = model_dir.join(file);
        if output_path.exists() {
            continue;
        }
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            NEMOTRON_HF_REPO, file
        );
        download_file_with_progress(&url, &output_path, on_progress).await?;
    }

    Ok(model_dir)
}

/// Download a whisper.cpp GGML model file
///
/// # Arguments
/// * `model_name` - Base model name (e.g., "base.en", "small", "medium")
/// * `quantization` - Quantization level to determine file variant
///
/// # Returns
/// Path to the downloaded model file
pub async fn download_whisper_cpp_model(
    model_name: &str,
    quantization: &QuantizationLevel,
) -> Result<PathBuf> {
    download_whisper_cpp_model_with_progress(model_name, quantization, None).await
}

pub async fn download_whisper_cpp_model_with_progress(
    model_name: &str,
    quantization: &QuantizationLevel,
    on_progress: Option<&(dyn Fn(f64) + Send + Sync)>,
) -> Result<PathBuf> {
    let models_dir = get_models_dir()?;

    // Determine quantization suffix for filename
    // Available for whisper.cpp: full precision (no suffix), q8_0, q5_1
    let quant_suffix = match quantization {
        QuantizationLevel::High => "", // Full precision (148MB for base.en)
        QuantizationLevel::Medium => "-q8_0", // Q8_0 quantization (82MB for base.en)
        QuantizationLevel::Low => "-q5_1", // Q5_1 quantization (60MB for base.en)
    };

    // Build filename: ggml-{model}{quant}.bin
    let filename = format!("ggml-{}{}.bin", model_name, quant_suffix);
    let output_path = models_dir.join(&filename);

    // Check if already exists
    if output_path.exists() {
        tracing::info!("whisper.cpp model already exists at: {:?}", output_path);
        return Ok(output_path);
    }

    // Download from HuggingFace ggerganov/whisper.cpp repository
    let url = format!(
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{}",
        filename
    );

    tracing::info!("Downloading whisper.cpp model: {}", filename);
    tracing::info!("  From: {}", url);
    tracing::info!("  To: {:?}", output_path);

    download_file_with_progress(&url, &output_path, on_progress).await?;

    tracing::info!("whisper.cpp model downloaded successfully!");
    Ok(output_path)
}

/// Get the expected path for a whisper.cpp model
///
/// # Arguments
/// * `model_name` - Base model name (e.g., "base.en", "small")
/// * `quantization` - Quantization level
///
/// # Returns
/// Expected path to the GGML model file
pub fn get_whisper_cpp_model_path(
    model_name: &str,
    quantization: &QuantizationLevel,
) -> Result<PathBuf> {
    let models_dir = get_models_dir()?;

    // Available for whisper.cpp: full precision (no suffix), q8_0, q5_1
    let quant_suffix = match quantization {
        QuantizationLevel::High => "",        // Full precision
        QuantizationLevel::Medium => "-q8_0", // Q8_0 quantization
        QuantizationLevel::Low => "-q5_1",    // Q5_1 quantization
    };

    let filename = format!("ggml-{}{}.bin", model_name, quant_suffix);
    Ok(models_dir.join(filename))
}
