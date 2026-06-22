# speechcore

[![crates.io](https://img.shields.io/crates/v/speechcore.svg)](https://crates.io/crates/speechcore)
[![docs.rs](https://img.shields.io/docsrs/speechcore)](https://docs.rs/speechcore)
[![CI](https://github.com/0xPD33/speechcore/actions/workflows/ci.yml/badge.svg)](https://github.com/0xPD33/speechcore/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/crates/l/speechcore.svg)](#license)

Reusable Rust speech-to-text runtime components for desktop and service
applications.

`speechcore` contains reusable local STT building blocks: microphone capture,
Silero VAD segmentation, backend selection, model provisioning, transcript
streaming, and runtime state. It deliberately does not contain UI, clipboard
integration, system tray code, global shortcuts, or transcript enhancement
behavior.

## What You Get

- Push-to-talk and real-time transcription orchestration.
- Audio capture via PortAudio.
- Silero VAD for speech segmentation.
- Pluggable transcription backends:
  - CTranslate2
  - Moonshine ONNX
  - NVIDIA Parakeet ONNX
  - NVIDIA Nemotron 3.5 ASR ONNX (streaming: cache-aware FastConformer + RNNT, with low-latency partials)
  - whisper.cpp via `whisper-rs` when explicitly enabled
- Model cache and download helpers.
- Transcript and backend status streams for app-owned UI or service logic.

## Platform Support

Developed and CI-tested on **Linux x86_64** (Ubuntu in CI, NixOS in dev). The
pure-Rust API (default features, no backends) is portable, but each backend pulls
native dependencies that determine real platform support:

| Target | Status |
| --- | --- |
| Linux x86_64 | ✅ Tested (CI + development) |
| macOS | ⚠️ Untested — the ONNX backends are most likely to build; `backend-whisper-cpp` needs MoltenVK for its `vulkan` feature |
| Windows | ⚠️ Untested — expect native-build friction; set `SPEECHCORE_MODEL_DIR` (or `HOME`/`XDG_CACHE_HOME`) as the model cache falls back to `$HOME/.cache` |
| ARM (Apple Silicon / ARM Linux) | ⚠️ Untested |

- `backend-whisper-cpp` builds whisper.cpp with the `vulkan` and `openblas`
  features, so it needs the Vulkan SDK (`glslc`) and OpenBLAS.
- The ONNX backends and Silero VAD use `ort`, which downloads a per-platform ONNX
  Runtime by default; combining them with `backend-ctranslate2` requires a
  **system** ONNX Runtime (see [ONNX Runtime + CTranslate2](#onnx-runtime--ctranslate2-important)).

Contributions to validate or fix other platforms are welcome.

## Installing

Published on [crates.io](https://crates.io/crates/speechcore). The default crate
has no heavy native runtime features enabled — opt into the runtime and the
backend(s) you want:

```toml
speechcore = { version = "0.1", default-features = false, features = ["runtime", "backend-whisper-cpp"] }
```

To track a specific Git revision or tag instead of the crates.io release:

```toml
speechcore = { git = "https://github.com/0xPD33/speechcore", tag = "v0.1.0", default-features = false, features = ["runtime", "backend-whisper-cpp"] }
```

For local development from this workspace:

```toml
speechcore = {
  path = "../speechcore",
  default-features = false,
  features = ["runtime", "backend-parakeet"]
}
```

## Feature Flags

Available features:

| Feature | Enables |
| --- | --- |
| `runtime` | `SpeechEngine`, `RealTimeTranscriber`, audio processing, VAD runtime, model downloads, WAV debug output |
| `all-backends` | CTranslate2, Moonshine, Parakeet, Nemotron, and whisper.cpp backends |
| `audio-capture` | PortAudio microphone capture |
| `vad-silero` | Silero VAD via ONNX Runtime |
| `model-downloads` | model cache and download helpers |
| `wav-output` | WAV debug output via `hound` |
| `backend-ctranslate2` | CTranslate2 backend via `ct2rs` |
| `backend-moonshine` | Moonshine ONNX backend |
| `backend-parakeet` | Parakeet ONNX backend |
| `backend-nemotron` | Nemotron 3.5 ASR streaming ONNX backend |
| `backend-whisper-cpp` | whisper.cpp backend via `whisper-rs` |
| `ort-cuda` | ONNX Runtime CUDA execution provider |
| `ort-tensorrt` | ONNX Runtime TensorRT execution provider |

`backend-whisper-cpp` is intentionally opt-in at the library level. Apps that
also link another `ggml` consumer, such as a llama.cpp enhancement layer, should
make an explicit choice about enabling both native stacks.

## Minimal Manual Session

```rust,no_run
use speechcore::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut engine = SpeechEngine::new(SpeechConfig::default()).await?;
    let _session_id = engine.start_session().await?;
    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
    let transcript = engine.stop_and_transcribe().await?;
    println!("{}", transcript.text);

    engine.shutdown().await?;
    Ok(())
}
```

## Logging

The library uses `tracing` for diagnostics. Applications decide how logs are
collected and displayed:

```rust,no_run
tracing_subscriber::fmt::init();
```

## Native Dependencies

Some features need native libraries:

- `audio-capture`: PortAudio and system audio headers.
- `backend-ctranslate2`: CTranslate2 dependencies required by `ct2rs`.
- ONNX backends and Silero VAD: ONNX Runtime as required by `ort`.
- `backend-whisper-cpp`: whisper.cpp build dependencies and selected
  acceleration features.

On NixOS or other declarative environments, prefer project-local shells or
flakes that pin these dependencies.

### ONNX Runtime + CTranslate2 (important)

`ort` (used by the ONNX backends and Silero VAD) downloads a **static** ONNX
Runtime by default. That static build bundles protobuf, which **collides with
CTranslate2's protobuf at link time** (duplicate `google::protobuf::*` symbols)
whenever both are enabled — and `runtime` enables Silero VAD (ort), so this hits
any `runtime` + `backend-ctranslate2` build. Link a **system** ONNX Runtime so it
is loaded dynamically instead:

```bash
# Install ONNX Runtime (matching ort's expected version, e.g. 1.22.x) to /usr/local, then:
export ORT_STRATEGY=system
export ORT_LIB_LOCATION=/usr/local/lib
export LIBRARY_PATH=/usr/local/lib       # link-time resolution of -lonnxruntime
export LD_LIBRARY_PATH=/usr/local/lib    # run-time loading
# OpenBLAS (whisper.cpp `openblas` feature) needs its headers located:
export BLAS_INCLUDE_DIRS=/usr/include/x86_64-linux-gnu
export OPENBLAS_PATH=/usr
```

The Nix flake sets this up automatically; `.github/workflows/ci.yml` is a
known-good Ubuntu reference for the native dependencies and env vars.

## Model Cache

Model downloads are stored in `~/.cache/speechcore/models` by default, or under
`$XDG_CACHE_HOME/speechcore/models` when `XDG_CACHE_HOME` is set. Applications
can override this with `SPEECHCORE_MODEL_DIR`.

## Repository Checks

Recommended checks before publishing or opening a PR:

```bash
cargo fmt --all -- --check
cargo check
cargo test --no-default-features
cargo check --no-default-features --features runtime
cargo clippy --no-default-features --features runtime --all-targets -- -D warnings
cargo test --no-default-features --features runtime,backend-ctranslate2
cargo check --no-default-features --features runtime,all-backends
cargo check --no-default-features --features runtime,backend-whisper-cpp --examples
cargo doc --no-deps
cargo package
cargo publish --dry-run
```

When validating an uncommitted local extraction, `cargo package --allow-dirty`
and `cargo publish --dry-run --allow-dirty` can preview the package contents. Use
plain `cargo package` and `cargo publish --dry-run` before publishing or tagging
a release.

## License

Licensed under either of:

- Apache License, Version 2.0
- MIT license

at your option.
