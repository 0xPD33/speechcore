# Repository Guidelines

## Project Scope
- `speechcore` is a reusable Rust speech-to-text runtime crate.
- Keep reusable STT behavior here: audio capture, VAD, backend selection, model provisioning, transcription orchestration, transcript/status streams, feedback events, and stats.
- Keep app-specific behavior out of this crate: UI, overlays, clipboard/paste, global shortcuts, system tray, app IPC, and transcript enhancement/Magic Mode.

## Project Structure
- `src/lib.rs` defines the public API and feature-gated re-exports.
- `src/config.rs` contains reusable speech runtime configuration.
- `src/backend/` contains backend traits, factory logic, and backend implementations.
- `src/audio_capture.rs`, `src/audio_processor.rs`, `src/silero_audio_processor.rs`, and `src/transcription_processor.rs` implement the runtime pipeline.
- `src/engine.rs` exposes the higher-level manual-session API.
- `examples/` contains small app-facing usage examples.

## Build And Test Commands
- `cargo fmt --all -- --check`
- `cargo check --no-default-features`
- `cargo test --no-default-features`
- `cargo clippy --no-default-features --all-targets -- -D warnings`
- `cargo check --no-default-features --features runtime`
- `cargo clippy --no-default-features --features runtime --all-targets -- -D warnings`
- `cargo check`
- `cargo test`
- `cargo clippy --all-targets -- -D warnings`
- `cargo check --examples`
- `cargo check --no-default-features --features backend-whisper-cpp`

## Feature Policy
- Keep dependencies optional unless required by the minimal public API.
- `runtime` should enable the complete local STT runtime without forcing every backend.
- Backend features should map one-to-one to backend implementations.
- Apps that need a small dependency graph should be able to use `default-features = false`.

## Public API Guidelines
- Prefer stable, app-facing types over exposing implementation details.
- Add new public exports intentionally in `src/lib.rs` and document the expected use case.
- Preserve feature gates on public exports so minimal builds stay lightweight.
- Avoid Sonori-specific names, defaults, cache paths, or UI assumptions.

## Model And Cache Behavior
- STT model downloads use `~/.cache/speechcore/models` by default.
- Respect `XDG_CACHE_HOME` and `SPEECHCORE_MODEL_DIR`.
- Do not write app-specific files or configuration from this crate.

## Testing Guidelines
- Add unit tests near the module being changed.
- For shared behavior used by apps, prefer narrow tests that validate config conversion, backend selection, chunking, or model-path resolution.
- When touching feature-gated code, run the smallest relevant feature profile and the default profile.
