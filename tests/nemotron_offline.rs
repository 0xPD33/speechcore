//! Offline transcription smoke test for the Nemotron backend.
//!
//! Requires the cached int4 model + a sample wav (set up by the de-risk step):
//!   ~/.cache/speechcore/models/nemotron-3.5-asr-streaming-0.6b-int4/{*.onnx,vocab.txt,sample1.wav}
//!
//! Ignored by default (needs the ~790MB model). Run explicitly:
//!   cargo test --features "ort-cpu,backend-nemotron,wav-output" \
//!     --test nemotron_offline -- --ignored --nocapture

#![cfg(all(feature = "backend-nemotron", feature = "wav-output"))]

use speechcore::backend::nemotron::NemotronBackend;
use speechcore::config::{CommonTranscriptionOptions, NemotronOptions};
use speechcore::BackendConfig;

fn model_dir() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap();
    std::path::Path::new(&home)
        .join(".cache/speechcore/models/nemotron-3.5-asr-streaming-0.6b-int4")
}

fn read_wav(path: &std::path::Path) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).unwrap();
    match reader.spec().sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i32>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect(),
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
    }
}

#[test]
#[ignore = "requires the cached Nemotron int4 model"]
fn transcribes_english_sample() {
    let dir = model_dir();
    let backend = NemotronBackend::new(&dir, &BackendConfig::default()).unwrap();
    let samples = read_wav(&dir.join("sample1.wav"));

    let text = backend
        .transcribe(
            &samples,
            "en-US",
            &CommonTranscriptionOptions::default(),
            &NemotronOptions::default(),
            16000,
        )
        .unwrap();

    println!("offline transcript: {text}");
    let lower = text.to_lowercase();
    assert!(
        lower.contains("slushy country roads"),
        "unexpected transcript: {text}"
    );
}

#[test]
#[ignore = "requires the cached Nemotron int4 model"]
fn short_slice_quality() {
    // Does the OFFLINE path degrade on short audio (the realtime/VAD case)?
    let dir = model_dir();
    let backend = NemotronBackend::new(&dir, &BackendConfig::default()).unwrap();
    let samples = read_wav(&dir.join("sample1.wav"));
    let opts = NemotronOptions::default();

    for secs in [0.8f32, 1.2, 1.5, 2.0, 3.0, 5.0] {
        let n = ((secs * 16000.0) as usize).min(samples.len());
        let text = backend
            .transcribe(
                &samples[..n],
                "en-US",
                &CommonTranscriptionOptions::default(),
                &opts,
                16000,
            )
            .unwrap();
        println!("[{secs:>3}s] {text:?}");
    }
    // Reference first words at 16k: "Going along slushy country roads and speaking..."
}

#[test]
#[ignore = "requires the cached Nemotron int4 model"]
fn noise_cancelled_silence() {
    // Mimic a noise-cancelled mic: exact-zero silence around / within speech.
    let dir = model_dir();
    let backend = NemotronBackend::new(&dir, &BackendConfig::default()).unwrap();
    let samples = read_wav(&dir.join("sample1.wav"));
    let opts = NemotronOptions::default();
    let speech = &samples[..(3.0 * 16000.0) as usize]; // ~3s of speech

    let pad = vec![0.0f32; 8000]; // 0.5s of EXACT zeros (NC silence)
    let mut padded = pad.clone();
    padded.extend_from_slice(speech);
    padded.extend_from_slice(&pad);

    let bare = backend
        .transcribe(
            speech,
            "en-US",
            &CommonTranscriptionOptions::default(),
            &opts,
            16000,
        )
        .unwrap();
    let with_silence = backend
        .transcribe(
            &padded,
            "en-US",
            &CommonTranscriptionOptions::default(),
            &opts,
            16000,
        )
        .unwrap();

    println!("bare:           {bare:?}");
    println!("zero-padded:    {with_silence:?}");
    // Exact-zero padding shouldn't garble the speech transcription.
    assert!(
        with_silence.to_lowercase().contains("country roads"),
        "exact-zero silence degraded transcription: {with_silence:?}"
    );
}

#[test]
#[ignore = "requires the cached Nemotron int4 model"]
fn streaming_matches_offline() {
    let dir = model_dir();
    let backend = NemotronBackend::new(&dir, &BackendConfig::default()).unwrap();
    let samples = read_wav(&dir.join("sample1.wav"));
    let opts = NemotronOptions::default();

    let offline = backend
        .transcribe(
            &samples,
            "en-US",
            &CommonTranscriptionOptions::default(),
            &opts,
            16000,
        )
        .unwrap();

    // Feed in 0.56s pushes; the final partial should match the offline result.
    backend.stream_reset("en-US", &opts).unwrap();
    let mut last = String::new();
    for chunk in samples.chunks(8960) {
        last = backend.stream_push(chunk).unwrap();
    }
    let streamed = backend.stream_finish().unwrap();

    println!("offline:  {offline}");
    println!("last partial: {last}");
    println!("streamed: {streamed}");
    assert_eq!(streamed, offline, "streaming final != offline");
}
