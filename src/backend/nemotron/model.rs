//! Session loading + the cache-aware streaming encoder for Nemotron 3.5 ASR.
//!
//! The encoder is strictly chunked: each call consumes 9 carried + 56 new mel
//! frames (`audio_signal[1,65,128]`) and emits 7 output frames, threading five
//! cache tensors forward. `EncoderCache` holds that thread so the same routine
//! serves both the offline whole-segment path and incremental streaming.

use crate::backend::onnx_utils::{load_session, OnnxSessionOptions};
use crate::backend::traits::TranscriptionError;
use anyhow::Result;
use ndarray::{s, Array1, Array2, Array3, ArrayD, IxDyn};
use ort::session::Session;
use ort::value::Tensor;
use parking_lot::Mutex;
use std::path::Path;

use super::mel::{MelSpectrogram, MEL_BINS};

use super::PRE_ENCODE_CACHE;
const CHUNK_NEW: usize = 56;
const CHUNK_IN: usize = PRE_ENCODE_CACHE + CHUNK_NEW; // 65
const ENC_LAYERS: usize = 24;
const D_MODEL: usize = 1024;
const LAST_CHANNEL_CACHE: usize = 56;
const LAST_TIME_CACHE: usize = 8;

pub struct NemotronModel {
    pub encoder: Mutex<Session>,
    pub decoder: Mutex<Session>,
    pub joiner: Mutex<Session>,
    pub mel: MelSpectrogram,
    pub vocab: Vec<String>,
}

impl NemotronModel {
    pub fn load(model_dir: impl AsRef<Path>, options: &OnnxSessionOptions) -> Result<Self> {
        let dir = model_dir.as_ref();
        tracing::info!("Loading Nemotron sessions from: {}", dir.display());
        let encoder = load_session(dir.join("encoder.onnx"), options)?;
        let decoder = load_session(dir.join("decoder.onnx"), options)?;
        let joiner = load_session(dir.join("joint.onnx"), options)?;
        let vocab = load_vocab(&dir.join("vocab.txt"))?;
        Ok(Self {
            encoder: Mutex::new(encoder),
            decoder: Mutex::new(decoder),
            joiner: Mutex::new(joiner),
            mel: MelSpectrogram::new(),
            vocab,
        })
    }

    pub fn validate_model_dir(model_dir: impl AsRef<Path>) -> Result<()> {
        let dir = model_dir.as_ref();
        for f in ["encoder.onnx", "decoder.onnx", "joint.onnx", "vocab.txt"] {
            if !dir.join(f).exists() {
                return Err(anyhow::anyhow!("Missing Nemotron model file: {f}"));
            }
        }
        Ok(())
    }

    /// Run the encoder over a full mel segment with a fresh cache (offline path).
    /// `mel` is `[time, 128]`; returns encoder frames as row-major `[T_enc][1024]`.
    pub fn encode_full(
        &self,
        mel: &Array2<f32>,
        lang_id: i64,
    ) -> Result<Vec<Vec<f32>>, TranscriptionError> {
        let mut cache = EncoderCache::new();
        // extended = [9 zero frames] ++ mel, so chunk c = extended[c*56 .. c*56+65]
        let total = mel.shape()[0];
        let mut ext = Array2::<f32>::zeros((PRE_ENCODE_CACHE + total, MEL_BINS));
        ext.slice_mut(s![PRE_ENCODE_CACHE.., ..]).assign(mel);

        let num_chunks = total.div_ceil(CHUNK_NEW);
        let mut frames = Vec::new();
        for c in 0..num_chunks {
            let start = c * CHUNK_NEW;
            let new_here = CHUNK_NEW.min(total - start);
            self.encode_chunk(&ext, start, new_here, lang_id, &mut cache, &mut frames)?;
        }
        Ok(frames)
    }

    /// Run a single encoder chunk. `ext` holds `[9 left-context + new]` mel rows;
    /// the chunk reads `ext[start .. start+65]`, with `new_here` valid new frames.
    pub fn encode_chunk(
        &self,
        ext: &Array2<f32>,
        start: usize,
        new_here: usize,
        lang_id: i64,
        cache: &mut EncoderCache,
        out_frames: &mut Vec<Vec<f32>>,
    ) -> Result<(), TranscriptionError> {
        let valid_in = PRE_ENCODE_CACHE + new_here;
        let mut chunk = Array3::<f32>::zeros((1, CHUNK_IN, MEL_BINS));
        for r in 0..valid_in {
            let src = start + r;
            if src < ext.shape()[0] {
                chunk
                    .slice_mut(s![0, r, ..])
                    .assign(&ext.slice(s![src, ..]));
            }
        }

        let length = Array1::<i64>::from(vec![valid_in as i64]);
        let lang = Array1::<i64>::from(vec![lang_id]);

        let mut session = self.encoder.lock();
        let outputs = session
            .run(ort::inputs! {
                "audio_signal" => tensor(chunk)?,
                "length" => tensor(length)?,
                "cache_last_channel" => tensor(cache.last_channel.clone())?,
                "cache_last_time" => tensor(cache.last_time.clone())?,
                "cache_last_channel_len" => tensor(cache.last_channel_len.clone())?,
                "lang_id" => tensor(lang)?,
            })
            .map_err(|e| TranscriptionError::InferenceError(format!("encoder run: {e}")))?;

        let out = extract_f32(&outputs, "outputs")?; // [1, 7, 1024]
        let enc_len = outputs
            .get("encoded_lengths")
            .ok_or_else(|| TranscriptionError::InferenceError("missing encoded_lengths".into()))?
            .try_extract_array::<i64>()
            .map_err(|e| TranscriptionError::InferenceError(format!("enc_len: {e}")))?;
        let valid_out = enc_len.iter().next().copied().unwrap_or(0) as usize;

        for t in 0..valid_out.min(out.shape()[1]) {
            out_frames.push(out.slice(s![0, t, ..]).to_vec());
        }

        cache.last_channel = extract_f32(&outputs, "cache_last_channel_next")?;
        cache.last_time = extract_f32(&outputs, "cache_last_time_next")?;
        cache.last_channel_len = extract_f32_i64(&outputs, "cache_last_channel_len_next")?;
        Ok(())
    }
}

/// The five-tensor encoder cache threaded across chunks.
pub struct EncoderCache {
    last_channel: ArrayD<f32>,
    last_time: ArrayD<f32>,
    last_channel_len: Array1<i64>,
}

impl EncoderCache {
    pub fn new() -> Self {
        Self {
            last_channel: ArrayD::zeros(IxDyn(&[1, ENC_LAYERS, LAST_CHANNEL_CACHE, D_MODEL])),
            last_time: ArrayD::zeros(IxDyn(&[1, ENC_LAYERS, D_MODEL, LAST_TIME_CACHE])),
            last_channel_len: Array1::zeros(1),
        }
    }
}

impl Default for EncoderCache {
    fn default() -> Self {
        Self::new()
    }
}

fn tensor<
    T: ort::tensor::PrimitiveTensorElementType + Clone + std::fmt::Debug + 'static,
    D: ndarray::Dimension + 'static,
>(
    a: ndarray::Array<T, D>,
) -> Result<Tensor<T>, TranscriptionError> {
    Tensor::from_array(a).map_err(|e| TranscriptionError::InferenceError(format!("tensor: {e}")))
}

fn extract_f32(
    outputs: &ort::session::SessionOutputs,
    name: &str,
) -> Result<ArrayD<f32>, TranscriptionError> {
    Ok(outputs
        .get(name)
        .ok_or_else(|| TranscriptionError::InferenceError(format!("missing {name}")))?
        .try_extract_array::<f32>()
        .map_err(|e| TranscriptionError::InferenceError(format!("{name} extract: {e}")))?
        .to_owned())
}

fn extract_f32_i64(
    outputs: &ort::session::SessionOutputs,
    name: &str,
) -> Result<Array1<i64>, TranscriptionError> {
    outputs
        .get(name)
        .ok_or_else(|| TranscriptionError::InferenceError(format!("missing {name}")))?
        .try_extract_array::<i64>()
        .map_err(|e| TranscriptionError::InferenceError(format!("{name} extract: {e}")))?
        .to_owned()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|e| TranscriptionError::InferenceError(format!("{name} dim: {e}")))
}

fn load_vocab(path: &Path) -> Result<Vec<String>> {
    Ok(std::fs::read_to_string(path)?
        .lines()
        .map(|l| l.to_string())
        .collect())
}
