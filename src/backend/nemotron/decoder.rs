//! RNNT greedy decoding: LSTM prediction network + joint network.
//!
//! State (`h`, `c`, last prediction output) is exposed so it can persist across
//! streaming chunks; the offline path just primes it once per utterance.

use crate::backend::traits::TranscriptionError;
use ndarray::{Array2, Array3, ArrayD, IxDyn};
use ort::session::Session;
use ort::value::Tensor;
use parking_lot::Mutex;

const D_MODEL: usize = 1024;
const DEC_LAYERS: usize = 2;
const DEC_HIDDEN: usize = 640;
const VOCAB: usize = 13088;
pub const BLANK: i64 = 13087;
const MAX_SYMBOLS: usize = 10;

/// Prediction-network recurrent state for one utterance.
pub struct RnntState {
    h: ArrayD<f32>,       // [2, 1, 640]
    c: ArrayD<f32>,       // [2, 1, 640]
    dec_out: ArrayD<f32>, // [1, 640, 1]
}

impl RnntState {
    /// Initialize and prime the prediction net with a blank token.
    pub fn primed(decoder: &Mutex<Session>) -> Result<Self, TranscriptionError> {
        let h = ArrayD::<f32>::zeros(IxDyn(&[DEC_LAYERS, 1, DEC_HIDDEN]));
        let c = ArrayD::<f32>::zeros(IxDyn(&[DEC_LAYERS, 1, DEC_HIDDEN]));
        let (dec_out, h, c) = run_decoder(decoder, BLANK, &h, &c)?;
        Ok(Self { h, c, dec_out })
    }
}

/// Greedy-decode a batch of encoder frames, appending emitted token ids to
/// `tokens` and advancing `state`. `enc_frames` is row-major `[T, 1024]`.
pub fn decode_frames(
    decoder: &Mutex<Session>,
    joiner: &Mutex<Session>,
    state: &mut RnntState,
    enc_frames: &[Vec<f32>],
    tokens: &mut Vec<i64>,
) -> Result<(), TranscriptionError> {
    for frame in enc_frames {
        let mut emitted = 0;
        loop {
            let logits = run_joiner(joiner, frame, &state.dec_out)?;
            let k = argmax(&logits) as i64;
            if k == BLANK || emitted >= MAX_SYMBOLS {
                break;
            }
            tokens.push(k);
            emitted += 1;
            let (dec_out, h, c) = run_decoder(decoder, k, &state.h, &state.c)?;
            state.dec_out = dec_out;
            state.h = h;
            state.c = c;
        }
    }
    Ok(())
}

// (decoder_output, h_out, c_out)
#[allow(clippy::type_complexity)]
fn run_decoder(
    decoder: &Mutex<Session>,
    token: i64,
    h: &ArrayD<f32>,
    c: &ArrayD<f32>,
) -> Result<(ArrayD<f32>, ArrayD<f32>, ArrayD<f32>), TranscriptionError> {
    let targets = Array2::<i64>::from_shape_vec((1, 1), vec![token])
        .map_err(|e| TranscriptionError::InferenceError(format!("targets shape: {e}")))?;
    let mut session = decoder.lock();
    let outputs = session
        .run(ort::inputs! {
            "targets" => tensor(targets)?,
            "h_in" => tensor(h.clone())?,
            "c_in" => tensor(c.clone())?,
        })
        .map_err(|e| TranscriptionError::InferenceError(format!("decoder run: {e}")))?;

    let dec_out = extract(&outputs, "decoder_output")?;
    let h_out = extract(&outputs, "h_out")?;
    let c_out = extract(&outputs, "c_out")?;
    Ok((dec_out, h_out, c_out))
}

fn run_joiner(
    joiner: &Mutex<Session>,
    enc_frame: &[f32],     // [1024]
    dec_out: &ArrayD<f32>, // [1, 640, 1]
) -> Result<Vec<f32>, TranscriptionError> {
    let enc = Array3::<f32>::from_shape_vec((1, 1, D_MODEL), enc_frame.to_vec())
        .map_err(|e| TranscriptionError::InferenceError(format!("enc frame shape: {e}")))?;
    // decoder_output [1,640,1] -> joint wants [1, target_len, 640] = [1,1,640]
    let dec_vec: Vec<f32> = dec_out.iter().copied().collect();
    let dec = Array3::<f32>::from_shape_vec((1, 1, DEC_HIDDEN), dec_vec)
        .map_err(|e| TranscriptionError::InferenceError(format!("dec out shape: {e}")))?;

    let mut session = joiner.lock();
    let outputs = session
        .run(ort::inputs! {
            "encoder_output" => tensor(enc)?,
            "decoder_output" => tensor(dec)?,
        })
        .map_err(|e| TranscriptionError::InferenceError(format!("joiner run: {e}")))?;

    let logits = outputs
        .get("joint_output")
        .ok_or_else(|| TranscriptionError::InferenceError("missing joint_output".into()))?
        .try_extract_array::<f32>()
        .map_err(|e| TranscriptionError::InferenceError(format!("joint extract: {e}")))?;
    Ok(logits.iter().copied().take(VOCAB).collect())
}

fn tensor<
    T: ort::tensor::PrimitiveTensorElementType + Clone + std::fmt::Debug + 'static,
    D: ndarray::Dimension + 'static,
>(
    a: ndarray::Array<T, D>,
) -> Result<Tensor<T>, TranscriptionError> {
    Tensor::from_array(a).map_err(|e| TranscriptionError::InferenceError(format!("tensor: {e}")))
}

fn extract(
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

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}
