//! Log-mel features for Nemotron 3.5 ASR (NeMo-faithful).
//!
//! Differs from the Parakeet featurizer: center reflect-pad, a 400-sample Hann
//! window centered inside the 512-point FFT, an additive log guard, and
//! crucially **no per-feature normalization** (`normalize: NA`). Output layout
//! is `[time, n_mels]` (mel-last), matching the encoder's `audio_signal` input.

use ndarray::Array2;
use rustfft::{num_complex::Complex, FftPlanner};
use std::sync::Arc;

const SAMPLE_RATE: f32 = 16000.0;
const FFT: usize = 512;
const WIN: usize = 400;
const HOP: usize = 160;
const N_MELS: usize = 128;
const PREEMPH: f32 = 0.97;
// Featurizer params per the model's audio_processor_config.json. The additive
// log guard (1e-10) and dither (1e-5) are a matched pair that governs how
// silent/quiet frames are represented — important for exact-zero input (e.g.
// noise-cancelled mics) which would otherwise be pinned to a too-loud floor.
const LOG_GUARD: f32 = 1e-10;
const DITHER: f32 = 1e-5;

pub const MEL_BINS: usize = N_MELS;
pub const HOP_SIZE: usize = HOP;

pub struct MelSpectrogram {
    fft: Arc<dyn rustfft::Fft<f32>>,
    window: Vec<f32>,          // centered Hann in 512
    filterbank: Vec<Vec<f32>>, // [n_mels][fft/2+1]
}

impl Default for MelSpectrogram {
    fn default() -> Self {
        Self::new()
    }
}

impl MelSpectrogram {
    pub fn new() -> Self {
        let mut planner = FftPlanner::<f32>::new();
        Self {
            fft: planner.plan_fft_forward(FFT),
            window: centered_hann(WIN, FFT),
            filterbank: build_filterbank(N_MELS, FFT, SAMPLE_RATE),
        }
    }

    /// Compute log-mel features for `samples`, returning `[time, n_mels]`.
    pub fn compute(&self, samples: &[f32]) -> Array2<f32> {
        let dithered = dither(samples);
        let emphasized = preemphasis(&dithered);
        let padded = reflect_pad(&emphasized, FFT / 2); // center=True
        let bins = FFT / 2 + 1;

        let num_frames = if padded.len() >= FFT {
            (padded.len() - FFT) / HOP + 1
        } else {
            0
        };

        let mut out = Array2::<f32>::zeros((num_frames, N_MELS));
        let mut buf = vec![Complex::new(0.0f32, 0.0); FFT];

        for t in 0..num_frames {
            let s = t * HOP;
            for (i, b) in buf.iter_mut().enumerate() {
                *b = Complex::new(padded[s + i] * self.window[i], 0.0);
            }
            self.fft.process(&mut buf);
            for (m, filter) in self.filterbank.iter().enumerate() {
                let acc: f32 = filter
                    .iter()
                    .zip(buf[..bins].iter())
                    .map(|(w, c)| w * c.norm_sqr())
                    .sum();
                out[[t, m]] = (acc + LOG_GUARD).ln();
            }
        }
        out
    }
}

/// Add low-level dither, matching the model's featurizer (`dither: 1e-5`). NeMo
/// trains on audio with a natural noise floor; dither restores that floor for
/// exact-zero input so silent frames aren't degenerate.
// ponytail: deterministic xorshift uniform noise — dither only needs to break up
// exact zeros, not be cryptographic or precisely Gaussian. Swap for randn if it
// ever measurably matters.
fn dither(samples: &[f32]) -> Vec<f32> {
    let mut s: u32 = 0x9E37_79B9;
    samples
        .iter()
        .map(|&x| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let u = (s as f32 / u32::MAX as f32) * 2.0 - 1.0; // [-1, 1]
            x + DITHER * u
        })
        .collect()
}

fn preemphasis(samples: &[f32]) -> Vec<f32> {
    let mut e = Vec::with_capacity(samples.len());
    if let Some(&first) = samples.first() {
        e.push(first);
        for i in 1..samples.len() {
            e.push(samples[i] - PREEMPH * samples[i - 1]);
        }
    }
    e
}

fn reflect_pad(x: &[f32], p: usize) -> Vec<f32> {
    let n = x.len();
    if n <= p {
        // Degenerate (very short) input: just zero-pad, avoids OOB reflect.
        let mut out = vec![0.0; p];
        out.extend_from_slice(x);
        out.extend(std::iter::repeat(0.0).take(p));
        return out;
    }
    let mut out = Vec::with_capacity(n + 2 * p);
    for k in 0..p {
        out.push(x[p - k]); // x[p], x[p-1], ..., x[1]
    }
    out.extend_from_slice(x);
    for k in 0..p {
        out.push(x[n - 2 - k]); // x[n-2], x[n-3], ...
    }
    out
}

fn centered_hann(win: usize, fft: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; fft];
    let off = (fft - win) / 2;
    for n in 0..win {
        w[off + n] = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / win as f32).cos());
    }
    w
}

fn hz_to_mel(hz: f32) -> f32 {
    if hz < 1000.0 {
        hz / 200.0 * 3.0
    } else {
        15.0 + (hz / 1000.0).ln() * (27.0 / (6400.0f32 / 1000.0).ln())
    }
}

fn mel_to_hz(mel: f32) -> f32 {
    if mel < 15.0 {
        mel * 200.0 / 3.0
    } else {
        1000.0 * ((mel - 15.0) * (6400.0f32 / 1000.0).ln() / 27.0).exp()
    }
}

fn build_filterbank(n_mels: usize, fft: usize, sr: f32) -> Vec<Vec<f32>> {
    let bins = fft / 2 + 1;
    let (mmin, mmax) = (hz_to_mel(0.0), hz_to_mel(sr / 2.0));
    let mel_pts: Vec<f32> = (0..n_mels + 2)
        .map(|i| mmin + (mmax - mmin) * i as f32 / (n_mels + 1) as f32)
        .collect();
    let bin_pts: Vec<f32> = mel_pts
        .iter()
        .map(|&m| mel_to_hz(m) * fft as f32 / sr)
        .collect();

    let mut fb = vec![vec![0.0f32; bins]; n_mels];
    for m in 0..n_mels {
        let (l, ce, r) = (bin_pts[m], bin_pts[m + 1], bin_pts[m + 2]);
        for (k, w) in fb[m].iter_mut().enumerate() {
            let kf = k as f32;
            if kf >= l && kf <= ce && ce > l {
                *w = (kf - l) / (ce - l);
            } else if kf > ce && kf <= r && r > ce {
                *w = (r - kf) / (r - ce);
            }
        }
        // Slaney normalization (unit area per filter).
        let enorm = 2.0 / (mel_to_hz(mel_pts[m + 2]) - mel_to_hz(mel_pts[m]));
        for w in fb[m].iter_mut() {
            *w *= enorm;
        }
    }
    fb
}
