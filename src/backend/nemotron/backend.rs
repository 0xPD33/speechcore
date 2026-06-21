//! Nemotron 3.5 ASR backend: offline transcription + incremental streaming.

use crate::backend::traits::TranscriptionError;
use crate::backend::{BackendCapabilities, BackendConfig};
use crate::config::{CommonTranscriptionOptions, NemotronOptions};
use parking_lot::Mutex;
use std::path::Path;

use super::decoder::{decode_frames, RnntState};
use super::lang;
use super::model::{EncoderCache, NemotronModel};

const CHUNK_NEW: usize = 56; // mel frames per streaming step

pub struct NemotronBackend {
    model: NemotronModel,
    config: BackendConfig,
    /// Per-utterance streaming state (None until a stream is started).
    stream: Mutex<Option<StreamSession>>,
}

struct StreamSession {
    lang_id: i64,
    cache: EncoderCache,
    rnnt: RnntState,
    samples: Vec<f32>,
    chunks_done: usize,
    tokens: Vec<i64>,
}

impl NemotronBackend {
    pub fn new(
        model_path: impl AsRef<Path>,
        backend_config: &BackendConfig,
    ) -> Result<Self, TranscriptionError> {
        let options = crate::backend::onnx_utils::OnnxSessionOptions {
            intra_threads: backend_config.threads.max(1),
            inter_threads: 1,
            execution_provider: if backend_config.gpu_enabled {
                crate::backend::onnx_utils::ExecutionProviderPreference::PreferGpu
            } else {
                crate::backend::onnx_utils::ExecutionProviderPreference::CpuOnly
            },
        };
        let model = NemotronModel::load(model_path, &options)
            .map_err(|e| TranscriptionError::ModelNotAvailable(e.to_string()))?;
        Ok(Self {
            model,
            config: backend_config.clone(),
            stream: Mutex::new(None),
        })
    }

    pub fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            name: "nemotron-3.5-asr",
            max_audio_duration: None,
            supported_languages: Some(
                lang::SUPPORTED_LOCALES
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
            ),
            supports_streaming: true,
            gpu_available: self.config.gpu_enabled,
        }
    }

    /// Offline: transcribe a whole audio segment in one shot.
    pub fn transcribe(
        &self,
        samples: &[f32],
        language: &str,
        _common: &CommonTranscriptionOptions,
        options: &NemotronOptions,
        _sample_rate: usize,
    ) -> Result<String, TranscriptionError> {
        let lang_id = self.resolve_lang(language, options);
        let mel = self.model.mel.compute(samples);
        if mel.shape()[0] == 0 {
            return Ok(String::new());
        }
        let frames = self.model.encode_full(&mel, lang_id)?;
        let mut state = RnntState::primed(&self.model.decoder)?;
        let mut tokens = Vec::new();
        decode_frames(
            &self.model.decoder,
            &self.model.joiner,
            &mut state,
            &frames,
            &mut tokens,
        )?;
        Ok(detokenize(&tokens, &self.model.vocab))
    }

    /// Start a fresh streaming utterance.
    pub fn stream_reset(
        &self,
        language: &str,
        options: &NemotronOptions,
    ) -> Result<(), TranscriptionError> {
        let lang_id = self.resolve_lang(language, options);
        let rnnt = RnntState::primed(&self.model.decoder)?;
        *self.stream.lock() = Some(StreamSession {
            lang_id,
            cache: EncoderCache::new(),
            rnnt,
            samples: Vec::new(),
            chunks_done: 0,
            tokens: Vec::new(),
        });
        Ok(())
    }

    /// Feed more audio; returns the cumulative partial transcript so far.
    /// Only full 56-frame chunks are processed; a trailing partial waits for
    /// more audio or `stream_finish`.
    pub fn stream_push(&self, new_samples: &[f32]) -> Result<String, TranscriptionError> {
        let mut guard = self.stream.lock();
        let s = guard
            .as_mut()
            .ok_or_else(|| TranscriptionError::InferenceError("stream not started".into()))?;
        s.samples.extend_from_slice(new_samples);
        self.advance(s, false)
    }

    /// Finish the utterance, flushing the trailing partial chunk; returns the
    /// final transcript and clears streaming state.
    pub fn stream_finish(&self) -> Result<String, TranscriptionError> {
        let mut guard = self.stream.lock();
        let text = match guard.as_mut() {
            Some(s) => self.advance(s, true)?,
            None => String::new(),
        };
        *guard = None;
        Ok(text)
    }

    /// Process newly-available mel chunks against the persistent cache.
    ///
    // ponytail: recomputes mel over the whole utterance buffer each call (O(n)
    // per push, O(n^2) per utterance). Fine for short dictation segments; switch
    // to incremental mel framing if long-form streaming throughput matters.
    fn advance(&self, s: &mut StreamSession, flush: bool) -> Result<String, TranscriptionError> {
        let mel = self.model.mel.compute(&s.samples);
        let total = mel.shape()[0];
        if total == 0 {
            return Ok(detokenize(&s.tokens, &self.model.vocab));
        }

        // extended = [9 zero frames] ++ mel, so chunk c = ext[c*56 .. c*56+65]
        let pre = super::PRE_ENCODE_CACHE;
        let mut ext = ndarray::Array2::<f32>::zeros((pre + total, super::mel::MEL_BINS));
        ext.slice_mut(ndarray::s![pre.., ..]).assign(&mel);

        let full_chunks = total / CHUNK_NEW;
        let target = if flush {
            total.div_ceil(CHUNK_NEW)
        } else {
            full_chunks
        };

        let mut new_frames = Vec::new();
        while s.chunks_done < target {
            let start = s.chunks_done * CHUNK_NEW;
            let new_here = CHUNK_NEW.min(total - start);
            self.model.encode_chunk(
                &ext,
                start,
                new_here,
                s.lang_id,
                &mut s.cache,
                &mut new_frames,
            )?;
            s.chunks_done += 1;
        }
        decode_frames(
            &self.model.decoder,
            &self.model.joiner,
            &mut s.rnnt,
            &new_frames,
            &mut s.tokens,
        )?;
        Ok(detokenize(&s.tokens, &self.model.vocab))
    }

    fn resolve_lang(&self, language: &str, options: &NemotronOptions) -> i64 {
        // Explicit per-backend locale wins; else the app-level language string.
        let locale = if !options.language.trim().is_empty() {
            options.language.as_str()
        } else {
            language
        };
        lang::lang_id(locale)
    }
}

/// SentencePiece detokenization: vocab line-index ids, `▁` -> space, skip `<…>`.
fn detokenize(tokens: &[i64], vocab: &[String]) -> String {
    let mut s = String::new();
    for &t in tokens {
        let Some(tok) = vocab.get(t as usize) else {
            continue;
        };
        if tok.starts_with('<') && tok.ends_with('>') {
            continue; // <unk>, <blank>, <xx-XX> language tags
        }
        s.push_str(tok);
    }
    s.replace('\u{2581}', " ").trim().to_string()
}
