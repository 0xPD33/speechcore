use portaudio as pa;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

use crate::transcription_stats::TranscriptionStats;
use parking_lot::Mutex;

/// How long the capture callback may go silent before the device is treated as
/// gone. The callback fires every ~64 ms at a 1024-frame buffer and 16 kHz, so
/// this is roughly fifteen missed callbacks — well clear of normal jitter.
const CALLBACK_STALL_TIMEOUT: Duration = Duration::from_secs(1);

/// Everything needed to reopen the input stream, kept so a device that
/// disappeared can be recovered without rebuilding the whole transcriber.
#[derive(Clone)]
struct StreamParams {
    tx: mpsc::Sender<Vec<f32>>,
    running: Arc<AtomicBool>,
    recording: Arc<AtomicBool>,
    transcription_stats: Arc<Mutex<TranscriptionStats>>,
}

/// Manages audio capture using PortAudio
pub struct AudioCapture {
    pa_stream: Option<pa::Stream<pa::NonBlocking, pa::Input<f32>>>,
    pa: Option<pa::PortAudio>,
    input_settings: Option<pa::InputStreamSettings<f32>>,
    samples_sent: Arc<AtomicUsize>,
    buffer_size: usize,
    stream_params: Option<StreamParams>,
    /// Last observed callback count and when it was observed.
    callback_seen: Option<(usize, Instant)>,
    /// Rate-limits recovery so a permanently absent device does not spin.
    last_recovery_attempt: Option<Instant>,
    /// Part of the input device name to record from; the default device when unset.
    input_device: Option<String>,
    /// Whether the caller wants the mic open. Stall recovery only reopens
    /// the device while this is set, so a stopped mic stays closed.
    wants_stream: bool,
}

impl AudioCapture {
    /// Creates a new AudioCapture instance
    pub fn new() -> Self {
        Self::with_buffer_size(crate::config::AudioProcessorConfig::default().buffer_size)
    }

    pub fn with_buffer_size(buffer_size: usize) -> Self {
        Self {
            pa_stream: None,
            pa: None,
            input_settings: None,
            samples_sent: Arc::new(AtomicUsize::new(0)),
            buffer_size,
            stream_params: None,
            callback_seen: None,
            last_recovery_attempt: None,
            wants_stream: false,
            input_device: None,
        }
    }

    /// Records from the first input device whose name contains `name`,
    /// ignoring case. `None` keeps the system default.
    pub fn with_input_device(mut self, name: Option<String>) -> Self {
        self.input_device = name.filter(|name| !name.trim().is_empty());
        self
    }

    /// Initializes PortAudio settings without starting the stream
    fn initialize_audio(&mut self) -> Result<(), anyhow::Error> {
        if self.pa.is_some() {
            return Ok(()); // Already initialized
        }

        tracing::debug!("Initializing PortAudio");
        let pa = pa::PortAudio::new()
            .map_err(|e| anyhow::anyhow!("Failed to initialize PortAudio: {}", e))?;

        let input_params = match &self.input_device {
            Some(name) => named_input_params(&pa, name)?,
            None => pa.default_input_stream_params::<f32>(1).map_err(|e| {
                anyhow::anyhow!("Failed to get default input stream parameters: {}", e)
            })?,
        };
        tracing::debug!("Input device resolved");

        let frames_per_buffer = u32::try_from(self.buffer_size)
            .map_err(|_| anyhow::anyhow!("Audio buffer size too large: {}", self.buffer_size))?;

        let input_settings = pa::InputStreamSettings::new(
            input_params,
            crate::config::SAMPLE_RATE as f64,
            frames_per_buffer,
        );

        self.pa = Some(pa);
        self.input_settings = Some(input_settings);
        Ok(())
    }

    /// Stores what the stream needs. The device opens on `start_recording`,
    /// so the mic stays closed until a recording starts.
    ///
    /// # Arguments
    /// * `tx` - Channel sender for audio samples
    /// * `running` - Atomic flag indicating whether the app is running
    /// * `recording` - Atomic flag indicating whether recording is active
    ///
    /// # Returns
    /// Result indicating success or error
    pub fn start(
        &mut self,
        tx: mpsc::Sender<Vec<f32>>,
        running: Arc<AtomicBool>,
        recording: Arc<AtomicBool>,
        transcription_stats: Arc<Mutex<TranscriptionStats>>,
    ) -> Result<(), anyhow::Error> {
        self.stream_params = Some(StreamParams {
            tx,
            running,
            recording,
            transcription_stats,
        });
        Ok(())
    }

    /// Opens the input stream from the parameters captured by `start`.
    fn open_stream(&mut self) -> Result<(), anyhow::Error> {
        let StreamParams {
            tx,
            running,
            recording,
            transcription_stats,
        } = self
            .stream_params
            .clone()
            .ok_or_else(|| anyhow::anyhow!("Audio capture has not been started"))?;

        self.initialize_audio()?;

        let pa = self
            .pa
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("PortAudio not initialized"))?;
        let input_settings = *self
            .input_settings
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Audio input settings not initialized"))?;

        // Clone the recording Arc before moving it into the closure
        let recording_for_callback = recording.clone();
        let stats_for_callback = transcription_stats.clone();
        let samples_sent_for_callback = self.samples_sent.clone();

        let callback = move |pa::InputStreamCallbackArgs { buffer, .. }| {
            // Only send samples when recording is active
            if recording_for_callback.load(Ordering::Relaxed) {
                let samples = buffer.to_vec();
                match tx.try_send(samples) {
                    Ok(_) => {
                        // Increment counter after successful send
                        samples_sent_for_callback.fetch_add(1, Ordering::Release);
                    }
                    Err(mpsc::error::TrySendError::Full(_)) => {
                        if let Some(mut stats) = stats_for_callback.try_lock() {
                            let total = stats.record_audio_drop(1);
                            tracing::warn!(
                                "Audio channel full, dropped samples (total: {})",
                                total
                            );
                        }
                    }
                    Err(mpsc::error::TrySendError::Closed(_)) => {
                        tracing::warn!("Failed to send samples: channel closed");
                        if let Some(mut stats) = stats_for_callback.try_lock() {
                            let total = stats.record_audio_drop(1);
                            tracing::warn!("Audio channel drop recorded (total: {})", total);
                        }
                    }
                }
            }

            // Check if we should continue based on running flag
            if running.load(Ordering::Relaxed) {
                pa::Continue
            } else {
                pa::Complete
            }
        };

        let mut stream = pa
            .open_non_blocking_stream(input_settings, callback)
            .map_err(|e| anyhow::anyhow!("Failed to open stream: {}", e))?;

        // Only start the stream if recording is active
        if recording.load(Ordering::Relaxed) {
            stream
                .start()
                .map_err(|e| anyhow::anyhow!("Failed to start stream: {}", e))?;
        }

        self.pa_stream = Some(stream);
        Ok(())
    }

    /// Starts the PortAudio stream when recording begins.
    ///
    /// The device is always rebuilt first rather than reusing the stream from
    /// the previous session. If the device went away while we were idle — audio
    /// server restart, mic unplugged, suspend — `Pa_StartStream` on the stale
    /// stream logs an ALSA failure and then blocks forever, wedging every task
    /// that later wants the capture lock. Nothing exposes whether a stream has
    /// gone stale, so the only safe move is not to keep one. Reopening costs
    /// ~20 ms, and it means a changed default input device is picked up
    /// without restarting the application.
    pub fn start_recording(&mut self) -> Result<(), anyhow::Error> {
        self.wants_stream = true;
        self.reopen()?;
        self.try_start_stream()?;
        self.callback_seen = None;
        Ok(())
    }

    fn try_start_stream(&mut self) -> Result<(), anyhow::Error> {
        let stream = self
            .pa_stream
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("No audio input stream"))?;

        if !stream.is_active().unwrap_or(false) {
            stream
                .start()
                .map_err(|e| anyhow::anyhow!("Failed to start recording: {}", e))?;
        }
        Ok(())
    }

    /// Rebuilds PortAudio and the input stream from scratch.
    ///
    /// PortAudio snapshots the device list at Pa_Initialize, so a device that
    /// appeared, vanished or changed since then only becomes visible after a
    /// full terminate/initialize cycle. Reopening just the stream is not
    /// enough — dropping the `PortAudio` handle is what terminates it.
    fn reopen(&mut self) -> Result<(), anyhow::Error> {
        self.close_stream();
        tracing::debug!("Reopening audio input stream");
        self.open_stream()
    }

    /// Rebuilds the device if the capture callback has stopped firing.
    ///
    /// Returns whether a recovery was performed. A frozen callback counter is
    /// the reliable signal: the callback runs on every buffer regardless of
    /// signal level, so silence still advances it and only a dead device stops
    /// it. Call this periodically while recording.
    pub fn recover_if_stalled(&mut self) -> Result<bool, anyhow::Error> {
        if !self.wants_stream {
            return Ok(false);
        }

        let now = Instant::now();

        // Don't hammer a device that is simply not coming back.
        if let Some(attempted) = self.last_recovery_attempt {
            if now.duration_since(attempted) < CALLBACK_STALL_TIMEOUT {
                return Ok(false);
            }
        }

        let sent = self.samples_sent.load(Ordering::Acquire);
        let (stalled, seen) = callback_stalled(self.callback_seen, sent, now);
        self.callback_seen = seen;

        let stream_dead = !matches!(&self.pa_stream, Some(s) if s.is_active().unwrap_or(false));
        if !stalled && !stream_dead {
            return Ok(false);
        }

        tracing::warn!(
            "Audio input stopped delivering samples (stalled: {}, stream inactive: {}); reopening device",
            stalled,
            stream_dead
        );
        self.last_recovery_attempt = Some(now);
        self.callback_seen = None;
        self.reopen()?;
        self.try_start_stream()?;
        tracing::info!("Audio input recovered");
        Ok(true)
    }

    /// Closes the device when recording ends, so the mic is released.
    pub fn stop_recording(&mut self) -> Result<(), anyhow::Error> {
        self.wants_stream = false;
        self.close_stream();
        Ok(())
    }

    /// Gets the count of audio samples sent through the channel
    pub fn get_samples_sent_count(&self) -> Arc<AtomicUsize> {
        self.samples_sent.clone()
    }

    /// Temporarily pauses audio capture without closing the stream
    /// This allows for resuming the stream later
    ///
    /// # Returns
    /// Result indicating success or error
    pub fn pause(&mut self) -> Result<(), anyhow::Error> {
        self.stop_recording()
    }

    /// Resumes a previously paused audio capture stream
    ///
    /// # Returns
    /// Result indicating success or error
    pub fn resume(&mut self) -> Result<(), anyhow::Error> {
        self.start_recording()
    }

    /// Completely stops and cleans up the audio capture
    /// This closes the stream and releases resources
    pub fn stop(&mut self) {
        self.wants_stream = false;
        self.close_stream();
        // Dropped last: it holds the sample channel open.
        self.stream_params = None;
    }

    /// Tears down the stream and the PortAudio instance, keeping the
    /// parameters needed to build them again.
    ///
    /// Always Pa_AbortStream: Pa_StopStream waits for the stream to drain and
    /// blocks forever when the device is already gone.
    fn close_stream(&mut self) {
        if let Some(stream) = &mut self.pa_stream {
            if let Err(e) = stream.abort() {
                tracing::warn!("Failed to stop stream: {}", e);
            }
            if let Err(e) = stream.close() {
                tracing::warn!("Failed to close stream: {}", e);
            }
            tracing::debug!("Audio stream closed");
        }
        // Order matters: the stream must drop before the PortAudio handle that
        // owns it, and dropping that handle is what calls Pa_Terminate.
        self.pa_stream = None;
        self.pa = None;
        self.input_settings = None;
        self.callback_seen = None;
    }
}

/// Stream parameters for the first input device whose name contains `name`.
fn named_input_params(
    pa: &pa::PortAudio,
    name: &str,
) -> Result<pa::StreamParameters<f32>, anyhow::Error> {
    let wanted = name.to_lowercase();
    let mut available = Vec::new();
    for device in pa
        .devices()
        .map_err(|e| anyhow::anyhow!("Failed to list audio devices: {}", e))?
    {
        let Ok((index, info)) = device else {
            continue;
        };
        if info.max_input_channels < 1 {
            continue;
        }
        if info.name.to_lowercase().contains(&wanted) {
            tracing::info!("Using input device: {}", info.name);
            return Ok(pa::StreamParameters::new(
                index,
                1,
                true,
                info.default_low_input_latency,
            ));
        }
        available.push(info.name.to_string());
    }
    Err(anyhow::anyhow!(
        "No input device matches \"{}\". Available: {}",
        name,
        available.join(", ")
    ))
}

/// Pure half of stall detection, split out so it is testable without a device.
/// Returns whether the callback has been frozen past the timeout, along with
/// the observation to carry into the next check.
fn callback_stalled(
    seen: Option<(usize, Instant)>,
    sent: usize,
    now: Instant,
) -> (bool, Option<(usize, Instant)>) {
    match seen {
        // Same count as last time: keep the original timestamp so the elapsed
        // window accumulates instead of resetting on every check.
        Some((last_sent, since)) if last_sent == sent => (
            now.duration_since(since) >= CALLBACK_STALL_TIMEOUT,
            Some((last_sent, since)),
        ),
        _ => (false, Some((sent, now))),
    }
}

impl Default for AudioCapture {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for AudioCapture {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_check_only_arms_the_window() {
        let now = Instant::now();
        let (stalled, seen) = callback_stalled(None, 7, now);
        assert!(!stalled);
        assert_eq!(seen, Some((7, now)));
    }

    #[test]
    fn advancing_callbacks_never_stall() {
        let start = Instant::now();
        let (_, seen) = callback_stalled(None, 7, start);
        let later = start + CALLBACK_STALL_TIMEOUT * 5;
        let (stalled, seen) = callback_stalled(seen, 8, later);
        assert!(!stalled, "a device still delivering samples is healthy");
        assert_eq!(seen, Some((8, later)));
    }

    #[test]
    fn frozen_callbacks_stall_only_after_the_timeout() {
        let start = Instant::now();
        let (_, seen) = callback_stalled(None, 7, start);

        let (stalled, seen) = callback_stalled(seen, 7, start + CALLBACK_STALL_TIMEOUT / 2);
        assert!(!stalled, "brief jitter is not a dead device");

        // The window must accumulate rather than reset on each check.
        let (stalled, _) = callback_stalled(seen, 7, start + CALLBACK_STALL_TIMEOUT);
        assert!(stalled);
    }
}
