use parking_lot::Mutex;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::transcription_stats::TranscriptionStats;

const STATS_INTERVAL_SECS: u64 = 10;

/// Handles reporting of transcription statistics
pub struct StatsReporter {
    transcription_stats: Arc<Mutex<TranscriptionStats>>,
    running: Arc<AtomicBool>,
    log_stats_enabled: bool,
}

impl StatsReporter {
    /// Creates a new StatsReporter
    pub fn new(
        transcription_stats: Arc<Mutex<TranscriptionStats>>,
        running: Arc<AtomicBool>,
        log_stats_enabled: bool,
    ) -> Self {
        Self {
            transcription_stats,
            running,
            log_stats_enabled,
        }
    }

    /// Start periodic reporting with specified interval
    pub fn start_periodic_reporting(&self) {
        // Exit early if stats logging is not enabled
        if !self.log_stats_enabled {
            tracing::info!("Stats reporting disabled - no statistics will be logged");
            return;
        }

        tracing::info!(
            "Stats reporting enabled - will report every {} seconds",
            STATS_INTERVAL_SECS
        );

        let transcription_stats = self.transcription_stats.clone();
        let running = self.running.clone();

        // Create stats file
        tracing::info!("Stats logging enabled - will write to console and transcription_stats.log");

        // Create or truncate the stats file
        if let Err(e) = File::create("transcription_stats.log") {
            tracing::warn!("Failed to create stats file: {}", e);
        }

        // Spawn an async task to periodically report transcription statistics
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(STATS_INTERVAL_SECS));
            while running.load(Ordering::Relaxed) {
                interval.tick().await;
                if let Some(stats) = transcription_stats.try_lock() {
                    if stats.segments_processed > 0 {
                        let stats_report = stats.report();
                        tracing::info!("\n--- Periodic Transcription Statistics ---");
                        tracing::info!("{}", stats_report);
                        tracing::info!("------------------------------------------\n");
                        let timestamp =
                            chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
                        let file_content = format!("\n--- {} ---\n{}\n", timestamp, stats_report);
                        match std::fs::OpenOptions::new()
                            .append(true)
                            .create(true)
                            .open("transcription_stats.log")
                        {
                            Ok(mut file) => {
                                if let Err(e) = writeln!(file, "{}", file_content) {
                                    tracing::warn!("Failed to write to stats file: {}", e);
                                }
                            }
                            Err(e) => tracing::warn!("Failed to open stats file: {}", e),
                        }
                    }
                }
            }
            tracing::info!("Stats reporting stopped");
        });
    }

    /// Print current statistics on demand
    pub fn print_stats(&self) {
        // Exit early if stats logging is not enabled
        if !self.log_stats_enabled {
            tracing::info!("Stats reporting disabled - no statistics will be logged on demand");
            return;
        }

        if let Some(stats) = self.transcription_stats.try_lock() {
            if stats.segments_processed > 0 {
                let stats_report = stats.report();

                // Log to console
                tracing::info!("\n--- Current Transcription Statistics ---");
                tracing::info!("{}", stats_report);
                tracing::info!("-----------------------------------------\n");

                // Write to file
                let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
                let file_content = format!(
                    "\n--- {} (On-Demand Report) ---\n{}\n",
                    timestamp, stats_report
                );

                match OpenOptions::new()
                    .append(true)
                    .create(true)
                    .open("transcription_stats.log")
                {
                    Ok(mut file) => {
                        if let Err(e) = writeln!(file, "{}", file_content) {
                            tracing::warn!("Failed to write to stats file: {}", e);
                        }
                    }
                    Err(e) => tracing::warn!("Failed to open stats file: {}", e),
                }
            } else {
                tracing::info!("No transcription statistics available yet.");
            }
        } else {
            tracing::info!("Could not access transcription statistics (locked).");
        }
    }
}
