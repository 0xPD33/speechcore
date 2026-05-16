use speechcore::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let mut engine = SpeechEngine::new(SpeechConfig::default()).await?;
    let session_id = engine.start_session().await?;
    tracing::info!(%session_id, "recording manual session");

    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
    let transcript = engine.stop_and_transcribe().await?;
    println!("{}", transcript.text);

    engine.shutdown().await?;
    Ok(())
}
