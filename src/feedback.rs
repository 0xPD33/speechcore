#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FeedbackEvent {
    RecordStart,
    RecordStop,
    SessionStart,
    SessionComplete,
    SessionCancel,
}

pub trait FeedbackSink: Send + Sync {
    fn play(&self, event: FeedbackEvent);
}
