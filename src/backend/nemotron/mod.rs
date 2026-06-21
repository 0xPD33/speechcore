pub mod backend;
pub mod decoder;
pub mod lang;
pub mod mel;
pub mod model;

pub use backend::NemotronBackend;

/// Mel left-context frames prepended to every encoder chunk (pre-encode cache).
pub(crate) const PRE_ENCODE_CACHE: usize = 9;
