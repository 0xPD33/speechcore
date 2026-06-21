//! Language-ID prompt mapping for Nemotron 3.5 ASR.
//!
//! The encoder takes a `lang_id` (one-hot over 128 prompt slots). The ordering
//! lives only in the `.nemo` `model_config.yaml` `prompt_dictionary`; the ONNX
//! export ships no map, so it is reproduced here. `auto` (101) enables the
//! model's built-in language detection (emits a language tag in the output).

/// Resolve a locale string (e.g. "en-US", "de", "auto") to its `lang_id`.
/// Falls back to en-US (0) for unknown inputs. Case/format tolerant.
pub fn lang_id(locale: &str) -> i64 {
    let key = locale.trim();
    LANG_TABLE
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case(key))
        .map(|(_, id)| *id)
        .unwrap_or(0)
}

/// Locales offered to the UI, in display order (one canonical entry each).
pub const SUPPORTED_LOCALES: &[&str] = &[
    "auto", "en-US", "en-GB", "es-ES", "es-US", "fr-FR", "fr-CA", "de-DE", "it-IT", "pt-BR",
    "pt-PT", "nl-NL", "ru-RU", "uk-UA", "pl-PL", "cs-CZ", "sk-SK", "hr-HR", "bg-BG", "sl-SI",
    "ro-RO", "el-GR", "hu-HU", "sv-SE", "da-DK", "fi-FI", "nb-NO", "nn-NO", "lt-LT", "lv-LV",
    "et-EE", "tr-TR", "ar-AR", "he-IL", "hi-IN", "ja-JP", "ko-KR", "zh-CN", "th-TH", "vi-VN",
    "mt-MT",
];

/// Full `prompt_dictionary` from the `.nemo` config (locale -> one-hot slot).
const LANG_TABLE: &[(&str, i64)] = &[
    ("en-US", 0),
    ("en", 0),
    ("en-GB", 1),
    ("es-ES", 2),
    ("es-US", 3),
    ("es", 3),
    ("zh-CN", 4),
    ("zh", 4),
    ("zh-TW", 5),
    ("hi-IN", 6),
    ("hi", 6),
    ("ar-AR", 7),
    ("ar", 7),
    ("fr-FR", 8),
    ("fr", 8),
    ("de-DE", 9),
    ("de", 9),
    ("ja-JP", 10),
    ("ja", 10),
    ("ru-RU", 11),
    ("ru", 11),
    ("pt-BR", 12),
    ("pt-PT", 13),
    ("pt", 13),
    ("ko-KR", 14),
    ("ko", 14),
    ("it-IT", 15),
    ("it", 15),
    ("nl-NL", 16),
    ("nl", 16),
    ("pl-PL", 17),
    ("pl", 17),
    ("tr-TR", 18),
    ("tr", 18),
    ("uk-UA", 19),
    ("uk", 19),
    ("ro-RO", 20),
    ("ro", 20),
    ("el-GR", 21),
    ("el", 21),
    ("cs-CZ", 22),
    ("cs", 22),
    ("hu-HU", 23),
    ("hu", 23),
    ("sv-SE", 24),
    ("sv", 24),
    ("da-DK", 25),
    ("da", 25),
    ("fi-FI", 26),
    ("fi", 26),
    ("no-NO", 27),
    ("no", 27),
    ("sk-SK", 28),
    ("sk", 28),
    ("hr-HR", 29),
    ("hr", 29),
    ("bg-BG", 30),
    ("bg", 30),
    ("lt-LT", 31),
    ("lt", 31),
    ("th-TH", 32),
    ("vi-VN", 33),
    ("et-EE", 60),
    ("et", 60),
    ("lv-LV", 61),
    ("lv", 61),
    ("sl-SI", 62),
    ("sl", 62),
    ("he-IL", 64),
    ("he", 64),
    ("fr-CA", 100),
    ("auto", 101),
    ("mt-MT", 102),
    ("nb-NO", 103),
    ("nb", 103),
    ("nn-NO", 104),
    ("nn", 104),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_locales_resolve() {
        assert_eq!(lang_id("en-US"), 0);
        assert_eq!(lang_id("EN"), 0); // case-insensitive alias
        assert_eq!(lang_id("de-DE"), 9);
        assert_eq!(lang_id("auto"), 101);
        assert_eq!(lang_id("nb-NO"), 103);
    }

    #[test]
    fn unknown_falls_back_to_english() {
        assert_eq!(lang_id("xx-XX"), 0);
    }
}
