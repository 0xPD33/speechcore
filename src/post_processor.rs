//! Text post-processing for transcription cleanup
//!
//! Handles removal of artifacts like leading/trailing dashes and
//! normalization of whitespace in transcription output.

use crate::config::PostProcessConfig;

/// Post-process transcription text to clean up artifacts
///
/// Applies configured cleaning rules to remove dashes and normalize whitespace
/// based on the provided configuration.
pub fn post_process_text(text: String, config: &PostProcessConfig) -> String {
    if !config.enabled {
        return text;
    }

    let mut processed = text;

    if config.remove_leading_dashes {
        processed = remove_leading_dashes(processed);
    }

    if config.remove_trailing_dashes {
        processed = remove_trailing_dashes(processed);
    }

    if config.remove_fillers {
        processed = remove_fillers(processed);
    }

    if config.collapse_repeated_words {
        processed = collapse_repeated_words(processed);
    }

    // Whitespace last of the removals: dropping words leaves double spaces
    // behind, and this is what tidies them up.
    if config.normalize_whitespace {
        processed = normalize_whitespace(processed);
    }

    // These two read sentence boundaries, so they run once spacing is settled.
    if config.capitalize_sentences {
        processed = capitalize_sentences(processed);
    }

    if config.ensure_terminal_punctuation {
        processed = ensure_terminal_punctuation(processed);
    }

    processed
}

/// Filler words dropped when `remove_fillers` is set.
///
/// Deliberately short. "ah", "oh", "so", "like" and "you know" all carry
/// meaning often enough that removing them would silently change what the
/// speaker said, which is worse than leaving a filler in.
const FILLERS: &[&str] = &["um", "uh", "uhm", "erm", "hm", "hmm", "mhm"];

/// The comparable core of a token: lowercased, stripped of surrounding
/// punctuation. "Um," and "um" both reduce to "um".
fn word_core(token: &str) -> String {
    token
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_lowercase()
}

/// Remove standalone filler words
///
/// Only whole tokens are considered, so "hmm" goes but "hmmm" and words that
/// merely contain a filler are left alone.
fn remove_fillers(text: String) -> String {
    let kept: Vec<&str> = text
        .split_whitespace()
        .filter(|token| {
            let core = word_core(token);
            !FILLERS.contains(&core.as_str())
        })
        .collect();

    kept.join(" ")
}

/// Collapse a word immediately repeated by a stutter or a chunk boundary
///
/// Only collapses when the first occurrence carries no trailing punctuation,
/// so "No, no" and "Done. Done." survive — there the repetition is intended.
fn collapse_repeated_words(text: String) -> String {
    let mut kept: Vec<&str> = Vec::new();

    for token in text.split_whitespace() {
        let is_repeat = kept.last().is_some_and(|previous: &&str| {
            let clean_break = previous.chars().last().is_some_and(|c| c.is_alphanumeric());
            clean_break && word_core(previous) == word_core(token) && !word_core(token).is_empty()
        });

        if !is_repeat {
            kept.push(token);
        }
    }

    kept.join(" ")
}

/// Capitalize the first letter of the text and of each following sentence
fn capitalize_sentences(text: String) -> String {
    let mut result = String::with_capacity(text.len());
    let mut start_of_sentence = true;

    for c in text.chars() {
        if start_of_sentence && c.is_alphabetic() {
            result.extend(c.to_uppercase());
            start_of_sentence = false;
        } else {
            result.push(c);
            if matches!(c, '.' | '!' | '?') {
                start_of_sentence = true;
            }
        }
    }

    result
}

/// Append a full stop when the text ends on a word rather than punctuation
fn ensure_terminal_punctuation(text: String) -> String {
    let trimmed = text.trim_end();
    match trimmed.chars().last() {
        Some(c) if c.is_alphanumeric() => format!("{trimmed}."),
        _ => text,
    }
}

/// Remove leading dashes and following whitespace
///
/// Removes patterns like "- text" at the start of a string
fn remove_leading_dashes(text: String) -> String {
    let trimmed = text.trim_start();
    if trimmed.starts_with('-') {
        trimmed.trim_start_matches('-').trim_start().to_string()
    } else {
        text
    }
}

/// Remove trailing dashes and preceding whitespace
///
/// Removes patterns like "text -" at the end of a string
fn remove_trailing_dashes(text: String) -> String {
    let trimmed = text.trim_end();
    if trimmed.ends_with('-') {
        trimmed.trim_end_matches('-').trim_end().to_string()
    } else {
        text
    }
}

/// Normalize whitespace in text
///
/// - Collapses multiple consecutive spaces (2+) into single spaces
/// - Removes leading and trailing whitespace
/// - Converts newlines and tabs to spaces
/// - Preserves single spaces and natural word boundaries from AI model
fn normalize_whitespace(text: String) -> String {
    let mut result = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            // Preserve single spaces between words
            ' ' if !result.is_empty()
                && !result.ends_with(' ')
                && !chars.peek().is_none_or(|&next| next.is_whitespace()) =>
            {
                // This is a single space between non-whitespace, keep it
                result.push(' ');
            }
            // Collapse multiple consecutive whitespace characters
            c if c.is_whitespace() => {
                // Skip all consecutive whitespace characters
                while chars.peek().is_some_and(|&next| next.is_whitespace()) {
                    chars.next();
                }
                // Add single space if not at beginning or end
                if !result.is_empty()
                    && !matches!(chars.peek(), None | Some(' ' | '\t' | '\n' | '\r'))
                {
                    result.push(' ');
                }
            }
            _ => {
                result.push(c);
            }
        }
    }

    // Clean up any trailing space that might have been added
    result.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> PostProcessConfig {
        PostProcessConfig::default()
    }

    #[test]
    fn disabled_config_is_a_passthrough() {
        let config = PostProcessConfig {
            enabled: false,
            ..cfg()
        };
        let input = "  um   so    I  said  ".to_string();
        assert_eq!(post_process_text(input.clone(), &config), input);
    }

    #[test]
    fn fillers_are_dropped_with_their_punctuation() {
        assert_eq!(
            remove_fillers("Um, I think uh we should go".to_string()),
            "I think we should go"
        );
    }

    #[test]
    fn fillers_only_match_whole_words() {
        // "humming" contains "hm"; "Umbrella" starts with "um".
        assert_eq!(
            remove_fillers("Umbrella humming hmmm".to_string()),
            "Umbrella humming hmmm"
        );
    }

    #[test]
    fn repeated_words_collapse_across_a_stutter() {
        assert_eq!(
            collapse_repeated_words("I I went to the the shop".to_string()),
            "I went to the shop"
        );
    }

    #[test]
    fn repetition_across_punctuation_is_intentional() {
        // The speaker meant both of these; only stutters should collapse.
        assert_eq!(
            collapse_repeated_words("No, no it is done. Done.".to_string()),
            "No, no it is done. Done."
        );
    }

    #[test]
    fn sentences_are_capitalized_after_terminators() {
        assert_eq!(
            capitalize_sentences("hello there. how are you? good!".to_string()),
            "Hello there. How are you? Good!"
        );
    }

    #[test]
    fn terminal_punctuation_is_added_only_when_missing() {
        assert_eq!(
            ensure_terminal_punctuation("no full stop".to_string()),
            "no full stop."
        );
        assert_eq!(
            ensure_terminal_punctuation("already there!".to_string()),
            "already there!"
        );
    }

    #[test]
    fn removals_do_not_leave_double_spaces() {
        let config = PostProcessConfig {
            collapse_repeated_words: true,
            ..cfg()
        };
        assert_eq!(
            post_process_text("- I um went went to the shop -".to_string(), &config),
            "I went to the shop"
        );
    }

    #[test]
    fn defaults_leave_ordinary_dictation_alone() {
        // The risky transforms are off by default, so a well-formed
        // transcript from a backend that already punctuates is untouched.
        let input = "I had had enough. That that is fine!".to_string();
        assert_eq!(post_process_text(input.clone(), &cfg()), input);
    }

    #[test]
    fn all_fillers_leaves_empty_text() {
        assert_eq!(post_process_text("Um, uh. Hmm".to_string(), &cfg()), "");
    }
}
