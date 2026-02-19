/// Utility functions for the project.

/// Truncates a string to a maximum number of characters.
pub fn truncate(s: &str, max_chars: usize) -> &str {
    match s.char_indices().nth(max_chars) {
        Some((idx, _)) => &s[..idx],
        None => s,
    }
}

/// Prints a separator line.
pub fn separator() {
    println!("{}", "=".repeat(60));
}