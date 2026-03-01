use serde::{Deserialize, Serialize};
use crate::data::docx_loader::Document;
use crate::data::tokenizer::Tokenizer;
use burn::data::dataset::Dataset;

/// A single question-answer pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QAPair {
    pub question: String,
    pub answer: String,
    pub source: String,
    /// The month section this event falls under, e.g. "JANUARY 2025".
    /// Empty if no month header was seen before this paragraph.
    #[serde(default)]
    pub month_section: String,
    /// The calendar day (1-31) this event falls on, or 0 if unknown.
    #[serde(default)]
    pub day: u8,
}

const MONTH_NAMES: [&str; 12] = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
];

/// Returns true if a paragraph looks like a standalone month-year header,
/// e.g. "JANUARY 2025", "MARCH 2026".
fn is_month_header(text: &str) -> bool {
    let t = text.trim().to_lowercase();
    MONTH_NAMES.iter().any(|m| {
        t.starts_with(m)
            && (t.len() == m.len()
                || t[m.len()..].trim().chars().all(|c| c.is_ascii_digit() || c.is_whitespace()))
    })
}

/// Generates Q&A pairs from documents by splitting into paragraphs
/// and creating simple factual questions.
///
/// Tracks "MONTH YEAR" headers so every event is tagged with its
/// calendar section (e.g. "JANUARY 2025").
/// Returns true if the text is a standalone day number (1-31),
/// possibly with an event glued onto it, e.g. "17WCED SCHOOLS OPEN".
fn is_day_number(text: &str) -> Option<u8> {
    let t = text.trim();
    // Pure digits
    if let Ok(n) = t.parse::<u8>() {
        if (1..=31).contains(&n) {
            return Some(n);
        }
    }
    // Leading digits like "17WCED SCHOOLS OPEN" or "29START OF TERM 1"
    let digits: String = t.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.len() >= 1 && digits.len() <= 2 {
        if let Ok(n) = digits.parse::<u8>() {
            if (1..=31).contains(&n) {
                return Some(n);
            }
        }
    }
    None
}

/// Day-of-week names that appear as standalone paragraphs in the calendars.
const DAY_NAMES: [&str; 7] = [
    "sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
];

pub fn generate_qa_pairs(documents: &[Document]) -> Vec<QAPair> {
    let mut pairs = Vec::new();

    for doc in documents {
        // Keep ALL non-empty paragraphs so we can detect day numbers.
        let paragraphs: Vec<&str> = doc
            .content
            .split('\n')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        let mut current_month = String::new();
        let mut current_day: u8 = 0;

        for (i, paragraph) in paragraphs.iter().enumerate() {
            // Detect month headers and update tracking
            if is_month_header(paragraph) {
                current_month = paragraph.trim().to_uppercase();
                current_day = 0;
                continue; // header itself is stored but we still make a QA pair below
            }

            // Skip day-of-week headers (SUNDAY, MONDAY, …)
            if DAY_NAMES.contains(&paragraph.to_lowercase().as_str()) {
                continue;
            }

            // Detect day numbers ("5", "17WCED SCHOOLS OPEN", etc.)
            if let Some(d) = is_day_number(paragraph) {
                current_day = d;
                // If this is a bare number (no event text), skip creating a QA pair
                let rest = paragraph.trim_start_matches(|c: char| c.is_ascii_digit());
                if rest.is_empty() {
                    continue;
                }
                // Otherwise fall through — the event text will become the answer
            }

            // Skip very short paragraphs that aren't month headers or days
            if paragraph.len() <= 10 && !is_month_header(paragraph) && is_day_number(paragraph).is_none() {
                continue;
            }

            // Generate a "what" question about each paragraph
            let question = format!(
                "What is described in paragraph {} of {}?",
                i + 1,
                doc.filename
            );
            pairs.push(QAPair {
                question,
                answer: paragraph.to_string(),
                source: doc.filename.clone(),
                month_section: current_month.clone(),
                day: current_day,
            });

            // If paragraph contains a date-like pattern, generate a "when" question
            if contains_date_hint(paragraph) {
                let question = format!(
                    "What events or information are mentioned around paragraph {} in {}?",
                    i + 1,
                    doc.filename
                );
                pairs.push(QAPair {
                    question,
                    answer: paragraph.to_string(),
                    source: doc.filename.clone(),
                    month_section: current_month.clone(),
                    day: current_day,
                });
            }
        }
    }

    pairs
}

fn contains_date_hint(text: &str) -> bool {
    let months = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    ];
    let lower = text.to_lowercase();
    months.iter().any(|m| lower.contains(m))
        || lower.chars().any(|c| c.is_ascii_digit())
}

/// A tokenized Q&A sample ready for training.
#[derive(Debug, Clone)]
pub struct QAItem {
    pub input_ids: Vec<usize>,
    pub target_ids: Vec<usize>,
}

/// Dataset wrapper for Burn.
pub struct QADataset {
    pub items: Vec<QAItem>,
}

impl QADataset {
    /// Loads Q&A pairs from JSON and tokenizes them.
    pub fn from_json(path: &str, tokenizer: &Tokenizer, max_len: usize) -> Self {
        let content = std::fs::read_to_string(path).expect("Failed to read QA JSON");
        let pairs: Vec<QAPair> =
            serde_json::from_str(&content).expect("Failed to parse QA JSON");

        let items = pairs
            .iter()
            .map(|pair| {
                let input_ids = tokenizer.encode(&pair.question, max_len);
                let target_ids = tokenizer.encode(&pair.answer, max_len);
                QAItem {
                    input_ids,
                    target_ids,
                }
            })
            .collect();

        QADataset { items }
    }
}

impl Dataset<QAItem> for QADataset {
    fn get(&self, index: usize) -> Option<QAItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}