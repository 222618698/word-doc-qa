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
}

/// Generates Q&A pairs from documents by splitting into paragraphs
/// and creating simple factual questions.
pub fn generate_qa_pairs(documents: &[Document]) -> Vec<QAPair> {
    let mut pairs = Vec::new();

    for doc in documents {
        let paragraphs: Vec<&str> = doc
            .content
            .split('\n')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty() && s.len() > 10)
            .collect();

        for (i, paragraph) in paragraphs.iter().enumerate() {
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