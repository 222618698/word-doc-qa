use burn::prelude::*;
use burn::backend::NdArray;
use burn::record::{DefaultFileRecorder, FullPrecisionSettings};
use std::collections::{HashMap, HashSet};

use crate::data::dataset::QAPair;
use crate::data::tokenizer::Tokenizer;
use crate::model::transformer::{TransformerModel, TransformerModelConfig};
use crate::training::config::TrainingConfig;

type Backend = NdArray;

const MONTHS: [&str; 12] = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
];

/// Public entry point: load model + tokenizer, answer the question.
///
/// Strategy:
///   1. Expand abbreviations in the question (e.g. "HDC" → "Higher Degrees Committee").
///   2. Detect counting questions ("how many times …") and return a count.
///   3. If the question targets a specific month (+ optional year), use
///      section-aware retrieval.
///   4. Otherwise, fall back to keyword + transformer-embedding re-ranking.
pub fn answer_question(question: &str) -> String {
    // Load QA pairs (knowledge base)
    let qa_json = std::fs::read_to_string("data/processed/qa_pairs.json")
        .expect("Failed to read qa_pairs.json — run 'generate' first");
    let qa_pairs: Vec<QAPair> =
        serde_json::from_str(&qa_json).expect("Failed to parse qa_pairs.json");

    if qa_pairs.is_empty() {
        return "No knowledge base found. Run 'generate' first.".to_string();
    }

    // ── Step 0: expand abbreviations ───────────────────────────────────
    let abbrev_map = build_abbreviation_map(&qa_pairs);
    let expanded_question = expand_abbreviations(question, &abbrev_map);

    // ── Step 1: detect counting questions ──────────────────────────────
    if is_counting_question(&expanded_question) {
        if let Some(answer) = handle_counting_question(&expanded_question, &qa_pairs) {
            return answer;
        }
    }

    // ── Step 2: try section-aware month retrieval ──────────────────────
    if let Some(answer) = try_month_section_lookup(&expanded_question, &qa_pairs) {
        return answer;
    }

    // ── Step 3: fallback keyword + embedding re-ranking ────────────────
    keyword_and_embedding_search(&expanded_question, &qa_pairs)
}

// ────────────────────────────────────────────────────────────────────────
// Abbreviation expansion
// ────────────────────────────────────────────────────────────────────────

/// Build a map of uppercase abbreviations → full names from the QA data.
/// e.g. "HDC" → "Higher Degrees Committee", "SRC" → "Student Representative Council"
fn build_abbreviation_map(qa_pairs: &[QAPair]) -> HashMap<String, String> {
    let mut names: HashSet<String> = HashSet::new();
    for pair in qa_pairs {
        let ans = pair.answer.trim();
        // Only consider answers that look like committee/event names (>= 2 words)
        let word_count = ans.split_whitespace().count();
        if word_count >= 2 && ans.len() > 5 {
            names.insert(ans.to_string());
        }
    }

    let mut map: HashMap<String, String> = HashMap::new();
    for name in &names {
        // Build abbreviation from capital first letters of each word
        let abbrev: String = name
            .split_whitespace()
            .filter_map(|w| {
                let first = w.chars().next()?;
                if first.is_uppercase() {
                    Some(first)
                } else {
                    None
                }
            })
            .collect();

        // Only register abbreviations that are 2+ chars and all uppercase
        if abbrev.len() >= 2 {
            // If multiple names produce the same abbreviation, keep the shorter one
            let entry = map.entry(abbrev).or_insert_with(|| name.clone());
            if name.len() < entry.len() {
                *entry = name.clone();
            }
        }
    }

    map
}

/// Replace abbreviations in the question with their full names.
fn expand_abbreviations(question: &str, map: &HashMap<String, String>) -> String {
    let mut result = question.to_string();
    // Sort abbreviations longest-first to avoid partial replacement
    let mut abbrevs: Vec<(&String, &String)> = map.iter().collect();
    abbrevs.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

    for (abbrev, full_name) in &abbrevs {
        // Match the abbreviation as a whole word (case-insensitive)
        let upper = abbrev.to_uppercase();
        let words: Vec<&str> = result.split_whitespace().collect();
        let new_words: Vec<String> = words
            .iter()
            .map(|w| {
                let clean: String = w.chars().filter(|c| c.is_alphanumeric()).collect();
                if clean.to_uppercase() == upper {
                    full_name.to_string()
                } else {
                    w.to_string()
                }
            })
            .collect();
        result = new_words.join(" ");
    }

    result
}

// ────────────────────────────────────────────────────────────────────────
// Counting questions
// ────────────────────────────────────────────────────────────────────────

/// Detect if a question is asking about frequency/count.
fn is_counting_question(question: &str) -> bool {
    let q = question.to_lowercase();
    q.contains("how many") || q.contains("how often") || q.contains("count")
}

/// Handle a counting question by finding matching events and counting
/// unique occurrences, optionally filtered by year.
fn handle_counting_question(question: &str, qa_pairs: &[QAPair]) -> Option<String> {
    let q_lower = question.to_lowercase();
    let keywords = extract_keywords(question);

    if keywords.is_empty() {
        return None;
    }

    // Extract a year filter if present
    let target_year: Option<String> = q_lower
        .split(|c: char| !c.is_ascii_digit())
        .find(|w| w.len() == 4)
        .map(|y| y.to_string());

    // Build the "subject" phrase from non-noise keywords
    let noise: HashSet<&str> = [
        "meetings", "meeting", "times", "time", "hold", "held", "their",
        "many", "often", "count", "2024", "2025", "2026", "did",
        "meet", "meets", "met", "does", "happen", "happens",
    ].iter().cloned().collect();
    let subject_keywords: Vec<String> = keywords
        .iter()
        .filter(|k| !noise.contains(k.to_lowercase().as_str()))
        .cloned()
        .collect();

    if subject_keywords.is_empty() {
        return None;
    }

    // Try phrase matching first: join subject keywords and look for
    // answers that contain ALL of them (case-insensitive)
    let mut matches: Vec<&QAPair> = Vec::new();
    for pair in qa_pairs {
        let a_lower = pair.answer.to_lowercase();
        let all_match = subject_keywords.iter().all(|kw| a_lower.contains(kw.as_str()));
        if !all_match {
            continue;
        }

        // Apply year filter via month_section
        if let Some(ref year) = target_year {
            if !pair.month_section.contains(year.as_str()) {
                continue;
            }
        }

        matches.push(pair);
    }

    if matches.is_empty() {
        return None;
    }

    // Count distinct months where the event appears
    let mut months_seen: HashSet<String> = HashSet::new();
    for pair in &matches {
        if !pair.month_section.is_empty() {
            months_seen.insert(pair.month_section.clone());
        }
    }

    // Build readable subject label
    let subject_label = subject_keywords.join(" ");
    let year_label = target_year.as_deref().unwrap_or("the calendar");

    let mut lines = Vec::new();
    lines.push(format!(
        "The {} appeared {} time(s) across {} month(s) in {}.",
        subject_label,
        matches.len(),
        months_seen.len(),
        year_label,
    ));

    // List the months with the matching event text and day
    let mut month_list: Vec<String> = months_seen.into_iter().collect();
    month_list.sort();
    lines.push(String::new());
    lines.push("Months:".to_string());
    for m in &month_list {
        let event = matches
            .iter()
            .find(|p| p.month_section == *m);
        let event_name = event.map(|p| strip_leading_day(&p.answer)).unwrap_or_else(|| "—".to_string());
        let day_num = event.map(|p| p.day).unwrap_or(0);
        let date_label = if day_num > 0 {
            format_date(day_num, m)
        } else {
            m.clone()
        };
        lines.push(format!("  • {} — {}", date_label, event_name));
    }

    Some(lines.join("\n"))
}

// ────────────────────────────────────────────────────────────────────────
// Section-aware month lookup
// ────────────────────────────────────────────────────────────────────────

/// Detect a "MONTH [YEAR]" pattern in the question, then use the
/// `month_section` field on each QA pair to return every event in that section.
fn try_month_section_lookup(question: &str, qa_pairs: &[QAPair]) -> Option<String> {
    let q_lower = question.to_lowercase();

    // Find which month the user asked about
    let target_month = MONTHS.iter().find(|m| q_lower.contains(**m))?;

    // Extract a 4-digit year if present
    let target_year: Option<&str> = q_lower
        .split(|c: char| !c.is_ascii_digit())
        .find(|w| w.len() == 4);

    // Build the section header we're looking for, e.g. "JANUARY 2025"
    let header_upper = match target_year {
        Some(y) => format!("{} {}", target_month, y).to_uppercase(),
        None => target_month.to_uppercase(),
    };

    // Filter QA pairs whose month_section matches
    let mut events: Vec<(String, u8)> = Vec::new(); // (event_text, day)
    let mut seen_texts: Vec<String> = Vec::new();
    for pair in qa_pairs {
        if pair.month_section.is_empty() {
            continue;
        }
        let section_match = if target_year.is_some() {
            pair.month_section == header_upper
        } else {
            pair.month_section.to_uppercase().starts_with(&header_upper)
        };

        if section_match {
            let ans = strip_leading_day(&pair.answer);
            // Skip the month header itself and empty answers
            if !ans.is_empty()
                && !is_month_year_header(&ans.to_lowercase())
                && !seen_texts.contains(&ans)
            {
                seen_texts.push(ans.clone());
                events.push((ans, pair.day));
            }
        }
    }

    if events.is_empty() {
        return None; // fall through to generic search
    }

    let mut lines: Vec<String> = vec![format!("Events in {}:", header_upper)];
    for (text, day) in &events {
        if *day > 0 {
            lines.push(format!("  • {} — {}", format_date(*day, &header_upper), text));
        } else {
            lines.push(format!("  • {}", text));
        }
    }
    Some(lines.join("\n"))
}

/// Format a day + month_section (e.g. 23, "MARCH 2024") into "23 March 2024".
fn format_date(day: u8, month_section: &str) -> String {
    // month_section is like "JANUARY 2025" — title-case it
    let parts: Vec<&str> = month_section.split_whitespace().collect();
    let month_tc = parts.first().map(|m| {
        let mut c = m.chars();
        match c.next() {
            None => String::new(),
            Some(f) => f.to_uppercase().to_string() + &c.as_str().to_lowercase(),
        }
    }).unwrap_or_default();
    let year = parts.get(1).copied().unwrap_or("");
    if day > 0 && !year.is_empty() {
        format!("{} {} {}", day, month_tc, year)
    } else if day > 0 {
        format!("{} {}", day, month_tc)
    } else if !year.is_empty() {
        format!("{} {}", month_tc, year)
    } else {
        month_tc
    }
}

/// Returns true if the text looks like a standalone month-year header,
/// e.g. "january 2025", "march 2026".
fn is_month_year_header(text: &str) -> bool {
    let trimmed = text.trim();
    MONTHS.iter().any(|m| {
        trimmed.starts_with(m)
            && (trimmed.len() == m.len()          // just the month
                || trimmed[m.len()..].trim().chars().all(|c| c.is_ascii_digit()))
    })
}

/// Strip a leading day number from event text, e.g.
/// "15WCED SCHOOLS OPEN" → "WCED SCHOOLS OPEN",
/// "29START OF TERM 1"   → "START OF TERM 1".
fn strip_leading_day(text: &str) -> String {
    let t = text.trim();
    let digits: String = t.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.len() >= 1 && digits.len() <= 2 {
        if let Ok(n) = digits.parse::<u8>() {
            if (1..=31).contains(&n) {
                let rest = t[digits.len()..].trim();
                if !rest.is_empty() {
                    return rest.to_string();
                }
            }
        }
    }
    t.to_string()
}

// ────────────────────────────────────────────────────────────────────────
// Generic keyword + embedding search (fallback)
// ────────────────────────────────────────────────────────────────────────

fn keyword_and_embedding_search(question: &str, qa_pairs: &[QAPair]) -> String {
    let device = <NdArray as burn::prelude::Backend>::Device::default();
    let config = TrainingConfig::default();
    let tokenizer = Tokenizer::load("checkpoints/tokenizer.json");

    let keywords = extract_keywords(question);
    let mut scored: Vec<(usize, f64)> = qa_pairs
        .iter()
        .enumerate()
        .map(|(i, pair)| {
            let score = keyword_score(&keywords, &pair.question, &pair.answer);
            (i, score)
        })
        .filter(|(_, s)| *s > 0.0)
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let candidates: Vec<usize> = scored.iter().take(20).map(|(i, _)| *i).collect();
    let candidates = if candidates.is_empty() {
        (0..qa_pairs.len()).collect::<Vec<_>>()
    } else {
        candidates
    };

    // Load model for embedding re-ranking
    let model_config = TransformerModelConfig {
        vocab_size: tokenizer.vocab_size,
        max_seq_len: config.max_seq_len,
        d_model: config.d_model,
        n_heads: config.n_heads,
        d_ff: config.d_ff,
        n_layers: config.n_layers,
    };
    let model: TransformerModel<Backend> = model_config.init(&device);
    let model = model
        .load_file(
            "checkpoints/model",
            &DefaultFileRecorder::<FullPrecisionSettings>::new(),
            &device,
        )
        .expect("Failed to load model weights");

    let q_emb = get_embedding(&model, &tokenizer, question, config.max_seq_len, &device);

    let mut ranked: Vec<(usize, f64)> = candidates
        .iter()
        .map(|&idx| {
            let pair = &qa_pairs[idx];
            let a_emb = get_embedding(
                &model, &tokenizer, &pair.answer, config.max_seq_len, &device,
            );
            let sim = cosine_similarity(&q_emb, &a_emb);
            let kw = scored.iter().find(|(i, _)| *i == idx).map_or(0.0, |(_, s)| *s);
            (idx, kw + sim)
        })
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Filter out standalone month headers from results
    let mut seen = std::collections::HashSet::new();
    let mut answer_parts: Vec<String> = Vec::new();

    for &(idx, _) in ranked.iter().take(10) {
        let ans = qa_pairs[idx].answer.trim().to_string();
        if ans.is_empty() || is_month_year_header(&ans.to_lowercase()) {
            continue;
        }
        if seen.insert(ans.clone()) {
            answer_parts.push(format!("• {}", ans));
        }
        if answer_parts.len() >= 5 {
            break;
        }
    }

    if answer_parts.is_empty() {
        "I could not find relevant information for your question.".to_string()
    } else {
        answer_parts.join("\n")
    }
}

// ── Helpers ────────────────────────────────────────────────────────────

fn extract_keywords(text: &str) -> Vec<String> {
    let stopwords = [
        "what", "which", "when", "where", "who", "how", "is", "are", "was",
        "were", "the", "a", "an", "in", "on", "at", "for", "of", "and",
        "or", "to", "do", "does", "did", "be", "been", "being", "have",
        "has", "had", "it", "its", "this", "that", "these", "those",
        "with", "from", "by", "about", "into", "there", "can", "will",
        "events", "described", "information", "mentioned", "paragraph",
    ];
    text.split(|c: char| !c.is_alphanumeric())
        .map(|w| w.to_lowercase())
        .filter(|w| w.len() > 1 && !stopwords.contains(&w.as_str()))
        .collect()
}

fn keyword_score(keywords: &[String], question: &str, answer: &str) -> f64 {
    let q_lower = question.to_lowercase();
    let a_lower = answer.to_lowercase();
    let mut score = 0.0;
    for kw in keywords {
        if q_lower.contains(kw.as_str()) {
            score += 2.0;
        }
        if a_lower.contains(kw.as_str()) {
            score += 1.0;
        }
    }
    score
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }
    (dot / (mag_a * mag_b)) as f64
}

fn get_embedding(
    model: &TransformerModel<Backend>,
    tokenizer: &Tokenizer,
    text: &str,
    max_len: usize,
    device: &<Backend as burn::prelude::Backend>::Device,
) -> Vec<f32> {
    let input_ids = tokenizer.encode(text, max_len);
    let input_data: Vec<i64> = input_ids.iter().map(|&x| x as i64).collect();
    let input_tensor = Tensor::<Backend, 1, Int>::from_ints(
        input_data.as_slice(),
        device,
    )
    .reshape([1, max_len]);

    let hidden = model.embeddings.forward(input_tensor);
    let mut x = hidden;
    for layer in &model.layers {
        x = layer.forward(x);
    }

    // Grab the dimension BEFORE consuming x
    let d_model = x.dims()[2];
    let pooled = x.mean_dim(1).reshape([d_model]);
    pooled.into_data().to_vec().unwrap()
}