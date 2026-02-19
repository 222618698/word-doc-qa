use burn::prelude::*;
use burn::backend::NdArray;

use crate::data::tokenizer::Tokenizer;
use crate::data::dataset::QAPair;
use crate::model::transformer::{TransformerModel, TransformerModelConfig};
use crate::training::config::TrainingConfig;

type Backend = NdArray;

/// Loads the trained model and answers a question using retrieval-based matching.
pub fn answer_question(question: &str) -> String {
    let config = TrainingConfig::default();
    let device = <Backend as burn::prelude::Backend>::Device::default();

    // Load tokenizer
    let tokenizer_path = "checkpoints/tokenizer.json";
    if !std::path::Path::new(tokenizer_path).exists() {
        return "Error: No tokenizer found. Train the model first.".to_string();
    }
    let tokenizer = Tokenizer::load(tokenizer_path);

    // Load Q&A pairs
    let qa_path = "data/processed/qa_pairs.json";
    if !std::path::Path::new(qa_path).exists() {
        return "Error: No Q&A data found. Run 'generate' first.".to_string();
    }
    let qa_content = std::fs::read_to_string(qa_path).expect("Failed to read QA data");
    let qa_pairs: Vec<QAPair> = serde_json::from_str(&qa_content).expect("Failed to parse QA data");

    if qa_pairs.is_empty() {
        return "No Q&A pairs available.".to_string();
    }

    // Build model
    let model_config = TransformerModelConfig {
        vocab_size: tokenizer.vocab_size,
        max_seq_len: config.max_seq_len,
        d_model: config.d_model,
        n_heads: config.n_heads,
        d_ff: config.d_ff,
        n_layers: config.n_layers,
    };

    let model: TransformerModel<Backend> = model_config.init(&device);

    // Load trained weights
    let model_path = "checkpoints/model";
    let model: TransformerModel<Backend> = model
        .load_file(
            model_path,
            &burn::record::DefaultFileRecorder::<burn::record::FullPrecisionSettings>::new(),
            &device,
        )
        .unwrap_or_else(|e| {
            eprintln!("Warning: Could not load model weights ({}). Using random weights.", e);
            model_config.init(&device)
        });

    // Get embedding for the user's question
    let question_emb = get_embedding(&model, &tokenizer, question, config.max_seq_len, &device);

    // Find the most similar question in the dataset
    let mut best_score = f32::NEG_INFINITY;
    let mut best_idx = 0;

    for (i, pair) in qa_pairs.iter().enumerate() {
        let pair_emb = get_embedding(&model, &tokenizer, &pair.question, config.max_seq_len, &device);
        let score = cosine_similarity(&question_emb, &pair_emb);

        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }

    let best_pair = &qa_pairs[best_idx];

    // Also find top 3 for context
    let mut scored: Vec<(usize, f32)> = qa_pairs
        .iter()
        .enumerate()
        .map(|(i, pair)| {
            let emb = get_embedding(&model, &tokenizer, &pair.question, config.max_seq_len, &device);
            (i, cosine_similarity(&question_emb, &emb))
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut result = format!("Best match (score: {:.4}):\n", best_score);
    result.push_str(&format!("  Q: {}\n", best_pair.question));
    result.push_str(&format!("  A: {}\n", best_pair.answer));
    result.push_str(&format!("  Source: {}\n", best_pair.source));

    if scored.len() > 1 {
        result.push_str("\nOther relevant matches:\n");
        for &(idx, score) in scored.iter().skip(1).take(2) {
            let pair = &qa_pairs[idx];
            result.push_str(&format!("  [{:.4}] {}\n", score, pair.answer));
        }
    }

    result
}

/// Get the mean-pooled embedding from the transformer for a text.
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

    // Forward pass through embeddings + transformer layers (not the output projection)
    let hidden = model.embeddings.forward(input_tensor);
    let mut x = hidden;
    for layer in &model.layers {
        x = layer.forward(x);
    }

    // Mean pooling over sequence length → [1, d_model] → [d_model]
    let pooled = x.mean_dim(1).reshape([x.dims()[2]]);
    pooled.into_data().to_vec().unwrap()
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}