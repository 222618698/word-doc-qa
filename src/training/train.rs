use burn::prelude::*;
use burn::backend::NdArray;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::data::dataset::Dataset;
use burn::tensor::loss::cross_entropy_with_logits;

use crate::data::dataset::{QAPair, QADataset};
use crate::data::tokenizer::Tokenizer;
use crate::model::transformer::{TransformerModel, TransformerModelConfig};
use crate::training::config::TrainingConfig;

type Backend = burn::backend::Autodiff<NdArray>;

/// Runs the full training loop with train/validation split,
/// loss reporting, accuracy tracking, and checkpoint saving.
pub fn run_training(data_path: &str, epochs: usize) {
    let config = TrainingConfig {
        epochs,
        ..Default::default()
    };

    let device = <NdArray as burn::prelude::Backend>::Device::default();

    // ── Build BPE tokenizer from the actual QA corpus ──────────────
    let qa_content = std::fs::read_to_string(data_path).expect("Failed to read QA JSON");
    let qa_pairs: Vec<QAPair> =
        serde_json::from_str(&qa_content).expect("Failed to parse QA JSON");

    let corpus: Vec<String> = qa_pairs
        .iter()
        .flat_map(|p| vec![p.question.clone(), p.answer.clone()])
        .collect();

    let tokenizer = Tokenizer::from_corpus(&corpus);
    println!("Vocab size: {}", tokenizer.vocab_size);

    // Save tokenizer for inference
    std::fs::create_dir_all("checkpoints").ok();
    tokenizer.save("checkpoints/tokenizer.json");

    // ── Load & split dataset ───────────────────────────────────────
    let full_dataset = QADataset::from_json(data_path, &tokenizer, config.max_seq_len);
    let total = full_dataset.len();
    println!("Total dataset size: {} samples", total);

    if total == 0 {
        eprintln!("No training data found. Run 'generate' first.");
        return;
    }

    // Deterministic train/validation split
    let val_count = ((total as f64) * config.val_split).max(1.0) as usize;
    let train_count = total - val_count;

    let train_items: Vec<_> = (0..train_count)
        .filter_map(|i| full_dataset.get(i))
        .collect();
    let val_items: Vec<_> = (train_count..total)
        .filter_map(|i| full_dataset.get(i))
        .collect();

    println!(
        "Split: {} training / {} validation samples",
        train_items.len(),
        val_items.len()
    );

    // ── Build model ────────────────────────────────────────────────
    let model_config = TransformerModelConfig {
        vocab_size: tokenizer.vocab_size,
        max_seq_len: config.max_seq_len,
        d_model: config.d_model,
        n_heads: config.n_heads,
        d_ff: config.d_ff,
        n_layers: config.n_layers,
    };

    let mut model: TransformerModel<Backend> = model_config.init(&device);

    // Optimizer
    let mut optimizer = AdamConfig::new()
        .with_epsilon(1e-8)
        .init::<Backend, TransformerModel<Backend>>();

    let lr = config.learning_rate;

    // ── Training loop ──────────────────────────────────────────────
    println!("\n{:<8} {:<14} {:<14} {:<12} {:<12}",
        "Epoch", "Train Loss", "Val Loss", "Train Acc", "Val Acc");
    println!("{}", "-".repeat(60));

    for epoch in 0..config.epochs {
        // ── Train ──────────────────────────────────────────────────
        let mut train_loss = 0.0;
        let mut train_correct = 0usize;
        let mut train_total_tokens = 0usize;
        let mut train_batches = 0;

        let mut i = 0;
        while i < train_items.len() {
            let batch_end = (i + config.batch_size).min(train_items.len());
            let batch: Vec<_> = train_items[i..batch_end].to_vec();

            if batch.is_empty() {
                i += config.batch_size;
                continue;
            }

            let batch_size = batch.len();
            let seq_len = config.max_seq_len;

            let input_data: Vec<i64> = batch
                .iter()
                .flat_map(|item| item.input_ids.iter().map(|&x| x as i64))
                .collect();
            let input_tensor = Tensor::<Backend, 1, Int>::from_ints(
                input_data.as_slice(),
                &device,
            )
            .reshape([batch_size, seq_len]);

            let target_data: Vec<i64> = batch
                .iter()
                .flat_map(|item| item.target_ids.iter().map(|&x| x as i64))
                .collect();
            let target_tensor = Tensor::<Backend, 1, Int>::from_ints(
                target_data.as_slice(),
                &device,
            )
            .reshape([batch_size, seq_len]);

            // Forward pass
            let logits = model.forward(input_tensor);

            let vocab_size = tokenizer.vocab_size;
            let logits_flat = logits.clone().reshape([batch_size * seq_len, vocab_size]);
            let target_flat = target_tensor.clone().reshape([batch_size * seq_len]);

            // Accuracy: argmax predictions vs targets
            let preds = logits_flat.clone().argmax(1).reshape([batch_size * seq_len]);
            let pred_data: Vec<i64> = preds.into_data().to_vec().unwrap();
            let tgt_data: Vec<i64> = target_flat.clone().into_data().to_vec().unwrap();
            for (p, t) in pred_data.iter().zip(tgt_data.iter()) {
                if *t != 0 {
                    // skip PAD tokens
                    train_total_tokens += 1;
                    if p == t {
                        train_correct += 1;
                    }
                }
            }

            let targets_one_hot = one_hot::<Backend>(target_flat, vocab_size, &device);
            let loss = cross_entropy_with_logits(logits_flat, targets_one_hot);

            let loss_val: f32 = loss.clone().into_data().to_vec().unwrap()[0];
            train_loss += loss_val as f64;
            train_batches += 1;

            // Backward + step
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(lr, model, grads);

            i += config.batch_size;
        }

        let avg_train_loss = if train_batches > 0 {
            train_loss / train_batches as f64
        } else {
            0.0
        };
        let train_acc = if train_total_tokens > 0 {
            train_correct as f64 / train_total_tokens as f64
        } else {
            0.0
        };

        // ── Validation ─────────────────────────────────────────────
        let (avg_val_loss, val_acc) = evaluate(
            &model,
            &val_items,
            &tokenizer,
            &config,
            &device,
        );

        println!(
            "{:<8} {:<14.6} {:<14.6} {:<12.4} {:<12.4}",
            epoch + 1,
            avg_train_loss,
            avg_val_loss,
            train_acc,
            val_acc,
        );
    }

    // ── Save model ─────────────────────────────────────────────────
    model
        .save_file(
            "checkpoints/model",
            &burn::record::DefaultFileRecorder::<burn::record::FullPrecisionSettings>::new(),
        )
        .expect("Failed to save model");
    println!("\nModel saved to checkpoints/model");
}

/// Run a forward pass on the validation set and return (avg_loss, accuracy).
fn evaluate(
    model: &TransformerModel<Backend>,
    items: &[crate::data::dataset::QAItem],
    tokenizer: &Tokenizer,
    config: &TrainingConfig,
    device: &<NdArray as burn::prelude::Backend>::Device,
) -> (f64, f64) {
    if items.is_empty() {
        return (0.0, 0.0);
    }

    let mut total_loss = 0.0;
    let mut batches = 0;
    let mut correct = 0usize;
    let mut total_tokens = 0usize;
    let seq_len = config.max_seq_len;
    let vocab_size = tokenizer.vocab_size;

    let mut i = 0;
    while i < items.len() {
        let batch_end = (i + config.batch_size).min(items.len());
        let batch: Vec<_> = items[i..batch_end].to_vec();
        if batch.is_empty() {
            i += config.batch_size;
            continue;
        }
        let batch_size = batch.len();

        let input_data: Vec<i64> = batch
            .iter()
            .flat_map(|item| item.input_ids.iter().map(|&x| x as i64))
            .collect();
        let input_tensor = Tensor::<Backend, 1, Int>::from_ints(
            input_data.as_slice(),
            device,
        )
        .reshape([batch_size, seq_len]);

        let target_data: Vec<i64> = batch
            .iter()
            .flat_map(|item| item.target_ids.iter().map(|&x| x as i64))
            .collect();
        let target_tensor = Tensor::<Backend, 1, Int>::from_ints(
            target_data.as_slice(),
            device,
        )
        .reshape([batch_size, seq_len]);

        // Forward (no gradients needed, but Autodiff backend is used)
        let logits = model.forward(input_tensor);
        let logits_flat = logits.reshape([batch_size * seq_len, vocab_size]);
        let target_flat = target_tensor.reshape([batch_size * seq_len]);

        // Accuracy
        let preds = logits_flat.clone().argmax(1).reshape([batch_size * seq_len]);
        let pred_data: Vec<i64> = preds.into_data().to_vec().unwrap();
        let tgt_data: Vec<i64> = target_flat.clone().into_data().to_vec().unwrap();
        for (p, t) in pred_data.iter().zip(tgt_data.iter()) {
            if *t != 0 {
                total_tokens += 1;
                if p == t {
                    correct += 1;
                }
            }
        }

        let targets_one_hot = one_hot::<Backend>(target_flat, vocab_size, device);
        let loss = cross_entropy_with_logits(logits_flat, targets_one_hot);
        let loss_val: f32 = loss.into_data().to_vec().unwrap()[0];
        total_loss += loss_val as f64;
        batches += 1;

        i += config.batch_size;
    }

    let avg_loss = if batches > 0 {
        total_loss / batches as f64
    } else {
        0.0
    };
    let acc = if total_tokens > 0 {
        correct as f64 / total_tokens as f64
    } else {
        0.0
    };

    (avg_loss, acc)
}

/// One-hot encode an integer tensor.
fn one_hot<B: burn::prelude::Backend>(
    indices: Tensor<B, 1, Int>,
    num_classes: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let len = indices.dims()[0];
    let idx_data: Vec<i64> = indices.into_data().to_vec().unwrap();

    let mut one_hot_data = vec![0.0f32; len * num_classes];
    for (i, &idx) in idx_data.iter().enumerate() {
        let idx = idx as usize;
        if idx < num_classes {
            one_hot_data[i * num_classes + idx] = 1.0;
        }
    }

    Tensor::<B, 1>::from_floats(one_hot_data.as_slice(), device)
        .reshape([len, num_classes])
}