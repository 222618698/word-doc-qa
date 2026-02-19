use burn::prelude::*;
use burn::backend::NdArray;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::data::dataset::Dataset;
use burn::tensor::loss::cross_entropy_with_logits;

use crate::data::dataset::QADataset;
use crate::data::tokenizer::Tokenizer;
use crate::model::transformer::{TransformerModel, TransformerModelConfig};
use crate::training::config::TrainingConfig;

type Backend = burn::backend::Autodiff<NdArray>;

/// Runs the full training loop.
pub fn run_training(data_path: &str, epochs: usize) {
    let config = TrainingConfig {
        epochs,
        ..Default::default()
    };

    let device = <NdArray as burn::prelude::Backend>::Device::default();

    // Build tokenizer
    let tokenizer = Tokenizer::default_ascii();
    println!("Vocab size: {}", tokenizer.vocab_size);

    // Save tokenizer for inference
    std::fs::create_dir_all("checkpoints").ok();
    tokenizer.save("checkpoints/tokenizer.json");

    // Load dataset
    let dataset = QADataset::from_json(data_path, &tokenizer, config.max_seq_len);
    println!("Dataset size: {} samples", dataset.len());

    if dataset.len() == 0 {
        eprintln!("No training data found. Run 'generate' first.");
        return;
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

    let mut model: TransformerModel<Backend> = model_config.init(&device);

    // Optimizer
    let mut optimizer = AdamConfig::new()
        .with_epsilon(1e-8)
        .init::<Backend, TransformerModel<Backend>>();

    let lr = config.learning_rate;

    // Training loop
    for epoch in 0..config.epochs {
        let mut total_loss = 0.0;
        let mut batch_count = 0;

        let mut i = 0;
        while i < dataset.len() {
            let batch_end = (i + config.batch_size).min(dataset.len());
            let batch_items: Vec<_> = (i..batch_end)
                .filter_map(|idx| dataset.get(idx))
                .collect();

            if batch_items.is_empty() {
                i += config.batch_size;
                continue;
            }

            let batch_size = batch_items.len();
            let seq_len = config.max_seq_len;

            // Build input tensor [batch, seq_len] — use i64
            let input_data: Vec<i64> = batch_items
                .iter()
                .flat_map(|item| item.input_ids.iter().map(|&x| x as i64))
                .collect();
            let input_tensor = Tensor::<Backend, 1, Int>::from_ints(
                input_data.as_slice(),
                &device,
            )
            .reshape([batch_size, seq_len]);

            // Build target tensor [batch, seq_len] — use i64
            let target_data: Vec<i64> = batch_items
                .iter()
                .flat_map(|item| item.target_ids.iter().map(|&x| x as i64))
                .collect();
            let target_tensor = Tensor::<Backend, 1, Int>::from_ints(
                target_data.as_slice(),
                &device,
            )
            .reshape([batch_size, seq_len]);

            // Forward pass → logits [batch, seq_len, vocab_size]
            let logits = model.forward(input_tensor);

            // Reshape for cross-entropy: [batch * seq_len, vocab_size]
            let vocab_size = tokenizer.vocab_size;
            let logits_flat = logits.reshape([batch_size * seq_len, vocab_size]);
            let target_flat = target_tensor.reshape([batch_size * seq_len]);

            // One-hot encode targets for cross_entropy_with_logits
            let targets_one_hot = one_hot::<Backend>(target_flat, vocab_size, &device);

            let loss = cross_entropy_with_logits(logits_flat, targets_one_hot);

            let loss_val: f32 = loss.clone().into_data().to_vec().unwrap()[0];
            total_loss += loss_val as f64;
            batch_count += 1;

            // Backward
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(lr, model, grads);

            i += config.batch_size;
        }

        let avg_loss = if batch_count > 0 {
            total_loss / batch_count as f64
        } else {
            0.0
        };
        println!("Epoch {}/{} — Loss: {:.6}", epoch + 1, config.epochs, avg_loss);
    }

    // Save model
    model
        .save_file("checkpoints/model", &burn::record::DefaultFileRecorder::<burn::record::FullPrecisionSettings>::new())
        .expect("Failed to save model");
    println!("Model saved to checkpoints/model");
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