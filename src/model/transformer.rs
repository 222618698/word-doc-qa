use burn::prelude::*;
use burn::nn;
use crate::model::embeddings::{Embeddings, EmbeddingsConfig};

/// A single Transformer encoder layer.
#[derive(Module, Debug)]
pub struct TransformerLayer<B: Backend> {
    pub self_attn_query: nn::Linear<B>,
    pub self_attn_key: nn::Linear<B>,
    pub self_attn_value: nn::Linear<B>,
    pub self_attn_out: nn::Linear<B>,
    pub ff1: nn::Linear<B>,
    pub ff2: nn::Linear<B>,
    pub norm1: nn::LayerNorm<B>,
    pub norm2: nn::LayerNorm<B>,
}

#[derive(Config, Debug)]
pub struct TransformerLayerConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
}

impl TransformerLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerLayer<B> {
        TransformerLayer {
            self_attn_query: nn::LinearConfig::new(self.d_model, self.d_model).init(device),
            self_attn_key: nn::LinearConfig::new(self.d_model, self.d_model).init(device),
            self_attn_value: nn::LinearConfig::new(self.d_model, self.d_model).init(device),
            self_attn_out: nn::LinearConfig::new(self.d_model, self.d_model).init(device),
            ff1: nn::LinearConfig::new(self.d_model, self.d_ff).init(device),
            ff2: nn::LinearConfig::new(self.d_ff, self.d_model).init(device),
            norm1: nn::LayerNormConfig::new(self.d_model).init(device),
            norm2: nn::LayerNormConfig::new(self.d_model).init(device),
        }
    }
}

impl<B: Backend> TransformerLayer<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Self-attention
        let q = self.self_attn_query.forward(x.clone());
        let k = self.self_attn_key.forward(x.clone());
        let v = self.self_attn_value.forward(x.clone());

        let d_k = q.dims()[2] as f64;
        let scores = q.matmul(k.transpose()) / d_k.sqrt();
        let attn_weights = burn::tensor::activation::softmax(scores, 2);
        let attn_output = attn_weights.matmul(v);
        let attn_output = self.self_attn_out.forward(attn_output);

        // Add & Norm
        let x = self.norm1.forward(x + attn_output);

        // Feed-forward
        let ff_out = self.ff1.forward(x.clone());
        let ff_out = burn::tensor::activation::relu(ff_out);
        let ff_out = self.ff2.forward(ff_out);

        // Add & Norm
        self.norm2.forward(x + ff_out)
    }
}

/// Full Transformer model for Q&A.
#[derive(Module, Debug)]
pub struct TransformerModel<B: Backend> {
    pub embeddings: Embeddings<B>,
    pub layers: Vec<TransformerLayer<B>>,
    pub output_proj: nn::Linear<B>,
}

#[derive(Config, Debug)]
pub struct TransformerModelConfig {
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub n_layers: usize,
}

impl TransformerModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerModel<B> {
        let embeddings = EmbeddingsConfig {
            vocab_size: self.vocab_size,
            max_seq_len: self.max_seq_len,
            d_model: self.d_model,
        }
        .init(device);

        let layer_config = TransformerLayerConfig {
            d_model: self.d_model,
            n_heads: self.n_heads,
            d_ff: self.d_ff,
        };

        let layers: Vec<TransformerLayer<B>> = (0..self.n_layers)
            .map(|_| layer_config.init(device))
            .collect();

        let output_proj =
            nn::LinearConfig::new(self.d_model, self.vocab_size).init(device);

        TransformerModel {
            embeddings,
            layers,
            output_proj,
        }
    }
}

impl<B: Backend> TransformerModel<B> {
    /// Forward pass.
    ///
    /// `input_ids`: [batch_size, seq_len] — integer tensor of token IDs.
    ///
    /// Returns logits: [batch_size, seq_len, vocab_size].
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let mut x = self.embeddings.forward(input_ids);

        for layer in &self.layers {
            x = layer.forward(x);
        }

        self.output_proj.forward(x)
    }
}