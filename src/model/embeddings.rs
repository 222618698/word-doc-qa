use burn::prelude::*;
use burn::nn;

/// Token embedding + positional embedding layer.
#[derive(Module, Debug)]
pub struct Embeddings<B: Backend> {
    pub token_embedding: nn::Embedding<B>,
    pub position_embedding: nn::Embedding<B>,
}

/// Configuration for building embeddings.
#[derive(Config, Debug)]
pub struct EmbeddingsConfig {
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub d_model: usize,
}

impl EmbeddingsConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Embeddings<B> {
        let token_embedding =
            nn::EmbeddingConfig::new(self.vocab_size, self.d_model).init(device);
        let position_embedding =
            nn::EmbeddingConfig::new(self.max_seq_len, self.d_model).init(device);

        Embeddings {
            token_embedding,
            position_embedding,
        }
    }
}

impl<B: Backend> Embeddings<B> {
    /// Forward pass: adds token embeddings and positional embeddings.
    ///
    /// `input_ids`: [batch_size, seq_len] integer tensor
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let device = input_ids.device();
        let [_batch_size, seq_len] = input_ids.dims();

        // Create position indices [0, 1, 2, ..., seq_len-1] — use i64
        let positions: Vec<i64> = (0..seq_len as i64).collect();
        let pos_tensor =
            Tensor::<B, 1, Int>::from_ints(positions.as_slice(), &device)
                .unsqueeze::<2>();  // [1, seq_len]

        let token_emb = self.token_embedding.forward(input_ids);  // [B, S, D]
        let pos_emb = self.position_embedding.forward(pos_tensor); // [1, S, D]

        token_emb + pos_emb
    }
}