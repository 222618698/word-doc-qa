/// Hyperparameters for training.
pub struct TrainingConfig {
    pub learning_rate: f64,
    pub batch_size: usize,
    pub max_seq_len: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub n_layers: usize,
    pub epochs: usize,
    /// Fraction of data used for validation (0.0–1.0).
    pub val_split: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        TrainingConfig {
            learning_rate: 1e-3,
            batch_size: 8,
            max_seq_len: 128,
            d_model: 64,
            n_heads: 4,
            d_ff: 128,
            n_layers: 6,
            epochs: 50,
            val_split: 0.1,
        }
    }
}