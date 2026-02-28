use tokenizers::models::bpe::BPE;
use tokenizers::normalizers::Sequence as NormSeq;
use tokenizers::normalizers::strip::Strip;
use tokenizers::normalizers::unicode::NFC;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::{AddedToken, Tokenizer as HfTokenizer};

/// Wrapper around HuggingFace `tokenizers` crate providing a
/// BPE tokenizer with special tokens for the Q&A pipeline.
///
/// Special tokens:
///   0 = <PAD>, 1 = <UNK>, 2 = <SOS>, 3 = <EOS>
pub struct Tokenizer {
    inner: HfTokenizer,
    pub vocab_size: usize,
    pub pad_id: usize,
    pub unk_id: usize,
}

impl Tokenizer {
    // ── Construction ────────────────────────────────────────────────

    /// Train a BPE tokenizer from a slice of text strings.
    ///
    /// This replaces the old character-level constructor. The resulting
    /// tokenizer is ready for encoding immediately.
    pub fn from_corpus(texts: &[String]) -> Self {
        use tokenizers::models::bpe::BpeTrainer;
        use tokenizers::models::TrainerWrapper;

        let mut tokenizer = HfTokenizer::new(BPE::default());

        // Normalizers: strip whitespace edges, NFC unicode
        tokenizer.with_normalizer(NormSeq::new(vec![
            tokenizers::NormalizerWrapper::from(Strip::new(true, true)),
            tokenizers::NormalizerWrapper::from(NFC),
        ]));

        // Pre-tokenizer: split on whitespace
        tokenizer.with_pre_tokenizer(Whitespace {});

        // Special tokens
        let special = vec![
            AddedToken::from("<PAD>", true),
            AddedToken::from("<UNK>", true),
            AddedToken::from("<SOS>", true),
            AddedToken::from("<EOS>", true),
        ];
        tokenizer.add_special_tokens(&special);

        // BPE trainer
        let trainer = BpeTrainer::builder()
            .vocab_size(500)
            .min_frequency(2)
            .special_tokens(special.clone())
            .show_progress(false)
            .build();

        // Train on the provided texts
        let mut wrapper: TrainerWrapper = trainer.into();
        let _ = tokenizer.train(
            &mut wrapper,
            texts.iter().map(|s| s.as_str()),
        );

        // Post-processor: add SOS/EOS
        let _ = tokenizer.with_post_processor(
            TemplateProcessing::builder()
                .try_single("<SOS> $A <EOS>")
                .unwrap()
                .special_tokens(vec![
                    ("<SOS>", 2),
                    ("<EOS>", 3),
                ])
                .build()
                .unwrap(),
        );

        let vocab_size = tokenizer.get_vocab_size(true);

        Tokenizer {
            inner: tokenizer,
            vocab_size,
            pad_id: 0,
            unk_id: 1,
        }
    }

    /// Build a default tokenizer covering printable ASCII,
    /// suitable for small-scale character-level work.
    pub fn default_ascii() -> Self {
        let chars: Vec<String> = (32u8..=126)
            .map(|b| String::from(b as char))
            .collect();
        Self::from_corpus(&chars)
    }

    // ── Encode / Decode ────────────────────────────────────────────

    /// Encode a string into token IDs, padded/truncated to `max_len`.
    pub fn encode(&self, text: &str, max_len: usize) -> Vec<usize> {
        let encoding = self.inner.encode(text, false).unwrap_or_else(|_| {
            self.inner.encode("", false).unwrap()
        });

        let mut ids: Vec<usize> = encoding
            .get_ids()
            .iter()
            .map(|&id| id as usize)
            .collect();

        ids.truncate(max_len);

        while ids.len() < max_len {
            ids.push(self.pad_id);
        }

        ids
    }

    /// Decode token IDs back into a string.
    pub fn decode(&self, ids: &[usize]) -> String {
        let u32_ids: Vec<u32> = ids
            .iter()
            .filter(|&&id| id != self.pad_id)
            .map(|&id| id as u32)
            .collect();

        self.inner
            .decode(&u32_ids, true)
            .unwrap_or_default()
    }

    // ── Persistence ────────────────────────────────────────────────

    /// Save the tokenizer to a JSON file.
    pub fn save(&self, path: &str) {
        self.inner
            .save(path, false)
            .expect("Failed to save tokenizer");
    }

    /// Load a tokenizer from a JSON file previously saved.
    pub fn load(path: &str) -> Self {
        let inner = HfTokenizer::from_file(path)
            .expect("Failed to load tokenizer");
        let vocab_size = inner.get_vocab_size(true);
        Tokenizer {
            inner,
            vocab_size,
            pad_id: 0,
            unk_id: 1,
        }
    }
}