use std::collections::HashMap;

/// A simple character-level tokenizer.
///
/// For production, replace with a BPE or WordPiece tokenizer.
pub struct Tokenizer {
    pub char_to_id: HashMap<char, usize>,
    pub id_to_char: HashMap<usize, char>,
    pub vocab_size: usize,
    pub pad_id: usize,
    pub unk_id: usize,
}

impl Tokenizer {
    /// Builds a tokenizer from a corpus of text.
    pub fn from_corpus(texts: &[String]) -> Self {
        let mut char_set: Vec<char> = Vec::new();

        // Reserve special tokens
        // 0 = <PAD>, 1 = <UNK>, 2 = <SOS>, 3 = <EOS>
        let special_count = 4;

        for text in texts {
            for c in text.chars() {
                if !char_set.contains(&c) {
                    char_set.push(c);
                }
            }
        }

        char_set.sort();

        let mut char_to_id = HashMap::new();
        let mut id_to_char = HashMap::new();

        for (i, &c) in char_set.iter().enumerate() {
            let id = i + special_count;
            char_to_id.insert(c, id);
            id_to_char.insert(id, c);
        }

        let vocab_size = char_set.len() + special_count;

        Tokenizer {
            char_to_id,
            id_to_char,
            vocab_size,
            pad_id: 0,
            unk_id: 1,
        }
    }

    /// Builds a default tokenizer covering printable ASCII.
    pub fn default_ascii() -> Self {
        let chars: Vec<String> = (32u8..=126)
            .map(|b| String::from(b as char))
            .collect();
        Self::from_corpus(&chars)
    }

    /// Encode a string into token IDs, padded/truncated to `max_len`.
    pub fn encode(&self, text: &str, max_len: usize) -> Vec<usize> {
        let mut ids: Vec<usize> = text
            .chars()
            .map(|c| *self.char_to_id.get(&c).unwrap_or(&self.unk_id))
            .collect();

        ids.truncate(max_len);

        while ids.len() < max_len {
            ids.push(self.pad_id);
        }

        ids
    }

    /// Decode token IDs back into a string.
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .filter(|&&id| id != self.pad_id)
            .map(|&id| {
                self.id_to_char.get(&id).copied().unwrap_or('?')
            })
            .collect()
    }

    /// Save tokenizer vocabulary to JSON.
    pub fn save(&self, path: &str) {
        let map: HashMap<String, usize> = self
            .char_to_id
            .iter()
            .map(|(c, id)| (c.to_string(), *id))
            .collect();
        let json = serde_json::to_string_pretty(&map).unwrap();
        std::fs::write(path, json).expect("Failed to save tokenizer");
    }

    /// Load tokenizer vocabulary from JSON.
    pub fn load(path: &str) -> Self {
        let content = std::fs::read_to_string(path).expect("Failed to read tokenizer");
        let map: HashMap<String, usize> = serde_json::from_str(&content).unwrap();

        let mut char_to_id = HashMap::new();
        let mut id_to_char = HashMap::new();
        let mut max_id = 0;

        for (s, id) in &map {
            if let Some(c) = s.chars().next() {
                char_to_id.insert(c, *id);
                id_to_char.insert(*id, c);
                if *id > max_id {
                    max_id = *id;
                }
            }
        }

        Tokenizer {
            char_to_id,
            id_to_char,
            vocab_size: max_id + 1,
            pad_id: 0,
            unk_id: 1,
        }
    }
}