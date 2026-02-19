# Word-Doc QA — Assignment Report

## Overview
This project implements a Question-Answering system in Rust using the Burn deep learning framework.
It reads `.docx` calendar files, generates Q&A training pairs, trains a small Transformer model,
and supports interactive question answering.

## Architecture

### Data Pipeline
1. **DOCX Loader** — Extracts plain text from `.docx` files by parsing `word/document.xml` inside the ZIP archive.
2. **QA Generator** — Splits text into paragraphs and creates question-answer pairs.
3. **Tokenizer** — Character-level tokenizer that maps characters to integer IDs.

### Model
- **Embeddings** — Token embeddings + learnable positional embeddings.
- **Transformer Layers** — Self-attention → Add & Norm → Feed-forward → Add & Norm.
- **Output Projection** — Linear layer mapping hidden states to vocabulary logits.

### Training
- Optimizer: Adam (lr = 1e-3)
- Loss: Cross-entropy
- Default: 2 layers, d_model=64, 4 heads, d_ff=128

### Inference
- Greedy decoding (argmax at each token position)

## Results
_(Fill in after training)_

## References
- [Burn Framework](https://burn.dev)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)