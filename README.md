# Word Document Q&A System

A Question-Answering system built with Rust and the Burn deep learning framework that reads Word documents (.docx) and answers questions about their content.

## Overview

This project implements a complete ML pipeline:
1. Data loading and processing from .docx files
2. Transformer-based neural network training
3. Natural language question answering
4. Command-line interface for training and inference

## Built With

- [Rust](https://www.rust-lang.org/) (Edition 2021)
- [Burn Framework](https://burn.dev/) v0.20.1 - Deep learning framework
- docx-rs v0.4 - Word document parsing
- tokenizers v0.15 - Text tokenization
- clap v4 - Command-line argument parsing
- serde & serde_json v1.0 - JSON serialization

## Prerequisites

Before running this project, you need to install:

### 1. Rust and Cargo
Install Rust using rustup:

**Windows:**
- Download and run [rustup-init.exe](https://rustup.rs/)

**macOS/Linux:**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Verify installation:
```bash
rustc --version
cargo --version
```

### 2. GPU Support (Optional but Recommended)
This project uses the `wgpu` backend for GPU acceleration.

**Requirements:**
- Vulkan, Metal (macOS), or DirectX 12 (Windows) compatible GPU
- Up-to-date graphics drivers

**Windows:**
- Install the latest GPU drivers from your manufacturer (NVIDIA, AMD, or Intel)

**Linux:**
- Install Vulkan support: `sudo apt install vulkan-tools libvulkan-dev` (Ubuntu/Debian)

**macOS:**
- Metal is included by default on macOS

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/222618698/word-doc-qa.git
cd word-doc-qa
```

2. **Build the project:**
```bash
cargo build --release
```

This will download all dependencies and compile the project. First build may take 10-15 minutes.

## Project Structure

```
word-doc-qa/
├── src/
│   ├── main.rs           # CLI entry point
│   ├── data/             # Data loading and processing
│   ├── model/            # Transformer model architecture
│   ├── training/         # Training loop and utilities
│   ├── inference/        # Question answering logic
│   └── utils/            # Helper functions
├── data/
│   ├── raw/              # Place your .docx files here
│   └── processed/        # Generated Q&A pairs
├── Cargo.toml            # Dependencies and project config
└── README.md
```

## Usage

The application has three main commands:

### 1. Generate Q&A Pairs from Documents

First, place your `.docx` files in the `data/raw/` directory, then:

```bash
# Using cargo run
cargo run --release -- generate

# Or with custom paths
cargo run --release -- generate --input data/raw --output data/processed/qa_pairs.json

# Using the compiled binary
./target/release/word-doc-qa generate
```

This will:
- Read all .docx files from the input directory
- Extract text content
- Generate question-answer pairs
- Save them to `data/processed/qa_pairs.json`

### 2. Train the Model

Train the transformer model on your Q&A pairs:

```bash
# Train with default settings (50 epochs)
cargo run --release -- train

# Custom training
cargo run --release -- train --data data/processed/qa_pairs.json --epochs 100

# Using the compiled binary
./target/release/word-doc-qa train --epochs 100
```

Training parameters:
- `--data`: Path to Q&A pairs JSON file (default: `data/processed/qa_pairs.json`)
- `--epochs`: Number of training epochs (default: 50)

### 3. Ask Questions

After training, ask questions about your documents:

```bash
# Ask a question
cargo run --release -- ask --question "What is the main topic of the document?"

# Using the compiled binary
./target/release/word-doc-qa ask --question "Your question here"
```

## Example Workflow

```bash
# 1. Create data directory and add your .docx files
mkdir -p data/raw
# (Copy your .docx files to data/raw/)

# 2. Generate Q&A pairs
cargo run --release -- generate

# 3. Train the model
cargo run --release -- train --epochs 50

# 4. Ask questions
cargo run --release -- ask --question "What is this document about?"
```

## Dependencies Explained

| Dependency | Version | Purpose |
|------------|---------|---------|
| **burn** | 0.20.1 | Deep learning framework with features: `train`, `wgpu`, `autodiff`, `ndarray` |
| **docx-rs** | 0.4 | Parse and read .docx (Word document) files |
| **tokenizers** | 0.15 | Tokenize text for transformer model input |
| **serde** | 1.0 | Serialize/deserialize data structures |
| **serde_json** | 1.0 | JSON parsing and generation |
| **clap** | 4.x | Command-line argument parsing |
| **rand** | 0.8 | Random number generation for training |

## Troubleshooting

### Build Issues

**Problem:** Long compilation time
- **Solution:** This is normal for the first build. Rust compiles all dependencies. Use `--release` flag for optimized builds.

**Problem:** Out of memory during compilation
- **Solution:** Close other applications or increase system swap space.

### Runtime Issues

**Problem:** GPU/WGPU errors
- **Solution:** Update graphics drivers or disable GPU acceleration by modifying `Cargo.toml` to remove the `wgpu` feature.

**Problem:** Cannot find .docx files
- **Solution:** Ensure files are in `data/raw/` or specify correct path with `--input` flag.

## Development

Run in development mode (faster compilation, slower execution):
```bash
cargo run -- generate
cargo run -- train --epochs 10
cargo run -- ask --question "test"
```

Run tests:
```bash
cargo test
```

Check code:
```bash
cargo check
cargo clippy
```

## Performance Tips

1. **Use release builds** for training: `cargo build --release`
2. **GPU acceleration** significantly speeds up training
3. **Adjust batch size** in training configuration for your hardware
4. **Start with fewer epochs** to test the pipeline before full training

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license here]

## Acknowledgments

- Built with the [Burn](https://burn.dev/) deep learning framework
- Uses the Rust programming language
## Built With

- [Rust](https://www.rust-lang.org/)
- [Burn Framework](https://burn.dev/) v0.20.1
- docx-rs for Word document parsing
- tokenizers for text tokenization
