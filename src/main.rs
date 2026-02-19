mod data;
mod model;
mod training;
mod inference;
mod utils;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "word-doc-qa")]
#[command(about = "A Q&A system trained on .docx files using Burn")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate Q&A pairs from .docx files
    Generate {
        #[arg(short, long, default_value = "data/raw")]
        input: String,
        #[arg(short, long, default_value = "data/processed/qa_pairs.json")]
        output: String,
    },
    /// Train the transformer model
    Train {
        #[arg(short, long, default_value = "data/processed/qa_pairs.json")]
        data: String,
        #[arg(short, long, default_value_t = 50)]
        epochs: usize,
    },
    /// Ask a question to the trained model
    Ask {
        #[arg(short, long)]
        question: String,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate { input, output } => {
            println!("Loading .docx files from: {}", input);
            let documents = data::docx_loader::load_all_docx(&input);
            println!("Loaded {} documents.", documents.len());

            let qa_pairs = data::dataset::generate_qa_pairs(&documents);
            println!("Generated {} Q&A pairs.", qa_pairs.len());

            let json = serde_json::to_string_pretty(&qa_pairs).expect("Failed to serialize");
            std::fs::create_dir_all("data/processed").ok();
            std::fs::write(&output, json).expect("Failed to write Q&A pairs");
            println!("Saved Q&A pairs to {}", output);
        }
        Commands::Train { data, epochs } => {
            println!("Training with data from: {} for {} epochs", data, epochs);
            training::train::run_training(&data, epochs);
        }
        Commands::Ask { question } => {
            println!("Question: {}", question);
            let answer = inference::qa::answer_question(&question);
            println!("Answer: {}", answer);
        }
    }
}