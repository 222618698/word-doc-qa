use std::fs;
use std::path::Path;

/// Represents a loaded document with its filename and extracted text.
#[derive(Debug, Clone)]
pub struct Document {
    pub filename: String,
    pub content: String,
}

/// Loads all .docx files from the given directory.
pub fn load_all_docx(dir: &str) -> Vec<Document> {
    let mut documents = Vec::new();
    let path = Path::new(dir);

    if !path.exists() {
        eprintln!("Directory does not exist: {}", dir);
        return documents;
    }

    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            let file_path = entry.path();
            if file_path.extension().and_then(|e| e.to_str()) == Some("docx") {
                match extract_text_from_docx(&file_path) {
                    Ok(text) => {
                        let filename = file_path
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .to_string();
                        println!("  Loaded: {} ({} chars)", filename, text.len());
                        documents.push(Document {
                            filename,
                            content: text,
                        });
                    }
                    Err(e) => {
                        eprintln!(
                            "  Failed to read {}: {}",
                            file_path.display(),
                            e
                        );
                    }
                }
            }
        }
    }

    documents
}

/// Extracts plain text from a .docx file using the docx-rs crate.
///
/// Iterates over document children (paragraphs, tables) and collects
/// all text runs into a single string separated by newlines.
fn extract_text_from_docx(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    let docx = docx_rs::read_docx(&bytes)
        .map_err(|e| format!("docx-rs parse error: {:?}", e))?;

    let mut lines: Vec<String> = Vec::new();

    for child in docx.document.children {
        match child {
            docx_rs::DocumentChild::Paragraph(para) => {
                let line = extract_paragraph_text(&para);
                if !line.is_empty() {
                    lines.push(line);
                }
            }
            docx_rs::DocumentChild::Table(table) => {
                // Extract text from table cells as well
                for row in &table.rows {
                    match row {
                        docx_rs::TableChild::TableRow(tr) => {
                            for cell in &tr.cells {
                                match cell {
                                    docx_rs::TableRowChild::TableCell(tc) => {
                                        for tc_child in &tc.children {
                                            if let docx_rs::TableCellContent::Paragraph(p) = tc_child {
                                                let line = extract_paragraph_text(p);
                                                if !line.is_empty() {
                                                    lines.push(line);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    Ok(lines.join("\n"))
}

/// Extract text from a single paragraph by iterating its children (runs).
fn extract_paragraph_text(para: &docx_rs::Paragraph) -> String {
    let mut text = String::new();
    for child in &para.children {
        match child {
            docx_rs::ParagraphChild::Run(run) => {
                for run_child in &run.children {
                    match run_child {
                        docx_rs::RunChild::Text(t) => {
                            text.push_str(&t.text);
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    text
}