use std::fs::{self, File};
use std::io::Read;
use std::path::Path;
use zip::ZipArchive;
use xml::reader::{EventReader, XmlEvent};

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

/// Extracts plain text from a .docx file by reading word/document.xml inside the ZIP.
fn extract_text_from_docx(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut archive = ZipArchive::new(file)?;

    let mut xml_content = String::new();
    {
        let mut doc_xml = archive.by_name("word/document.xml")?;
        doc_xml.read_to_string(&mut xml_content)?;
    }

    let mut text = String::new();
    let parser = EventReader::from_str(&xml_content);
    let mut in_text_element = false;

    for event in parser {
        match event {
            Ok(XmlEvent::StartElement { name, .. }) => {
                // <w:t> elements contain the actual text
                if name.local_name == "t" {
                    in_text_element = true;
                }
                // <w:p> = paragraph boundary → add newline
                if name.local_name == "p" && !text.is_empty() {
                    text.push('\n');
                }
            }
            Ok(XmlEvent::Characters(s)) if in_text_element => {
                text.push_str(&s);
            }
            Ok(XmlEvent::EndElement { name, .. }) => {
                if name.local_name == "t" {
                    in_text_element = false;
                }
            }
            _ => {}
        }
    }

    Ok(text.trim().to_string())
}