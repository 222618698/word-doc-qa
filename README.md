# Word Document Q&A System

A Question-Answering system built with Rust and the Burn deep learning framework that reads Word documents (.docx) and answers questions about their content.

## Overview

This project implements a complete ML pipeline:
1. Data loading and processing from .docx files
2. Transformer-based neural network training
3. Natural language question answering
4. Command-line interface for training and inference

## Built With

- [Rust](https://www.rust-lang.org/)
- [Burn Framework](https://burn.dev/) v0.20.1
- docx-rs for Word document parsing
- tokenizers for text tokenization
