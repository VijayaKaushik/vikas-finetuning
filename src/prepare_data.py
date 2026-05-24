"""
Data preparation script: reads a .docx file, chunks its text into training
samples formatted for instruction-tuning, and saves as a Hugging Face dataset.
"""

import argparse
import os
import re

import yaml
from datasets import Dataset
from docx import Document


def extract_text_from_docx(docx_path: str) -> str:
    """Extract all paragraph text from a .docx file."""
    doc = Document(docx_path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks by word count.

    Args:
        text: Full document text.
        chunk_size: Approximate number of words per chunk.
        overlap: Number of words to overlap between consecutive chunks.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def format_as_instruction(chunk: str) -> dict:
    """Wrap a text chunk into an instruction-tuning format.

    Uses a simple prompt asking the model to continue/summarise the content,
    with the chunk itself as the response. Adjust the instruction to match
    your actual task (Q&A, summarisation, chat, etc.).
    """
    return {
        "text": (
            "<start_of_turn>user\n"
            "Learn and memorize the following content:\n\n"
            f"{chunk}\n"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
            "I have learned the above content and can answer questions about it."
            "<end_of_turn>"
        )
    }


def prepare_dataset(config: dict) -> Dataset:
    """Full pipeline: docx -> chunks -> instruction dataset."""
    data_cfg = config["data"]
    docx_path = data_cfg["docx_path"]

    print(f"Reading {docx_path} ...")
    text = extract_text_from_docx(docx_path)
    print(f"Extracted {len(text.split())} words.")

    chunks = chunk_text(
        text,
        chunk_size=data_cfg["chunk_size"],
        overlap=data_cfg["overlap"],
    )
    print(f"Created {len(chunks)} chunks.")

    samples = [format_as_instruction(c) for c in chunks]
    dataset = Dataset.from_list(samples)

    output_dir = data_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)
    print(f"Dataset saved to {output_dir}/")

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare docx data for fine-tuning")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config file"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    prepare_dataset(config)


if __name__ == "__main__":
    main()
