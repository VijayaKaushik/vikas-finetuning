"""
Extracts all paragraphs from a .docx file, preserving heading levels,
and outputs as a structured JSON file.
"""

import json
import argparse
from docx import Document


def extract_paragraphs(docx_path: str) -> list[dict]:
    """Extract paragraphs with their style (heading level or body text)."""
    doc = Document(docx_path)
    paragraphs = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        style = para.style.name if para.style else "Normal"
        if style.startswith("Heading"):
            try:
                level = int(style.split()[-1])
            except ValueError:
                level = None
            para_type = "heading"
        else:
            level = None
            para_type = "body"

        paragraphs.append({
            "type": para_type,
            "heading_level": level,
            "style": style,
            "text": text,
        })

    return paragraphs


def main():
    parser = argparse.ArgumentParser(description="Extract docx paragraphs to JSON")
    parser.add_argument(
        "--input", type=str,
        default="data/Equity_Plan_Management_Guide_Updated.docx",
        help="Path to .docx file",
    )
    parser.add_argument(
        "--output", type=str,
        default="data/equity_raw.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    paragraphs = extract_paragraphs(args.input)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(paragraphs, f, indent=2, ensure_ascii=False)

    print(f"Extracted {len(paragraphs)} paragraphs to {args.output}")


if __name__ == "__main__":
    main()
