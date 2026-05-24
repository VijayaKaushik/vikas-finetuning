# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
## Project Memory Instructions(must be followed throughout the project extremely strictly)
  1. progress.md
  This document captures the evolution of the implementation. Each update must include:
  Detailed description of newly written or modified code
  Logic applied, including data flow and reasoning steps
  Assumptions made during implementation
  Any relevant context required for future contributors
  Every implementation step requires a timestamped entry.
## Project

Fine-tuning Gemma 3 on custom .docx documents using Hugging Face Transformers, TRL (SFTTrainer), and PEFT (LoRA/QLoRA).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Step 1: Prepare data (place your .docx in data/input.docx first)
python3 src/prepare_data.py --config config/config.yaml

# Step 2: Train
python3 src/train.py --config config/config.yaml
```

## Architecture

- `config/config.yaml` — single config driving model, LoRA, data, and training settings
- `src/prepare_data.py` — reads .docx, chunks text, formats as instruction-tuning samples, saves HF dataset to `data/processed/`
- `src/train.py` — loads quantized Gemma 3, applies LoRA, trains with SFTTrainer, saves adapter to `outputs/`
- `data/` — input .docx files and processed datasets
- `outputs/` — training checkpoints and final LoRA adapter
