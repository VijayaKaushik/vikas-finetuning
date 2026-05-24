# Progress Log

## 2026-02-28 — Project Initialization

### What was done
- Created `CLAUDE.md` with project memory instructions and project description.
- Created `progress.md` to track implementation evolution.

### Logic / Reasoning
- The project is a new LLM fine-tuning repository. No code or structure exists yet.
- `progress.md` was established per CLAUDE.md instructions to log every implementation step with timestamps.

### Assumptions
- The fine-tuning framework, model, and task type have not yet been decided by the user.

### Context
- The repository is currently empty aside from `CLAUDE.md` and this file.
- Next step: user needs to specify fine-tuning framework/task so project structure can be scaffolded.

## 2026-02-28 — Project Structure Created (Gemma 3 + LoRA)

### What was done
- Created `requirements.txt` with all dependencies (torch, transformers, trl, peft, datasets, accelerate, bitsandbytes, python-docx, pyyaml, wandb).
- Created `config/config.yaml` — centralized config for model (Gemma 3 4B IT), LoRA hyperparameters (r=16, alpha=32), data paths, and training arguments.
- Created `src/prepare_data.py` — data pipeline: reads .docx via python-docx, extracts paragraph text, splits into overlapping word-based chunks, wraps each chunk in Gemma's chat template (`<start_of_turn>user/model`), saves as a HF Dataset to disk.
- Created `src/train.py` — training pipeline: loads Gemma 3 with 4-bit quantization (NF4 via BitsAndBytesConfig), applies LoRA to attention projections, trains with SFTTrainer (paged AdamW 8-bit optimizer, cosine LR schedule, bf16), saves final adapter.
- Created `src/__init__.py`, `.gitignore`, directory structure (`data/`, `outputs/`, `config/`, `src/`).
- Updated `CLAUDE.md` with architecture overview and commands.

### Logic / Data Flow
1. User places a `.docx` file at `data/input.docx`.
2. `prepare_data.py` extracts text → chunks by word count (default 512 words, 64 overlap) → formats each chunk into Gemma's chat turn template → saves as HF Dataset to `data/processed/`.
3. `train.py` loads the processed dataset → loads Gemma 3 with 4-bit quantization → applies LoRA (rank 16 on q/k/v/o projections) → trains via SFTTrainer → saves adapter weights to `outputs/gemma3-lora/final_adapter/`.

### Assumptions
- User has a CUDA-capable GPU with enough VRAM for 4-bit Gemma 3 4B (~6-8 GB).
- The instruction format uses a "learn and memorize" prompt; user should customize `format_as_instruction()` for their actual task (Q&A, summarization, etc.).
- Chunking is word-based (not token-based) for simplicity; chunk_size in config approximates word count, not exact token count.

### Context
- All hyperparameters are configurable via `config/config.yaml`.
- Next steps: user places their .docx, runs data prep, then training.

## 2026-03-01 — Data Extraction & Dataset Generation

### What was done
- Created `src/extract_equity_data.py` — extracts all paragraphs from docx preserving heading levels and styles, outputs `data/equity_raw.json` (252 paragraphs).
- Created `src/generate_dataset.py` — reads `equity_raw.json`, parses into 79 heading-grouped sections, generates 445 instruction-tuning Q&A pairs across 9 categories (definitions, comparisons, processes, compliance, Vijaya_equity workflows, lists, scenarios, best practices, summaries).
- Outputs saved as `data/equity_dataset.json` (plain JSON) and `data/equity_hf_dataset/` (HF Dataset format).

### Logic / Data Flow
1. `extract_equity_data.py` reads docx → iterates paragraphs → captures text, style name, heading level → writes structured JSON.
2. `generate_dataset.py` loads `equity_raw.json` → groups paragraphs into sections under their heading hierarchy (h1/h2/h3) → runs 9 generator functions that produce different question styles → deduplicates by instruction text → shuffles → saves both JSON and HF Dataset.
3. Generator categories:
   - **Definitions**: "What is X?" for each section with body text, plus contextual "Explain X in the context of Y" when nested under a parent heading.
   - **Comparisons**: Pairs sibling h3 sections and asks "What is the difference between A and B?"
   - **Processes**: Targets sections with process-related keywords (exercise, vesting, settlement, etc.) with "How does X work?" and "Describe the process of X."
   - **Compliance**: Targets tax/regulatory sections (IRC, ASC, blackout, etc.) with compliance-focused questions.
   - **Vijaya_equity**: Generates 3 question variants for each section under the Vijaya_equity company profile.
   - **Lists**: Asks "List the key aspects of X" when body has 3+ lines.
   - **Scenarios**: Hand-mapped practical scenario questions for key topics (stock splits, change in control, forfeiture, etc.).
   - **Best Practices**: Targets the "Best Practices and Considerations" h1 section.
   - **Summaries**: Aggregates h3 content under shared h2 headings for overview questions.

### Assumptions
- The instruction format `{instruction, input, output}` with empty `input` is standard Alpaca-style and compatible with SFTTrainer.
- Comparisons between sibling headings are meaningful (same parent topic).
- 445 pairs exceeds the 200-pair minimum target.

### Context
- `equity_raw.json`: 252 paragraphs, 79 sections with body text.
- `equity_dataset.json`: 445 unique Q&A pairs, shuffled with seed 42.
- Next step: update training pipeline to use `equity_dataset.json` or `equity_hf_dataset/` and run fine-tuning.

## 2026-03-01 — Hugging Face Authentication

### What was done
- Created `src/hf_login.py` — reads `HF_TOKEN` from environment variable, authenticates via `huggingface_hub.login()`.
- User ran the script and logged in successfully.

### Logic / Reasoning
- HF authentication is required to download gated models (like Gemma 3) and push artifacts to the Hub.
- Token is read from env var rather than hardcoded to avoid leaking secrets.

### Assumptions
- User has accepted the Gemma 3 model license on Hugging Face.

### Context
- Login is session-scoped; credentials are cached by `huggingface_hub` in `~/.cache/huggingface/`.
- Next step: run fine-tuning.

## 2026-03-01 — Training Pipeline Updated for Equity Dataset

### What was done
- Updated `config/config.yaml`: changed `data.output_dir` from `data/processed` to `data/equity_hf_dataset` so training loads the 445 Q&A pairs instead of the old chunked data.
- Updated `src/train.py`: added `format_sample()` function that converts Alpaca-style `{instruction, input, output}` records into Gemma 3 chat-template format (`<start_of_turn>user/model`). Passed as `formatting_func` to SFTTrainer.

### Logic / Reasoning
- SFTTrainer needs either a `text` column or a `formatting_func` to know how to convert dataset rows into training text. Since the equity dataset uses Alpaca-style fields, a formatting function is the correct approach.
- The chat template matches Gemma 3's expected format for instruction-tuned models.

### Assumptions
- The `input` field is empty for all 445 samples (verified from dataset), but the formatter handles non-empty `input` as well for robustness.

### Context
- Pipeline is now ready to run: `python3 src/train.py --config config/config.yaml`
- Next step: execute training on a CUDA-capable GPU.
