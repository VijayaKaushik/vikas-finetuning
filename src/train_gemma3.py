"""
Fine-tune Llama 3.2 3B-Instruct on the equity dataset using QLoRA + SFTTrainer.
"""

import os
import sys

import torch
from datasets import load_from_disk
from huggingface_hub import login
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


def format_sample(sample: dict) -> str:
    """Format an instruction/output pair into Llama 3 chat template."""
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{sample['instruction']}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{sample['output']}<|eot_id|>"
    )


def main():
    # ── HF Auth ────────────────────────────────────────────────────────
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN environment variable is not set.")
        print("Set it with: export HF_TOKEN='your_token_here'")
        sys.exit(1)
    login(token=hf_token)
    print("Logged in to Hugging Face.")

    # ── Config ──────────────────────────────────────────────────────────
    MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
    DATASET_PATH = "data/equity_hf_dataset"
    OUTPUT_DIR = "llama3-equity-finetuned"

    # ── 4-bit quantization (QLoRA) ──────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ── Load model & tokenizer ──────────────────────────────────────────
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model = prepare_model_for_kbit_training(model)

    # ── LoRA config ─────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── Dataset ─────────────────────────────────────────────────────────
    print(f"Loading dataset from {DATASET_PATH}")
    dataset = load_from_disk(DATASET_PATH)

    # Format each sample into Llama 3 chat template
    dataset = dataset.map(
        lambda sample: {"text": format_sample(sample)},
        remove_columns=dataset.column_names,
    )
    print(f"Training on {len(dataset)} samples.")

    # ── Training arguments ──────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_steps=50,
        save_total_limit=2,
        bf16=True,
        optim="paged_adamw_8bit",
        seed=42,
        report_to="none",
    )

    # ── Trainer ─────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        max_seq_length=1024,
    )

    # ── Train ───────────────────────────────────────────────────────────
    print("Starting training...")
    result = trainer.train()

    # Print final metrics
    print(f"\nTraining complete.")
    print(f"  Total steps: {result.global_step}")
    print(f"  Final loss:  {result.training_loss:.4f}")

    # ── Save adapter ────────────────────────────────────────────────────
    adapter_path = f"{OUTPUT_DIR}/final_adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"LoRA adapter saved to {adapter_path}/")


if __name__ == "__main__":
    main()
