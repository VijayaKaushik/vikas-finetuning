"""
Training script: loads Gemma 3, applies QLoRA via PEFT, and fine-tunes
using SFTTrainer from trl on the prepared dataset.
"""

import argparse
import os

import torch
import yaml
from datasets import load_from_disk
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
)
from trl import SFTConfig, SFTTrainer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_quantization_config(quant: str) -> BitsAndBytesConfig | None:
    if quant == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif quant == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def load_model_and_tokenizer(config: dict):
    model_name = config["model"]["name"]
    quant = config["model"]["quantization"]

    print(f"Loading model: {model_name} (quantization: {quant})")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = get_quantization_config(quant)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )

    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    return model, tokenizer


def get_lora_config(config: dict) -> LoraConfig:
    lora_cfg = config["lora"]
    return LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def tokenize_dataset(dataset, tokenizer, max_length: int):
    """Pre-tokenize dataset using apply_chat_template to produce token_type_ids."""

    def tokenize_fn(sample):
        instruction = sample["instruction"]
        context = sample.get("input", "")
        response = sample["output"]

        if context:
            user_msg = f"{instruction}\n\n{context}"
        else:
            user_msg = instruction

        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": response},
        ]

        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors=None,
            max_length=max_length,
            truncation=True,
        )

        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized = dataset.map(
        tokenize_fn,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    return tokenized


def train(config: dict):
    train_cfg = config["training"]
    data_cfg = config["data"]

    # Wandb setup
    if config.get("wandb", {}).get("enabled", False):
        os.environ["WANDB_PROJECT"] = config["wandb"]["project"]
    else:
        os.environ["WANDB_DISABLED"] = "true"

    # Load dataset
    dataset = load_from_disk(data_cfg["output_dir"])
    print(f"Loaded {len(dataset)} training samples.")

    # Load model
    model, tokenizer = load_model_and_tokenizer(config)
    peft_config = get_lora_config(config)

    # Pre-tokenize with chat template (produces token_type_ids for Gemma 3)
    max_length = train_cfg["max_seq_length"]
    dataset = tokenize_dataset(dataset, tokenizer, max_length)
    print(f"Tokenized dataset columns: {dataset.column_names}")

    # Training arguments
    training_args = SFTConfig(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=2,
        bf16=train_cfg["bf16"],
        optim="paged_adamw_8bit",
        seed=train_cfg["seed"],
        max_length=max_length,
        report_to="wandb" if config.get("wandb", {}).get("enabled") else "none",
        dataset_text_field=None,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Starting training...")
    trainer.train()

    # Save adapter
    save_path = os.path.join(train_cfg["output_dir"], "final_adapter")
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Adapter saved to {save_path}/")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 3 with LoRA")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config file"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
