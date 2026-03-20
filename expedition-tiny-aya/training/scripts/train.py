#!/usr/bin/env python3
"""QLoRA fine-tuning script for Language Decoded experiments.

Loads the shared config (qlora-base.yaml), applies QLoRA to Tiny Aya base,
and trains on the specified dataset condition.

Requirements:
    pip install transformers peft bitsandbytes datasets accelerate trl pyyaml

Usage:
    # Train with default config (condition-1-en)
    python train.py --config ../configs/qlora-base.yaml

    # Override the dataset condition
    python train.py --config ../configs/qlora-base.yaml --condition condition-2-zh

    # Dry run — load everything but don't train (validates setup)
    python train.py --config ../configs/qlora-base.yaml --dry-run

    # Quick smoke test (100 examples, 10 steps)
    python train.py --config ../configs/qlora-base.yaml --smoke-test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for Language Decoded"
    )
    parser.add_argument("--config", required=True, help="Path to qlora-base.yaml")
    parser.add_argument(
        "--condition",
        default=None,
        help="Override dataset config (e.g., condition-2-zh)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Load model + data but don't train"
    )
    parser.add_argument(
        "--smoke-test", action="store_true", help="Quick test: 100 examples, 10 steps"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Allow condition override from CLI
    if args.condition:
        cfg["dataset"]["config"] = args.condition

    condition = cfg["dataset"]["config"]
    print(f"{'=' * 60}")
    print("Language Decoded — QLoRA Training")
    print(f"Condition: {condition}")
    print(f"{'=' * 60}")

    # -------------------------------------------------------------------------
    # 1. Load tokenizer
    # -------------------------------------------------------------------------
    print(f"\n[1/5] Loading tokenizer: {cfg['model']['tokenizer']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["tokenizer"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Required for causal LM training

    # -------------------------------------------------------------------------
    # 2. Load quantized model
    # -------------------------------------------------------------------------
    print(f"[2/5] Loading model: {cfg['model']['name']} (4-bit quantized)")
    quant_cfg = cfg["quantization"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model)

    # Print model size info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # -------------------------------------------------------------------------
    # 3. Apply LoRA
    # -------------------------------------------------------------------------
    print(f"[3/5] Applying LoRA (r={cfg['lora']['r']}, alpha={cfg['lora']['alpha']})")
    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )

    model = get_peft_model(model, peft_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable_params / all_params
    print(f"  Trainable: {trainable_params:,} / {all_params:,} ({pct:.2f}%)")

    # -------------------------------------------------------------------------
    # 4. Load dataset
    # -------------------------------------------------------------------------
    ds_cfg = cfg["dataset"]
    print(f"[4/5] Loading dataset: {ds_cfg['name']} ({ds_cfg['config']})")
    dataset = load_dataset(ds_cfg["name"], ds_cfg["config"])

    train_dataset = dataset[ds_cfg["train_split"]]
    eval_dataset = dataset[ds_cfg["eval_split"]]

    if args.smoke_test:
        train_dataset = train_dataset.select(range(min(100, len(train_dataset))))
        eval_dataset = eval_dataset.select(range(min(20, len(eval_dataset))))
        print(f"  Smoke test: {len(train_dataset)} train, {len(eval_dataset)} eval")

    print(f"  Train: {len(train_dataset):,} examples")
    print(f"  Eval:  {len(eval_dataset):,} examples")

    # -------------------------------------------------------------------------
    # 5. Configure training
    # -------------------------------------------------------------------------
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    out_cfg = cfg["output"]

    output_dir = out_cfg["dir"].replace("${dataset.config}", condition)
    run_name = cfg["tracking"]["run_name"].replace("${dataset.config}", condition)

    # Compute total steps for reporting
    effective_batch = (
        train_cfg["per_device_train_batch_size"]
        * train_cfg["gradient_accumulation_steps"]
    )
    total_steps = (len(train_dataset) // effective_batch) * train_cfg["num_epochs"]
    warmup_steps = int(total_steps * train_cfg["warmup_ratio"])
    print("\n[5/5] Training setup:")
    print(f"  Output:           {output_dir}")
    print(f"  Effective batch:  {effective_batch}")
    print(f"  Total steps:      {total_steps:,}")
    print(f"  Warmup steps:     {warmup_steps}")
    print(f"  Max seq length:   {data_cfg['max_seq_length']}")
    print(f"  Packing:          {data_cfg['packing']}")

    if args.dry_run:
        print("\n--- DRY RUN: Setup validated successfully. Exiting. ---")
        # Save config snapshot
        snapshot = {
            "condition": condition,
            "model": cfg["model"]["name"],
            "lora_r": lora_cfg["r"],
            "lora_alpha": lora_cfg["alpha"],
            "target_modules": lora_cfg["target_modules"],
            "trainable_params": trainable_params,
            "trainable_pct": round(pct, 2),
            "train_examples": len(train_dataset),
            "eval_examples": len(eval_dataset),
            "effective_batch_size": effective_batch,
            "total_steps": total_steps,
            "max_seq_length": data_cfg["max_seq_length"],
        }
        print(json.dumps(snapshot, indent=2))
        return

    max_steps = 10 if args.smoke_test else -1  # -1 = use num_epochs

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg["num_epochs"],
        max_steps=max_steps,
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        fp16=train_cfg["fp16"],
        bf16=train_cfg["bf16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        optim=train_cfg["optim"],
        seed=train_cfg["seed"],
        save_strategy=out_cfg["save_strategy"],
        logging_steps=out_cfg["logging_steps"],
        eval_strategy=out_cfg["eval_strategy"],
        eval_steps=out_cfg["eval_steps"],
        report_to=cfg["tracking"]["report_to"],
        run_name=run_name,
        push_to_hub=out_cfg["push_to_hub"] and not args.smoke_test,
        hub_model_id=out_cfg.get("hub_model_id"),
    )

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Starting training...")
    print(f"{'=' * 60}\n")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        dataset_text_field=data_cfg["dataset_text_field"],
        max_seq_length=data_cfg["max_seq_length"],
        packing=data_cfg["packing"],
    )

    trainer.train()

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    print(f"\nSaving adapter to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training metrics
    metrics = trainer.state.log_history
    metrics_path = Path(output_dir) / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    if out_cfg["push_to_hub"] and not args.smoke_test:
        print(f"\nPushing to {out_cfg['hub_model_id']}/{condition}")
        trainer.push_to_hub()

    print(f"\n{'=' * 60}")
    print(f"Training complete: {condition}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
