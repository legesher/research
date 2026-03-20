# Training Pipeline

**Owner:** Saad (crew:saad)

QLoRA fine-tuning of [Tiny Aya base](https://huggingface.co/CohereLabs/tiny-aya-base) (3.35B params) on Kaggle T4 GPUs across experimental conditions.

## Contents

- `configs/qlora-base.yaml` — Shared QLoRA config (identical across all conditions)
- `scripts/train.py` — Training entrypoint with dry-run and smoke-test modes
- `logs/` — Training logs

## Quick Start

```bash
# Install dependencies (on Kaggle or local)
pip install transformers peft bitsandbytes datasets accelerate trl pyyaml

# Dry run — validate setup without training (no GPU needed for config check)
python scripts/train.py --config configs/qlora-base.yaml --dry-run

# Smoke test — 100 examples, 10 steps (validates full pipeline on GPU)
python scripts/train.py --config configs/qlora-base.yaml --smoke-test

# Train Condition 1 (English code)
python scripts/train.py --config configs/qlora-base.yaml --condition condition-1-en

# Train Condition 2 (keyword-swapped, per language)
python scripts/train.py --config configs/qlora-base.yaml --condition condition-2-zh
python scripts/train.py --config configs/qlora-base.yaml --condition condition-2-es
python scripts/train.py --config configs/qlora-base.yaml --condition condition-2-ur
```

## Conditions

| Condition      | Data Source                         | Dataset Config         |
| -------------- | ----------------------------------- | ---------------------- |
| Condition 1    | English Python from The Stack Dedup | `condition-1-en`       |
| Condition 2-zh | Chinese keyword-swapped Python      | `condition-2-zh`       |
| Condition 2-es | Spanish keyword-swapped Python      | `condition-2-es`       |
| Condition 2-ur | Urdu keyword-swapped Python         | `condition-2-ur`       |
| Condition 3-zh | Transpiled + native Chinese code    | `condition-3-zh`       |
| Condition 3-es | Transpiled + native Spanish code    | `condition-3-es`       |
| Condition 3-ur | Transpiled + native Urdu code       | `condition-3-ur`       |
| Condition 4    | All strictly native code (combined) | `condition-4-combined` |

All data is loaded from [`legesher/language-decoded-data`](https://huggingface.co/datasets/legesher/language-decoded-data) via named configs. The only thing that changes between runs is `--condition`.

## Config Summary

| Parameter            | Value                               | Rationale                                     |
| -------------------- | ----------------------------------- | --------------------------------------------- |
| LoRA rank            | 16                                  | Sweet spot for 3B model on T4                 |
| LoRA alpha           | 32                                  | 2x rank (standard scaling)                    |
| Target modules       | All linear (q/k/v/o + gate/up/down) | Best quality per QLoRA research               |
| Learning rate        | 5e-5                                | Conservative for QLoRA continued pre-training |
| Epochs               | 1                                   | Standard for code pre-training                |
| Effective batch size | 16 (2 × 8 grad accum)               | Fits T4 VRAM                                  |
| Max seq length       | 1024 tokens                         | Balances context vs VRAM                      |
| Packing              | Yes                                 | Concatenates short examples for efficiency    |
| Quantization         | NF4 + double quant                  | ~5.4GB VRAM                                   |
| Optimizer            | paged_adamw_8bit                    | Saves ~1GB VRAM                               |

## Compute

- Kaggle T4 GPUs (16GB VRAM)
- ~5.4GB VRAM per run with QLoRA
- Estimated ~30-60 min per condition (depending on dataset size)

## Output

Adapters are saved to [`legesher/language-decoded-lora`](https://huggingface.co/legesher/language-decoded-lora) organized by condition subfolder.
