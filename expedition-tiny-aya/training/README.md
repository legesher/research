# Training Pipeline

QLoRA fine-tuning of Tiny Aya (3.35B params) on Kaggle T4 GPUs across experimental conditions.

## Contents

- `scripts/qlora.ipynb` — Training notebook (pretokenization + DDP training)

## How It Works

The notebook writes two Python scripts to disk via `%%writefile`, then runs them:

1. **`pretokenize.py`** — Downloads dataset from HuggingFace, tokenizes with Tiny Aya tokenizer, saves to disk
2. **`train.py`** — Loads pretokenized data, runs QLoRA training with Unsloth + SFTTrainer, uploads adapter to HuggingFace

```bash
# Cell 3: pretokenize
!python pretokenize.py

# Cell 4: train with DDP on 2 GPUs
!torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py
```

## Configuration

Set `CONDITION_NAME` at the top of both cells before running. Available conditions:

| Condition            | Data Config                        | Description               |
| -------------------- | ---------------------------------- | ------------------------- |
| `condition-1-en-32k` | Full English Python (31,818 files) | Replicates original paper |
| `condition-1-en-5k`  | English Python 5K subset           | Controlled comparison     |
| `condition-2-zh-5k`  | Chinese keyword-swapped (5K)       | Legesher transpiled       |
| `condition-2-es-5k`  | Spanish keyword-swapped (5K)       | Legesher transpiled       |
| `condition-2-ur-5k`  | Urdu keyword-swapped (5K)          | Legesher transpiled       |
| `condition-3-zh-5k`  | Chinese mixed native (5K)          | Transpiled + native blend |

## QLoRA Hyperparameters

| Parameter             | Value            |
| --------------------- | ---------------- |
| LoRA r                | 16               |
| LoRA alpha            | 32               |
| LoRA dropout          | 0.0              |
| Learning rate         | 2e-4             |
| Batch size            | 8 per GPU        |
| Gradient accumulation | 1                |
| Epochs                | 1                |
| Optimizer             | paged_adamw_8bit |
| Scheduler             | cosine           |
| Max sequence length   | 1024             |
| Packing               | enabled          |

Full config documented at [`configs/qlora-base.json`](https://huggingface.co/datasets/legesher/language-decoded-experiments/blob/main/configs/qlora-base.json) on HuggingFace.

## Data and Output Repos

| Repo                                                                                                  | Purpose                                     |
| ----------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| [language-decoded-data](https://huggingface.co/datasets/legesher/language-decoded-data)               | Training data (per-condition configs)       |
| [language-decoded-lora](https://huggingface.co/legesher/language-decoded-lora)                        | Trained adapters (per-condition subfolders) |
| [language-decoded-experiments](https://huggingface.co/datasets/legesher/language-decoded-experiments) | Training config + eval results              |

## Compute

- Kaggle T4 x2 (DDP)
- ~5.4GB VRAM per GPU with QLoRA 4-bit
