# Training Pipeline

QLoRA fine-tuning on Kaggle T4 GPUs across experimental conditions, with model-specific settings loaded from YAML configs.

## Contents

- `scripts/qlora.ipynb` — Training notebook (pretokenization + DDP training)

## How It Works

The notebook writes two Python scripts to disk via `%%writefile`, then runs them:

1. **`pretokenize.py`** — Loads a model config, downloads dataset from HuggingFace, tokenizes with that model's tokenizer, saves to disk
2. **`train.py`** — Loads the same model config, runs QLoRA training with Unsloth + SFTTrainer, uploads the adapter to HuggingFace

```bash
# Cell 3: pretokenize
!python pretokenize.py

# Cell 4: train with DDP on 2 GPUs
!torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py
```

## Configuration

Set `MODEL_CONFIG_NAME` and `CONDITION_NAME` at the top of both notebook script cells before running.

Available model configs:

- `tiny-aya-base`
- `qwen25-3b`
- `gemma4-e4b`
- `aya-expanse-8b`

Available conditions:

| Condition            | Data Config                        | Description               |
| -------------------- | ---------------------------------- | ------------------------- |
| `condition-1-en-32k` | Full English Python (31,818 files) | Replicates original paper |
| `condition-1-en-5k`  | English Python 5K subset           | Controlled comparison     |
| `condition-2-zh-5k`  | Chinese keyword-swapped (5K)       | Legesher transpiled       |
| `condition-2-es-5k`  | Spanish keyword-swapped (5K)       | Legesher transpiled       |
| `condition-2-ur-5k`  | Urdu keyword-swapped (5K)          | Legesher transpiled       |
| `condition-3-zh-5k`  | Chinese mixed native (5K)          | Transpiled + native blend |

## Model Configs

Model configs live in [`configs`](configs/) as YAML files. They define model selection, tokenizer length, LoRA target modules, and training hyperparameters, for example:

```yaml
model_id: CohereLabs/tiny-aya-base
max_seq_length: 1024
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
lora_r: 16
lora_alpha: 32
lora_dropout: 0.0
load_in_4bit: true
learning_rate: 2.0e-4
per_device_train_batch_size: 8
num_train_epochs: 1
warmup_ratio: 0.05
max_grad_norm: 1.0
use_unsloth: true
```

Adding a new model now only requires adding another YAML file and setting `MODEL_CONFIG_NAME` to its filename stem.

## Data and Output Repos

| Repo                                                                                                  | Purpose                                     |
| ----------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| [language-decoded-data](https://huggingface.co/datasets/legesher/language-decoded-data)               | Training data (per-condition configs)       |
| [language-decoded-lora](https://huggingface.co/legesher/language-decoded-lora)                        | Trained adapters (per-model/per-condition subfolders) |
| [language-decoded-experiments](https://huggingface.co/datasets/legesher/language-decoded-experiments) | Training config + eval results              |

## Compute

- Kaggle T4 x2 (DDP)
- ~5.4GB VRAM per GPU with QLoRA 4-bit
