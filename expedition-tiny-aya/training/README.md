# Training Pipeline

QLoRA fine-tuning on Kaggle T4 GPUs across experimental conditions.

## Contents

- `scripts/qlora.ipynb`: training notebook (pretokenization + DDP training)

## How It Works

The notebook writes two Python scripts to disk via `%%writefile`, then runs them:

1. **`pretokenize.py`**: reads the model settings from the config cell, downloads the dataset from Hugging Face, tokenizes it with that model's tokenizer and saves it to disk
2. **`train.py`**: reads the same settings, runs QLoRA training with Unsloth + SFTTrainer and uploads the adapter to Hugging Face

```bash
# Pretokenize run cell
!python pretokenize.py

# Training run cell (abridged): train each seed in TRAIN_SEEDS with DDP on 2 GPUs
%%bash
IFS=',' read -r -a SEEDS <<< "$TRAIN_SEEDS"
for seed in "${SEEDS[@]}"; do
  TRAIN_SEED="$seed" torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py
done
```

## Configuration

Set `MODEL_CONFIG_NAME`, `CONDITION_NAME`, `TRAIN_SEEDS`, `UPLOAD_TO_HF`, `OVERWRITE_HF_ADAPTER` and the `MODEL_CONFIG` dict in the notebook's config cell (the first cell after the install cell) before running; the script cells read them from the environment.

`MODEL_CONFIG_NAME` names the adapter's top-level folder on the LoRA repo. It does not load any settings: the model and hyperparameters come from the config cell's `MODEL_CONFIG` dict. Model config names in use:

- `tiny-aya-base`
- `qwen25-3b`
- `gemma4-e4b`
- `aya-expanse-8b`

Available conditions:

`CONDITION_NAME` must be a config of [`legesher/language-decoded-data`](https://huggingface.co/datasets/legesher/language-decoded-data); the dataset card lists them all. The Phase 3 paper conditions:

| `CONDITION_NAME`                                  | Training data                                                  |
| ------------------------------------------------- | -------------------------------------------------------------- |
| `condition-1-en-5k`, `condition-1-en-20k`         | English Python                                                 |
| `condition-2-{zh,es,ur}-5k`, `condition-2-{zh,es,ur}-20k` | The same files with Python's reserved words transpiled by Legesher |
| `condition-3-zh-5k`                               | Community-collected Chinese code                               |
| `condition-5-{zh,es,ur}-5k-c4ai-aya-expanse-32b`  | Reserved words transpiled, remaining content translated by `c4ai-aya-expanse-32b` |

## Model Configs

[`configs`](configs/) holds a YAML file per model with its model id, tokenizer length, LoRA target modules and training hyperparameters. The notebook does not read these files; they are the reference values to copy into the config cell's `MODEL_CONFIG` dict. For example:

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

To train another model, add its YAML file here, copy its values into `MODEL_CONFIG`, and set `MODEL_CONFIG_NAME` to the file's stem so the adapter lands in that model's folder.

## Data and Output Repos

| Repo                                                                                                  | Purpose                                     |
| ----------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| [language-decoded-data](https://huggingface.co/datasets/legesher/language-decoded-data)               | Training data (per-condition configs)       |
| [language-decoded-lora](https://huggingface.co/legesher/language-decoded-lora)                        | Trained adapters (per-model/per-condition-seed subfolders) |
| [language-decoded-experiments](https://huggingface.co/datasets/legesher/language-decoded-experiments) | Training config + eval results              |

## Seeded Training Outputs

- `TRAIN_SEEDS` in the config cell sets the seeds. Phase 3 uses `42`, `123` and `456` for the 5k condition-1 and condition-2 runs, and `42` and `123` for the rest; a new seed is evaluated only once it is registered in the evaluation notebook's `EVAL_MATRIX`
- Adapters are saved as `{condition}-seed{seed}`, for example `condition-2-zh-5k-seed42`
- Each seed run writes its own `training_metrics.json` alongside the adapter weights, including an `environment` block with the Python, CUDA and package versions it ran with (unsloth, transformers, trl, peft, torch and others)

## Upload to Hugging Face

With `UPLOAD_TO_HF = True` (the default), `train.py` uploads `adapter_*` and `training_metrics.json` to `legesher/language-decoded-lora` at `{MODEL_CONFIG_NAME}/{condition}-seed{seed}/`. The upload writes straight to the repo's `main` branch, not to a pull request, and needs a Kaggle secret named `HF_TOKEN` with write access to the `legesher` organization.

The folder name must match the evaluation notebook's `EVAL_MATRIX` subfolder template for the condition, or the evaluation cannot load the adapter; see the adapter naming rule in [`evaluation/README.md`](../evaluation/README.md).

Before training starts, `train.py` checks that `HF_TOKEN` is valid and that the target folder does not already exist on `main`. If it exists, the run stops before using any GPU time, so a rerun cannot silently replace an adapter behind published numbers. Set `OVERWRITE_HF_ADAPTER = True` only to replace it on purpose. A token without write access still fails at the upload, after training; the weights are then in `/kaggle/working/output/` until the session ends.

Set `UPLOAD_TO_HF = False` for a test run. The adapter then stays in the Kaggle session's `/kaggle/working/output/`, which is discarded when the session ends.

## Evaluation Expectations

- Aggregate results across all selected seeds
- Report `mean ± std` across runs
- Compute bootstrap 95% confidence intervals on individual test examples
- Use a paired bootstrap test for pairwise condition comparisons

## Compute

- Kaggle T4 x2 (DDP)
- ~5.4GB VRAM per GPU with QLoRA 4-bit
