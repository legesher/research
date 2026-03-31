# Models

All trained adapters live on HuggingFace in the unified lora repo. This directory is no longer used for model cards.

## Adapter Repo

[legesher/language-decoded-lora](https://huggingface.co/legesher/language-decoded-lora) — one subfolder per condition.

| Subfolder             | Condition            | Description                                    |
| --------------------- | -------------------- | ---------------------------------------------- |
| `condition-1-en-32k/` | English code (full)  | QLoRA fine-tune on 31,818 English Python files |
| `condition-1-en-5k/`  | English code (5K)    | QLoRA fine-tune on 5K English subset           |
| `condition-2-zh-5k/`  | Chinese transpiled   | QLoRA fine-tune on zh keyword-swapped code     |
| `condition-2-es-5k/`  | Spanish transpiled   | QLoRA fine-tune on es keyword-swapped code     |
| `condition-2-ur-5k/`  | Urdu transpiled      | QLoRA fine-tune on ur keyword-swapped code     |
| `condition-3-zh-5k/`  | Chinese mixed native | QLoRA fine-tune on transpiled + native blend   |

## Base Model

- [CohereLabs/tiny-aya-base](https://huggingface.co/CohereLabs/tiny-aya-base) (3.35B params)
- QLoRA 4-bit quantization via Unsloth
- Training on Kaggle T4 x2 (DDP)
