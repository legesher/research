# Model Cards

HuggingFace model cards for each fine-tuned adapter. Trained models live on HuggingFace — this directory holds the cards for version control and review.

## Models

| Directory | Condition | Description | HuggingFace Link |
| --- | --- | --- | --- |
| `condition-1-english/` | English code | QLoRA fine-tune on English Python | TBD |
| `condition-2-multilingual/` | Multilingual transpiled | QLoRA fine-tune on zh/am/ur transpiled code | TBD |
| `condition-3-nl-text/` | NL text control | QLoRA fine-tune on volume-matched NL text | TBD |
| `condition-4-native/` | Native code | QLoRA fine-tune on human-written native code | TBD |

## Base Model

- **Tiny Aya** (3.35B params) by Cohere
- QLoRA 4-bit quantization
- Training on Kaggle T4 GPUs (~5.4GB VRAM)