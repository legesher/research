# Training Pipeline

**Owner:** Saad (crew:saad)

QLoRA fine-tuning of Tiny Aya (3.35B params) on T4 GPUs across 4 experimental conditions.

## Contents

- `configs/` — Per-condition QLoRA configuration files
- `scripts/` — Training entrypoint, adapter merge, orchestration
- `logs/` — Training logs and W&B dashboard links

## Conditions

| Suggested Config | Condition | Data Source |
| --- | --- | --- |
| `condition_1_en.yaml` | English code | The Stack Python subset |
| `condition_2_multi.yaml` | Multilingual transpiled code | Legesher-transpiled zh/am/ur |
| `condition_3_nl.yaml` | NL text control | CC-100/OSCAR volume-matched |
| `condition_4_native.yaml` | Native code | Human-written native code |

## Suggested Entrypoints

These scripts are suggested starting points — adapt as needed:

```bash
# Train a single condition
python scripts/train.py --config configs/condition_1_en.yaml

# Run all conditions
bash scripts/run_all_conditions.sh
```

## Compute

- Kaggle T4 GPUs (team-pooled, 300-420 hours)
- QLoRA 4-bit: ~5.4GB VRAM per run