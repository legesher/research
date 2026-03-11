# Transpilation Pipeline

**Owner:** Madi (crew:madi)

Batch transpilation of Python source files from English into Chinese (zh), Amharic (am), and Urdu (ur) using Legesher.

## Contents

- `scripts/` — Batch transpilation wrapper, stress test, HuggingFace upload
- `configs/` — Language targets, file counts, path configuration
- `results/` — Stress test reports and batch run logs

## Dependencies

- Legesher CLI (`legesher translate`)
- Filtered Python files from `../data-pipeline/`
- Language packs: `legesher-i18n-python-zh`, `legesher-i18n-python-am`, `legesher-i18n-python-ur`

## Quick Start

```bash
# Run batch transpilation (after filtered files are ready)
python scripts/batch_transpile.py --config configs/transpile_config.yaml

# Upload to HuggingFace
python scripts/upload_to_hf.py --lang zh --dataset-name legesher/transpiled-python-zh
```