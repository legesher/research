# Transpilation Pipeline

Batch transpilation of Python source files from English into Chinese (zh), Spanish (es), and Urdu (ur) using Legesher.

## Contents

- `scripts/` — Batch transpilation wrapper, stress test
- `results/` — Stress test reports and batch run logs

## Dependencies

- Legesher CLI (`legesher translate`)
- Filtered Python files from `../data-pipeline/`
- Language packs: `legesher-i18n-python-zh`, `legesher-i18n-python-es`, `legesher-i18n-python-ur`

## Usage

```bash
# Run batch transpilation
python scripts/batch_transpile.py --config configs/transpile_config.yaml

# Run stress test for a language
python scripts/stress_test_transpiler.py --lang zh --num-files 1000
```

## Output

Transpiled files are packaged and uploaded to [language-decoded-data](https://huggingface.co/datasets/legesher/language-decoded-data) as condition-2 configs.
