# Transpilation Pipeline

Stress-testing and validation of Legesher's transpilation of Python source files from English into Chinese (zh), Spanish (es), and Urdu (ur). The batch transpilation runner itself lives in [`../data-pipeline/scripts/batch_transpile.py`](../data-pipeline/scripts/batch_transpile.py); this directory holds the transpiler stress-test harness and its reports.

## Contents

- `scripts/stress_test_transpiler.py` — Transpiler stress-test harness
- `results/` — Stress-test reports (`reports/*.json`) and the [findings writeup](results/STRESS_TEST_FINDINGS.md)

## Dependencies

- Legesher CLI (`legesher translate`)
- Filtered Python files from `../data-pipeline/`
- Language packs: `legesher-i18n-python-zh`, `legesher-i18n-python-es`, `legesher-i18n-python-ur`

## Usage

```bash
# Run the transpiler stress test for a language
python scripts/stress_test_transpiler.py --lang zh --num-files 1000
```

## Output

Transpiled files are packaged and uploaded to [language-decoded-data](https://huggingface.co/datasets/legesher/language-decoded-data) as condition-2 configs.
