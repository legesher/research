# Data Pipeline

**Owner:** Pipeline Engineer (crew:pipeline-eng)

Pull, filter, and package Python source code from The Stack for transpilation.

## Contents

- `scripts/` — Streaming, filtering, batch processing, HF packaging
- `configs/` — Filter thresholds, parallelism settings
- `logs/` — Pipeline run logs

## Pipeline Steps

1. **Stream** Python files from The Stack v1 via HuggingFace (`stream_the_stack.py`)
2. **Filter** by AST validity, deduplication, length, and license (`filter_pipeline.py`)
3. **Batch process** with parallelization for scale (`batch_processor.py`)
4. **Package** as HuggingFace Datasets with metadata (`package_dataset.py`)

## Dependency Chain

```
stream_the_stack.py → filter_pipeline.py → delivers to transpilation/
                                          → package_dataset.py → HuggingFace
```

## Quick Start

```bash
# Pull and filter
python scripts/stream_the_stack.py --config configs/pipeline_config.yaml
python scripts/filter_pipeline.py --input raw/ --output filtered/

# Package for HuggingFace
python scripts/package_dataset.py --input filtered/ --name legesher/the-stack-python-filtered
```
