# Natural Language Text Datasets

**Owner:** NL Curator (crew:nl-curator)

Pull, clean, and volume-match multilingual NL text for the control condition (Condition 3).

## Contents

- `scripts/` — Data pull, cleaning, tokenization, packaging
- `configs/` — Source URLs, target token counts per language

## Suggested Pipeline Steps

1. **Pull** NL text for zh/am/ur from CC-100 or OSCAR (e.g. `pull_nl_text.py`)
2. **Filter & clean** — language verification, quality filtering (e.g. `clean_and_filter.py`)
3. **Tokenize & volume-match** using Tiny Aya tokenizer (e.g. `tokenize_and_match.py`)
4. **Package** as HuggingFace Datasets (e.g. `package_nl_dataset.py`)

## Key Constraint

Token volume must match the transpiled code datasets exactly — this is the control condition.

## Deliverable

Volume-matched NL datasets delivered to Saad for Condition 3 training by Data Handoff (Day 5).