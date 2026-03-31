# Datasets

All datasets live on HuggingFace. This directory is no longer used for dataset cards.

## HuggingFace Repos

| Repo                                                                                                  | Description                       |
| ----------------------------------------------------------------------------------------------------- | --------------------------------- |
| [language-decoded-data](https://huggingface.co/datasets/legesher/language-decoded-data)               | Training data for all conditions  |
| [language-decoded-community](https://huggingface.co/datasets/legesher/language-decoded-community)     | Human-written native code samples |
| [language-decoded-experiments](https://huggingface.co/datasets/legesher/language-decoded-experiments) | Eval results and training configs |

## Data Configs (on `language-decoded-data`)

| Config               | Description                                   | Files  |
| -------------------- | --------------------------------------------- | ------ |
| `condition-1-en-32k` | English Python from The Stack                 | 31,818 |
| `condition-1-en-5k`  | English Python 5K subset                      | 5,000  |
| `condition-2-zh-5k`  | Chinese keyword-swapped (Legesher transpiled) | 5,000  |
| `condition-2-es-5k`  | Spanish keyword-swapped (Legesher transpiled) | 5,000  |
| `condition-2-ur-5k`  | Urdu keyword-swapped (Legesher transpiled)    | 5,000  |
| `condition-2-zh-32k` | Chinese keyword-swapped (full)                | 31,818 |
| `condition-2-es-32k` | Spanish keyword-swapped (full)                | 31,818 |
| `condition-2-ur-32k` | Urdu keyword-swapped (full)                   | 31,818 |
| `condition-3-zh-5k`  | Chinese mixed native blend                    | 5,000  |
| `condition-4-zh-5k`  | Chinese strictly native code                  | varies |
