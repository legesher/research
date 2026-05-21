# Evaluation Pipeline

Per-condition benchmark evaluation across 4 dataset languages (en, zh, es, ur) and 4 benchmarks, with prompt-template ablation and per-condition instruction-language matrices. Each fine-tuned condition has one or more registered training seeds; evals are run per `(condition, seed)`.

## Contents

- `scripts/preprocess.ipynb` — Loads + caches benchmark datasets and tokenized prompts (publish output as a Kaggle Dataset and attach it to evaluate.ipynb)
- `scripts/evaluate.ipynb` — Runs the eval suite against a configured `(condition, seed)` (dual-GPU, template-split)
- `scripts/rescore_xnli.py` — One-time XNLI re-scoring correction script (legacy)
- `requirements.txt` — Python dependencies

Results are stored on HuggingFace, not in this directory.

## Benchmark Suite

| Benchmark    | What It Measures                                                  | Dataset Languages |
| ------------ | ----------------------------------------------------------------- | ----------------- |
| **XNLI**     | Natural language inference (entailment / contradiction / neutral) | en, zh, es, ur    |
| **X-CSQA**   | Commonsense reasoning (5-way multiple choice)                     | en, zh, es, ur    |
| **SIB-200**  | Topic classification (7-way concrete categories)                  | en, zh, es, ur    |
| **Belebele** | Reading comprehension (4-way multiple choice)                     | en, zh, es, ur    |

## Per-Condition Eval Matrix

The preprocessing notebook caches all 4 dataset languages × 4 instruction-language prompts × 2 templates per row. The eval notebook then selects a subset per condition. Each fine-tune is identified by `(condition, seed)`; you run one Kaggle session per pair.

| Condition          | Seeds        | Dataset langs  | Instruction langs | Cells per (condition, seed) | Adapter on HF    |
| ------------------ | ------------ | -------------- | ----------------- | --------------------------- | ---------------- |
| baseline (no FT)   | —            | en, zh, es, ur | en, zh, es, ur    | **128**                     | n/a (base model) |
| condition-1-en-5k  | 42, 123, 456 | en, zh, es, ur | en, zh, es, ur    | **128**                     | ✅ all seeds     |
| condition-1-en-20k | 42           | en, zh, es, ur | en, zh, es, ur    | **128**                     | ✅               |
| condition-2-zh-5k  | 42, 123, 456 | en, zh, es, ur | en, zh            | **64**                      | ✅ all seeds     |
| condition-2-zh-20k | 42           | en, zh, es, ur | en, zh            | **64**                      | ✅               |
| condition-2-es-5k  | 42, 123, 456 | en, zh, es, ur | en, es            | **64**                      | ✅ all seeds     |
| condition-2-es-20k | 42           | en, zh, es, ur | en, es            | **64**                      | ✅               |
| condition-2-ur-5k  | 42, 123, 456 | en, zh, es, ur | en, ur            | **64**                      | ✅ all seeds     |
| condition-2-ur-20k | 42           | en, zh, es, ur | en, ur            | **64**                      | ✅               |
| condition-3-zh-5k  | 42           | en, zh, es, ur | en, zh            | **64**                      | ✅               |
| condition-5-zh-5k  | 42           | en, zh, es, ur | en, zh            | **64**                      | ✅               |
| condition-5-es-5k  | 42           | en, zh, es, ur | en, es            | **64**                      | ⏳ in progress   |
| condition-5-ur-5k  | 42           | en, zh, es, ur | en, ur            | **64**                      | ✅               |

Cells per `(condition, seed)` = 4 benchmarks × 2 templates × dataset-langs × instruction-langs.

**Rule for condition-2/3/5:** instruction language is always English OR the model's trained-on language, regardless of which dataset language we're evaluating against. A condition-2-ur model evaluated on `xnli_zh` gets either English or Urdu instructions, never Chinese — the model wasn't trained to follow Chinese instructions in that condition.

**Adapter locations.** All non-baseline adapters live in one HF repo, `legesher/language-decoded-lora`, organized by base model and condition+seed:

```text
legesher/language-decoded-lora/
└── tiny-aya-base/
    ├── condition-1-en-5k-seed{42,123,456}/
    ├── condition-1-en-20k-seed42/
    ├── condition-2-{zh,es,ur}-5k-seed{42,123,456}/
    ├── condition-2-{zh,es,ur}-20k-seed42/
    ├── condition-3-zh-5k-native-code-seed42/
    └── condition-5-{ur,zh}-5k-c4ai-aya-expanse-32b-seed42/
```

> `condition-5-es-5k-c4ai-aya-expanse-32b-seed42/` is pending upload — see the matrix above.

The eval script loads each adapter via `FastLanguageModel.from_pretrained(model_name=LORA_REPO, subfolder=<path>)`. For `baseline`, no subfolder; it loads `CohereLabs/tiny-aya-base` directly.

**Seeds and statistical reporting.** Conditions with multiple registered seeds (cond-1-en-5k, cond-2-{zh,es,ur}-5k) should be evaluated at every seed and reported as mean ± std. Single-seed conditions are point estimates. Aggregation happens offline from the per-`(condition, seed)` summary JSONs.

## Prompt Templates

The preprocessing notebook caches two prompt templates for ablation tests:

| Template    | Benchmarks                      | Notes                                               |
| ----------- | ------------------------------- | --------------------------------------------------- |
| `template1` | Belebele, SIB-200, X-CSQA, XNLI | Baseline wording                                    |
| `template2` | Belebele, SIB-200, X-CSQA, XNLI | Light rephrasing of template1 for sensitivity check |

Each template has English, Spanish, Urdu, and Chinese wording for the same benchmark structure. Template 2 is a slight natural rephrasing of Template 1, intended to test whether small prompt wording changes affect model performance.

Prompts are cached per-instruction-language as columns: `prompt_{template_id}_{instruction_lang}` plus matching `input_ids_*` / `attention_mask_*`. The eval notebook reads only the instruction-language subset configured for each condition.

## Kaggle Workflow

The pipeline is two-stage. **You must publish preprocess output as a Kaggle Dataset and attach it to evaluate.ipynb** — preprocess writes to `/kaggle/working/...` but evaluate reads from `/kaggle/input/...`, so skipping the publish step will land you a `FileNotFoundError`.

### Stage 1 — Preprocess (CPU)

1. Create a Kaggle Notebook, upload `scripts/preprocess.ipynb`, set accelerator to CPU (GPU is wasted here).
2. Add `HF_TOKEN` to Kaggle Secrets (the notebook reads it via `UserSecretsClient`).
3. Run all cells. The notebook downloads XNLI, X-CSQA, SIB-200, Belebele for en/zh/es/ur and tokenizes prompts (4 instruction langs × 2 templates × 4 benchmarks) against `CohereLabs/tiny-aya-base`.
4. Commit (Save Version). 16 JSONL files land in `/kaggle/working/eval_unsloth_artifacts/datasets/`.

### Stage 2 — Publish as a Kaggle Dataset

From the saved preprocess notebook version, use **Output → New Dataset** (or **File → Save As Dataset**). Pick a slug (e.g., `<your-username>/aya-eval-cache-phase3`). Note it down — you'll need it in Stage 3.

### Stage 3 — Evaluate (2× T4 GPU)

1. Create a Kaggle Notebook, upload `scripts/evaluate.ipynb`, set accelerator to **GPU T4 ×2**.
2. **Add Input** → attach the Kaggle Dataset you published in Stage 2. It mounts at `/kaggle/input/<your-slug>/...`.
3. **Override `KAGGLE_DATASET_INPUT_DIR`** in a cell at the top of the notebook (before the `%%writefile` cell):

   ```python
   import os
   os.environ["KAGGLE_DATASET_INPUT_DIR"] = (
       "/kaggle/input/<your-slug>/eval_unsloth_artifacts/datasets"
   )
   ```

   No silent default — the script will raise `RuntimeError` if this is unset, with a message pointing you here.

4. **Set `CONDITION` and `SEED`** in the picker cell. Use `SEED = "none"` for baseline, otherwise one of the registered seeds for that condition.

5. **Run the launcher cell.** Two subprocesses spawn — GPU 0 runs `template1`, GPU 1 runs `template2` — and a backgrounded `tail -F` streams both log files back to the cell output so you can watch progress live. The cell blocks until both subprocesses finish, then prints `=== Both templates done ===`.

   Output JSONs land in `/kaggle/working/`:
   - `{condition}_seed{seed}_summary_{template}.json` — per-cell accuracies + parse-failure rates
   - `{condition}_seed{seed}_results_{template}.json` — full per-row outputs (includes `raw_output` so you can re-parse offline if extractors change)
   - `{condition}_seed{seed}_partial_{template}.json` — incremental checkpoint (updates after each `(dataset_lang, instruction_lang)` block; survives mid-run crashes within Kaggle's 12h cap)

   > **Single-GPU fallback:** if you only have one T4 (or want simpler live output without the `tail` indirection), replace the launcher cell with `!python /kaggle/working/run_eval_single.py --condition {CONDITION} --seed {SEED} --batch_size 32`. The script loops over both templates serially in one foreground process. Roughly 2× wall-clock vs. the dual-GPU launcher.

6. **Commit** to save outputs, or upload them to `legesher/language-decoded-experiments` on HF.

7. **Repeat for each `(condition, seed)`.** With the seeds registered above, the full sweep is **21 Kaggle sessions** when every adapter is on HF:
   - 1 baseline
   - 3 cond-1-en-5k (seeds 42, 123, 456) + 1 cond-1-en-20k
   - 9 cond-2-{zh,es,ur}-5k (3 langs × 3 seeds) + 3 cond-2-{zh,es,ur}-20k
   - 1 cond-3-zh-5k
   - 3 cond-5-{zh,es,ur}-5k (single-seed each)

   Today, **20 of 21** can run — only `condition-5-es-5k` is still pending its adapter upload (see matrix above). Skip that one and you can land 20 sessions immediately; add the last when its adapter arrives.

> **Schema note:** if you have cached JSONLs from an earlier version of this pipeline (the PR #37-era `english`/`language` column suffixes), delete `eval_unsloth_artifacts/datasets/*.jsonl` and rerun preprocess before running evaluate. The current schema uses `prompt_{template_id}_{instruction_lang}` columns.

## Data Sources

| Benchmark | Source                                                                                                                      |
| --------- | --------------------------------------------------------------------------------------------------------------------------- |
| XNLI      | [facebook/xnli](https://huggingface.co/datasets/facebook/xnli) — config `en`/`zh`/`es`/`ur`                                 |
| X-CSQA    | [INK-USC/xcsr](https://huggingface.co/datasets/INK-USC/xcsr) — config `X-CSQA-{lang}`                                       |
| SIB-200   | [mteb/sib200](https://huggingface.co/datasets/mteb/sib200) — filter by FLORES lang code                                     |
| Belebele  | [facebook/belebele](https://huggingface.co/datasets/facebook/belebele) — config `eng_Latn`/`zho_Hans`/`spa_Latn`/`urd_Arab` |

## Results

All results are stored on HuggingFace:

| Repo                                                                                                  | Contents                                                                  |
| ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| [language-decoded-experiments](https://huggingface.co/datasets/legesher/language-decoded-experiments) | Per-`(condition, seed)` results (one summary + results JSON per template) |

Phase-2 results are archived under `phase2/` in that dataset. Note that phase-2 `english-forgetting/` filenames use abbreviated condition names (e.g., `cond-2-zh_english_results.json`); phase-3 outputs from this pipeline use full condition names (e.g., `condition-2-zh-5k_seed42_summary_template1.json`).

See [analysis/evaluation-summary.md](../analysis/evaluation-summary.md) for the full analysis.
