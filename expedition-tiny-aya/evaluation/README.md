# Evaluation Pipeline

Per-condition benchmark evaluation across 4 dataset languages (en, zh, es, ur) and 4 benchmarks, with prompt-template ablation and per-condition instruction-language matrices. Each fine-tuned condition has one or more registered training seeds; evals are run per `(condition, seed)`.

## Contents

- `scripts/preprocess.ipynb` — Loads + caches benchmark datasets and tokenized prompts (publish output as a Kaggle Dataset and attach it to evaluate.ipynb)
- `scripts/evaluate.ipynb` — Runs the eval suite against a configured `(condition, seed)` (dual-GPU, template-split)
- `scripts/rescore_xnli.py` — One-time XNLI re-scoring correction script (legacy)
- `requirements.txt` — Python dependencies
- `scripts/test_*.py` — Stdlib `unittest` suite for the analysis scripts. Run from the repository root with `python -m unittest discover -s expedition-tiny-aya/evaluation/scripts -p 'test_*.py' -v` (`huggingface_hub` must be installed); CI runs the same command in `.github/workflows/tests.yml`

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
| condition-1-en-20k | 42, 123      | en, zh, es, ur | en, zh, es, ur    | **128**                     | ✅ all seeds     |
| condition-2-zh-5k  | 42, 123, 456 | en, zh, es, ur | en, zh            | **64**                      | ✅ all seeds     |
| condition-2-zh-20k | 42, 123      | en, zh, es, ur | en, zh            | **64**                      | ✅ all seeds     |
| condition-2-es-5k  | 42, 123, 456 | en, zh, es, ur | en, es            | **64**                      | ✅ all seeds     |
| condition-2-es-20k | 42, 123      | en, zh, es, ur | en, es            | **64**                      | ✅ all seeds     |
| condition-2-ur-5k  | 42, 123, 456 | en, zh, es, ur | en, ur            | **64**                      | ✅ all seeds     |
| condition-2-ur-20k | 42, 123      | en, zh, es, ur | en, ur            | **64**                      | ✅ all seeds     |
| condition-3-zh-5k  | 42, 123      | en, zh, es, ur | en, zh            | **64**                      | ✅ all seeds     |
| condition-5-zh-5k  | 42, 123      | en, zh, es, ur | en, zh            | **64**                      | ✅ all seeds     |
| condition-5-es-5k  | 42, 123      | en, zh, es, ur | en, es            | **64**                      | ✅ all seeds     |
| condition-5-ur-5k  | 42, 123      | en, zh, es, ur | en, ur            | **64**                      | ✅ all seeds     |

Cells per `(condition, seed)` = 4 benchmarks × 2 templates × dataset-langs × instruction-langs.

The table lists the seeds whose evaluations are published, and its adapters are all on Hugging Face. `EVAL_MATRIX` in `evaluate.ipynb` can register a seed before its sessions have run, so it may list more seeds than this table; a registered seed whose adapter is not yet on Hugging Face fails to load in the eval notebook. [Running a new seed](#running-a-new-seed) shows how to find and run the sessions that are not in the table yet.

**Rule for condition-2/3/5:** instruction language is always English OR the model's trained-on language, regardless of which dataset language we're evaluating against. A condition-2-ur model evaluated on `xnli_zh` gets either English or Urdu instructions, never Chinese — the model wasn't trained to follow Chinese instructions in that condition.

**Adapter locations.** All non-baseline adapters live in one HF repo, `legesher/language-decoded-lora`, organized by base model and condition+seed:

```text
legesher/language-decoded-lora/
└── tiny-aya-base/
    ├── condition-1-en-5k-seed{42,123,456}/
    ├── condition-1-en-20k-seed{42,123}/
    ├── condition-2-{zh,es,ur}-5k-seed{42,123,456}/
    ├── condition-2-{zh,es,ur}-20k-seed{42,123}/
    ├── condition-3-zh-5k-seed{42,123}/
    └── condition-5-{es,zh,ur}-5k-c4ai-aya-expanse-32b-seed{42,123}/
```

**Naming rule.** An adapter folder is `{data config}-seed{seed}`, where `{data config}` is the `legesher/language-decoded-data` config the adapter was trained on. That is the name the training notebook uploads under (`{CONDITION_NAME}-seed{seed}`), so the eval template finds a new seed without a rename. The data config is the condition key itself, except for condition 5, whose configs carry the generator model (`-c4ai-aya-expanse-32b`). Folders under `legesher/language-decoded-experiments` use the condition key alone (`phase3/conditions/condition-5-zh-5k/`). The cond-3 folders were named `condition-3-zh-5k-native-code-seed{42,123}` until October 2026; the LoRA repo's `MANIFEST.md` records the rename. `test_refresh_seed_results.py` checks every registered template against this rule.

The eval script loads each adapter via `FastLanguageModel.from_pretrained(model_name=LORA_REPO, subfolder=<path>)`. For `baseline`, no subfolder; it loads `CohereLabs/tiny-aya-base` directly.

**Seeds and statistical reporting.** Conditions with multiple registered seeds (cond-1-en-5k and cond-2-{zh,es,ur}-5k at three seeds; cond-1-en-20k, cond-2-{zh,es,ur}-20k, cond-3-zh-5k and cond-5-{zh,es,ur}-5k at two) should be evaluated at every seed and reported as mean ± std. Single-seed conditions are point estimates. The seed-123 sessions of the 20k, cond-3 and cond-5 conditions were all published by 2026-10-18 and are labelled provisional: they were evaluated in a later software environment than the seed-42 sessions, and `analysis/phase-3/cross-seed-variance-by-condition.md` marks them until the drift check between the two environments is resolved. `scripts/refresh_seed_results.py status` lists the current state per session. Aggregation happens offline from the per-`(condition, seed)` summary JSONs.

## Prompt Templates

The preprocessing notebook caches two prompt templates for ablation tests:

| Template    | Benchmarks                      | Notes                                               |
| ----------- | ------------------------------- | --------------------------------------------------- |
| `template1` | Belebele, SIB-200, X-CSQA, XNLI | Baseline wording                                    |
| `template2` | Belebele, SIB-200, X-CSQA, XNLI | Light rephrasing of template1 for sensitivity check |

Each template has English, Spanish, Urdu, and Chinese wording for the same benchmark structure. Template 2 is a slight natural rephrasing of Template 1, intended to test whether small prompt wording changes affect model performance.

Prompts are cached per-instruction-language as columns: `prompt_{template_id}_{instruction_lang}` plus matching `input_ids_*` / `attention_mask_*`. The eval notebook reads only the instruction-language subset configured for each condition.

## Kaggle Workflow

The pipeline is two-stage. **You must attach the preprocess output to evaluate.ipynb as an input.** Preprocess writes to `/kaggle/working/...` but evaluate reads from `/kaggle/input/...`, so an eval session with no preprocess input stops at its setup cell.

Preprocess runs once. Every later eval session, including a new seed, attaches the same saved output, so all sessions read identical prompts, token ids and gold labels. Rerun preprocess only when `preprocess.ipynb` itself changes, and expect `refresh_seed_results.py` to report input differences against the published sessions if you do.

### Stage 1 — Preprocess (CPU)

1. Create a Kaggle Notebook, upload `scripts/preprocess.ipynb`, set accelerator to CPU (GPU is wasted here).
2. Add `HF_TOKEN` to Kaggle Secrets (the notebook reads it via `UserSecretsClient`).
3. Run all cells. The notebook downloads XNLI, X-CSQA, SIB-200, Belebele for en/zh/es/ur and tokenizes prompts (4 instruction langs × 2 templates × 4 benchmarks) against `CohereLabs/tiny-aya-base`.
4. Commit (Save Version). 16 JSONL files land in `/kaggle/working/eval_unsloth_artifacts/datasets/`.

### Stage 2 — Make the output attachable

The eval notebook can attach the saved preprocess notebook's output directly: in **Add Input**, pick the saved preprocess notebook. To share it more widely, publish it instead: from the saved version, use **Output → New Dataset** (or **File → Save As Dataset**) and pick a slug (e.g., `<your-username>/aya-eval-cache-phase3`).

### Stage 3 — Evaluate (2× T4 GPU)

1. Create a Kaggle Notebook, upload `scripts/evaluate.ipynb`, set accelerator to **GPU T4 ×2** and turn internet on. Add `HF_TOKEN` to Kaggle Secrets: the notebook uses it to download the adapter and, in the upload cell, to open the results pull request, so the token needs write access to `legesher/language-decoded-experiments` if you upload from Kaggle.
2. **Add Input** → attach the preprocess output from Stage 2. Attach the preprocess output, not a training notebook's output: the setup cell looks for `xnli_en.jsonl` under `/kaggle/input/`.
3. **Set `CONDITION` and `SEED`** in the picker cell, the notebook's first cell. Both ship empty, with examples in comments, and the cell raises an error until both are set, so a notebook run as it is stops before installing anything. Use `SEED = "none"` for baseline, otherwise one of the seeds the picker comment lists for that condition (the comment mirrors `EVAL_MATRIX`).

   The same cell holds `N_SAMPLES`. Leave it `None` for a full run. Set it to a small number (e.g., `20`) for a smoke test that runs the whole matrix on the first N rows of each benchmark in a few minutes; its output files carry a `_smoke{N}` infix and upload to `phase3/smoke/`, so they never overwrite a full run.

4. **Run the session-setup cell.** It searches `/kaggle/input/` for the preprocess output, checks the cached columns (`prompt_template{1,2}_{en,zh,es,ur}` and the matching `input_ids_*`), and sets `KAGGLE_DATASET_INPUT_DIR` for the eval processes. It raises if it finds no cache or the columns are missing. If more than one cache is attached it lists them and sets nothing; set the path yourself in that case, then continue:

   ```python
   import os
   os.environ["KAGGLE_DATASET_INPUT_DIR"] = "<one of the listed paths>"
   ```

   The eval script raises `RuntimeError` if `KAGGLE_DATASET_INPUT_DIR` is still unset when it starts.

5. **Run the launcher cell.** Two subprocesses spawn, GPU 0 running `template1` and GPU 1 running `template2`, and a background thread streams both log files back to the cell output so you can watch progress live. The cell blocks until both subprocesses finish, then prints `=== Both templates done ===`.

   Output JSONs land in `/kaggle/working/`:
   - `{condition}_seed{seed}_summary_{template}.json` — per-cell accuracies + parse-failure rates
   - `{condition}_seed{seed}_results_{template}.json` — full per-row outputs (includes `raw_output` so you can re-parse offline if extractors change)
   - `{condition}_seed{seed}_partial_{template}.json` — incremental checkpoint (updates after each `(dataset_lang, instruction_lang)` block; survives mid-run crashes within Kaggle's 12h cap)

   > **Single-GPU fallback:** if you only have one T4 (or want simpler live output without the log-streaming thread), replace the launcher cell with `!python /kaggle/working/run_eval_single.py --condition {CONDITION} --seed {SEED} --batch_size 32`. The script loops over both templates serially in one foreground process. Roughly 2× wall-clock vs. the dual-GPU launcher.

6. **Run the upload cell** once the launcher has finished. It uploads the four summary and results JSONs (not the partial checkpoints or logs) to `phase3/conditions/{condition}/seed{seed}/` on `legesher/language-decoded-experiments` as a Hugging Face pull request and prints the PR link; nothing is written to `main` directly. A PR opened from Kaggle can land as a draft. Commit the notebook too if you want the outputs kept on Kaggle.

7. **Repeat for each `(condition, seed)`.** With the seeds in the table above, the full sweep is **29 Kaggle sessions**:
   - 1 baseline
   - 3 cond-1-en-5k (seeds 42, 123, 456) + 2 cond-1-en-20k (seeds 42, 123)
   - 9 cond-2-{zh,es,ur}-5k (3 langs × 3 seeds) + 6 cond-2-{zh,es,ur}-20k (3 langs × 2 seeds)
   - 2 cond-3-zh-5k (seeds 42, 123)
   - 6 cond-5-{zh,es,ur}-5k (3 langs × 2 seeds)

   Each further seed adds one session per condition it is run for; see [Running a new seed](#running-a-new-seed).

> **Schema note:** if you have cached JSONLs from an earlier version of this pipeline (the PR #37-era `english`/`language` column suffixes), delete `eval_unsloth_artifacts/datasets/*.jsonl` and rerun preprocess before running evaluate. The current schema uses `prompt_{template_id}_{instruction_lang}` columns.

## Running a new seed

A new seed is any `(condition, seed)` session that is not in the [eval matrix table](#per-condition-eval-matrix) yet. Run the commands below from the repository root with `huggingface_hub` installed; `status` reads Hugging Face, so leave `HF_HUB_OFFLINE` unset.

1. **Find the sessions still to run.**

   ```sh
   python expedition-tiny-aya/evaluation/scripts/refresh_seed_results.py status
   ```

   It reads the ladder from `EVAL_MATRIX` in `evaluate.ipynb`, checks Hugging Face for every registered `(condition, seed)`, prints a table and writes `status.md` / `status.json` under `--refresh-dir` (default `/tmp/phase3_refresh`). The state column says what each session needs:

   - `awaiting adapter`: the adapter is not on `main` of `legesher/language-decoded-lora`. Train it first, and merge the training notebook's pull request before evaluating.
   - `adapter misnamed`: an adapter for this seed is on `main` under a folder name the eval template does not expect (step 3).
   - `awaiting eval`: the adapter is there and the results are not. This session is ready to run.
   - `results PR #N open` or `results partial`: the eval has run; merge or finish its results upload (step 5).
   - `needs reparse` or `reparse PR #N open`: the results are on `main`; the refined summaries are next (step 6).
   - `complete`: the reparsed summaries for both templates are on `main`.

   `status` does not count draft pull requests, and the upload cell's PRs can land as drafts, so also check the open pull requests on `legesher/language-decoded-experiments` before starting a session that may already have run.

2. **Register the seed if `status` does not list it.** A seed found on Hugging Face but missing from `EVAL_MATRIX` is reported as `(unregistered)` and skipped, and `run_eval_single.py` refuses an unregistered seed. Add it to the condition's `seeds` list in `EVAL_MATRIX` (the `%%writefile` cell of `evaluate.ipynb`) and to the picker comment in the first cell, in a pull request to this repository. `test_refresh_seed_results.py` fails if the two disagree.

3. **Check the adapter folder name.** The eval loads `tiny-aya-base/{data config}-seed{seed}` (see the [naming rule](#per-condition-eval-matrix)). For condition 5 the folder carries the `-c4ai-aya-expanse-32b` infix. A folder for the right condition and seed under another name shows as `adapter misnamed`; rename it on the LoRA repo rather than changing the template.

4. **Run the session** with [Stage 3](#stage-3--evaluate-2-t4-gpu): attach the same preprocess output the published sessions used, set `CONDITION` and `SEED`, run the launcher, then the upload cell. A smoke run with `N_SAMPLES` first is cheap and catches a missing adapter or cache before the multi-hour run.

5. **Merge the results pull request** on `legesher/language-decoded-experiments` once it carries the four JSONs for both templates.

6. **Rescore, verify and rebuild the tables.**

   ```sh
   python expedition-tiny-aya/evaluation/scripts/refresh_seed_results.py all \
     --provisional '<condition>/seed<N>' ... --upload
   ```

   `all` reruns `status`, reparses each new session with the refined extractor, verifies it against the condition's reference session (seed 42 where it exists: same cells, row counts, inputs and gold labels), rebuilds the refined tables, the cross-seed variance files and the figures, and with `--upload` opens one Hugging Face PR per reparsed session plus one for the changed tables. It never merges. Run it again after each session lands; it reports what changed since the last run. It needs `scipy` as well. `--provisional` labels sessions whose results are not final yet (the seed-123 batch used `'*-20k/seed123' 'condition-3-*/seed123' 'condition-5-*/seed123'`); whether a new seed carries the label is a decision to make before uploading. The step-by-step version is [`analysis/phase-3/data-refresh-cookbook.md`](../analysis/phase-3/data-refresh-cookbook.md).

7. **Audit the batch** once every session of the seed is merged and the tables are rebuilt:

   ```sh
   python expedition-tiny-aya/evaluation/scripts/refresh_seed_results.py audit --seed <N> \
     --provisional '<condition>/seed<N>' ...
   ```

   It is read-only and exits non-zero on a missing session file, a session whose cells, rows, inputs or gold labels differ from the reference seed, reparsed counts that disagree with the results file, or LoRA settings and training steps that differ from the reference adapter. It writes `audit-seed<N>.md` under `--refresh-dir`.

8. **Update this README** when the results are on `main`: add the seed to the condition's row in the [eval matrix table](#per-condition-eval-matrix) and the adapter tree, and recount the sweep in Stage 3 step 7. The table may trail `EVAL_MATRIX` but never list a seed it does not register, and the sweep count must match the table; `test_refresh_seed_results.py` checks both.

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

See [analysis/phase-2/evaluation-summary.md](../analysis/phase-2/evaluation-summary.md) for the full analysis.
