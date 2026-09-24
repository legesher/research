# research

Research code and analysis for Legesher's multilingual programming work. The
active project is Expedition Tiny Aya (Language Decoded): the code that produced
the datasets, adapters and evaluation results published on Hugging Face. Data,
weights and raw results live on Hugging Face, not here; the HF dataset card is
the scientific source of truth and this repository is the navigational one.

## Layout

- `expedition-tiny-aya/README.md`: project entry point, the Phase 3 ladder, reproduction pointers.
- `expedition-tiny-aya/data-pipeline/`: configs, scripts and logs that build the training corpora.
- `expedition-tiny-aya/transpilation/`: transpilation tooling and its generated
  `results/reports/*.json` (marked `linguist-generated` in `.gitattributes`).
- `expedition-tiny-aya/training/`: LoRA training configs and scripts.
- `expedition-tiny-aya/evaluation/scripts/`: the eval notebooks, the refined extractor
  (`reparse_results.py`), the `build_*.py` analysis-table scripts, the `upload_*.py` publishers,
  and the `test_*.py` unittest suite. `evaluation/requirements.txt` is the full eval stack.
- `expedition-tiny-aya/analysis/`: per-phase writeups (`phase-2/`, `phase-3/`), figures,
  notebooks, and `WHEN_REPORTED_NUMBERS_CHANGE.md` (what to update when a number moves).
- `expedition-tiny-aya/language-review/`: native-speaker review material per language.
- `expedition-tiny-aya/assets/`: images used by the READMEs.
- `.github/workflows/tests.yml`: the SHA-pinned unittest workflow. `.github/FUNDING.yml`.

## Validate locally

Run from the repository root:

```sh
python3 -m pip install huggingface_hub     # the scripts import it at module level
HF_HUB_OFFLINE=1 python3 -m unittest discover -s expedition-tiny-aya/evaluation/scripts -p 'test_*.py' -v
```

Summary lines to expect (count as of 2026-09-24; it grows, it never shrinks):

- `Ran 139 tests in 0.1s` then `OK`

`.github/workflows/tests.yml` runs the same command on Python 3.12 with `HF_HUB_OFFLINE: "1"` set
for the job; Python 3.14 also passes locally.

Facts behind that command:

- The suite is stdlib `unittest`; there is no pytest configuration and nothing to install beyond
  `huggingface_hub`. The tests themselves make no network calls and patch every Hub call.
- Each test module inserts its own directory on `sys.path` and imports the script under test by
  bare module name (`import reparse_results`), so no package install is needed.
- The `build_*.py` scripts that rebuild published tables do download per-row data from Hugging Face
  and are not part of the test run.

## Merge evidence and PR conventions

- Branch `madi/core-####-slug` (or your own handle), one branch per unit of work.
- Squash merge. Subject `type(scope): summary [CORE-####] (#N)`; GitHub appends the `(#N)`.
- The branch ruleset requires a review before merge.
- Quote the unittest summary (`Ran N tests` and `OK`) in the PR body. CI may be queued; the local
  validate run is recorded in the PR body.
- When a change moves a reported number, follow `analysis/WHEN_REPORTED_NUMBERS_CHANGE.md` and say
  so in the PR body; artifacts that reproduce byte-identically are the evidence a fix was latent.
- No em dashes in prose, commit messages or PR bodies.

## Testing contract

- Known failures are named `unittest.skip` / `expectedFailure` ledgers citing a ticket, never
  weakened assertions.
- Parametrize from the real data (the condition and language registries the scripts read), never a
  frozen list.
- Prove every new guard or validation test can fail in a throwaway copy before raising the PR, and
  record the probe (what was injected, which named failures fired) in the PR body.
- Latin identifiers prove nothing. Unicode assertions use combining-mark scripts: Devanagari,
  Sinhala with ZWJ, pointed Hebrew, vocalised Arabic, Thai, CJK.
- The repo-hygiene guard `expedition-tiny-aya/evaluation/scripts/test_repo_hygiene.py` runs in the
  default test command and fails on a tracked folderOpen editor task, an automatic-tasks setting,
  non-font bytes under a font extension, or `.env` not ignored or tracked.

## Gotchas

- Set `HF_HUB_OFFLINE=1` for the test run. Without it an unpatched Hub call reaches the network
  instead of failing fast, and a test that passes for the wrong reason is worse than one that fails.
- Always pass `-s expedition-tiny-aya/evaluation/scripts`. The directory has no `__init__.py`, so
  discovery started from a parent finds nothing and reports `Ran 0 tests` then `NO TESTS RAN`.
- Data and model files (`*.parquet`, `*.arrow`, `*.safetensors`, `*.bin`, `*.pt`) and local script
  output (`rescored_results/`) are gitignored. They live on Hugging Face; never commit them.
- The transpilation reports under `results/reports/` are generated. Regenerate them, do not edit.
- `__pycache__/` appears under `evaluation/scripts/` after a test run; it is ignored.
