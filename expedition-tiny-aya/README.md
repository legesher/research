# Expedition Tiny Aya — Language Decoded

_Paper: **Language, Decoded: Exploring the Impact of Fine-Tuning a Multilingual Model on Native-Language Code**._

In-repo home for the **Language Decoded** project: code, scripts, configs, transpilation tooling, evaluation pipeline, and per-phase analysis. Datasets, trained adapters, and raw evaluation outputs live on Hugging Face — this repo is what produces them.

> Project lead: Madi Edgar (Legesher) · Technical lead: Saad Ahmed Bazaz (Grayhat) · Research mentor: Tom Sherborne (Cohere). Originated as a proposal during [Cohere's Tiny Aya Expedition](https://aya.for.ai) (March 2026 hackathon) and extended into Phase 3 for the accompanying paper.

## Canonical source of truth

The scientific reference for this project is the **dataset card on Hugging Face**:

- **[`legesher/language-decoded-experiments`](https://huggingface.co/datasets/legesher/language-decoded-experiments)** — full methodology, the Phase 3 experimental ladder, the refined-extractor banner, per-condition framing, evaluation matrix, and citation.

This in-repo README is the **navigational** entry point — what code lives where, how to reproduce, where to read the analysis. For every methodological question, defer to the HF card.

---

## Phase 2 → Phase 3 in one paragraph

The project has run in two phases. **The accompanying paper is Phase 3.** Phase 2 (March 2026 hackathon) trained on `bigcode/the-stack` (v1) with Legesher v0.5.1 / v0.6.0, evaluated on MGSM, X-CSQA, and XNLI, and produced single-seed results scored by an inference-time extractor. Phase 3 re-trained the cond-1, cond-2, and cond-5 adapters from scratch on a cleaner subset of [`bigcode/the-stack-v2-dedup`](https://huggingface.co/datasets/bigcode/the-stack-v2-dedup) with Legesher v0.7.3, added 20k variants and three seeds at 5k, swapped MGSM for **SIB-200** and **Belebele**, and added a **refined post-hoc extractor** for paper-grade scoring on free-form answers. Cond-3's training corpus is unchanged across phases (community-collected raw); Cond-3's adapter was re-trained for Phase 3 on the same dataset. See the [Phase 2 → Phase 3 table](https://huggingface.co/datasets/legesher/language-decoded-experiments#phase-2--phase-3-at-a-glance) on the HF card for the full diff.

> **Paper-grade numbers come from the refined extractor.** Original Phase 3 `_summary_*.json` files under-report cond-5 SIB-200 accuracy by 20–35pp because the strict inference-time extractor refused native-script answers. Cite the `_summary_reparsed_*.json` siblings. The refined extractor and its provenance live at [`expedition-tiny-aya/evaluation/scripts/reparse_results.py`](evaluation/scripts/reparse_results.py); the analysis writeup is at [`expedition-tiny-aya/analysis/phase-3/`](analysis/phase-3/).

---

## The Phase 3 Experimental Ladder

Each row trains a separate per-language LoRA adapter on top of `CohereLabs/tiny-aya-base` (3.35B). The ladder is cumulative and isolating — each condition changes exactly one variable from a specific predecessor.

| # | Condition | What its training corpus is | What it isolates |
| --- | --- | --- | --- |
| 0 | **Baseline** | No fine-tuning. The base `CohereLabs/tiny-aya-base` model, re-evaluated on the Phase 3 matrix. | Floor — what does Tiny Aya know already? |
| 1 | **English Code** | The shared 5k (and parallel 20k) file subset from `bigcode/the-stack-v2-dedup`, raw English Python. | Does code help at all? (Aryabumi replication.) |
| 2 | **Reserved-Word Translation (Legesher)** | The **same 5k / 20k files as cond-1**, processed through Legesher v0.7.3. Python's reserved words — keywords, exceptions, built-in functions, and (for some target languages) the numerical system — translated to the target language; user logic and library calls preserved. | Does the language Python's reserved words are written in change model behavior, once file content is held constant? |
| 3 | **Mixed Native Sources** | `condition-3-zh-5k` only. Chinese code community-collected raw from **varied online public-source repositories** — reflecting how non-English Python is actually used in real-world projects. **Different source-file population from cond-1/2/5 by design.** | Does code humans actually wrote in or with the target language add value beyond Legesher's mechanical translation? |
| 5 | **Synthesized Native Code** | The **same 5k subset as cond-1**, first processed through Legesher v0.7.3 to translate Python's reserved words (as in cond-2), then run through [`c4ai-aya-expanse-32b`](https://huggingface.co/CohereLabs/aya-expanse-32b) via the Cohere API to translate everything else translatable — identifiers, comments, docstrings, string literals, and any other natural-language wording. Logic and structure preserved. | What happens when the **entire translatable content of a file** is rendered in the target language, not just Python's reserved words? |

**Cond-4 (Community-Contributed Native Code)** is not in Phase 2 or Phase 3. Its design goal is code whose problem-solving logic is itself native — written as if a native speaker of the target language were approaching the problem, not English code that was later translated. It is pending sufficient community contributions via the [`legesher-native-code` Space](https://huggingface.co/spaces/legesher/legesher-native-code); cond-5's fully-translated data was used in Phase 3 as the practical proxy.

The "Mixed Native Sources" name for cond-3 is a Phase 2 holdover; its meaning shifted between phases — in Phase 2 the label referred to a planned cond-2-plus-native composite, while in Phase 3 the physical dataset is unchanged and "mixed" refers to the diversity of source locations. See the HF card for the per-phase comparison.

### Source-file control (the experimental keystone)

**Cond-1, cond-2, and cond-5 all share the same 5,000-file subset** drawn from `bigcode/the-stack-v2-dedup` (and a parallel 20k subset at the 20k tier). The input file population is held constant; only the processing pipeline (raw / reserved-word-translated / fully translated) varies. Differences in downstream model behavior are attributable to the **processing step**, not to file-quality or content drift.

- Source is **Python-only** for cond-1, cond-2, and cond-5.
- **Cond-3 is the deliberate exception** — its community-collected corpus is a different source-file population and may include non-Python files (other non-English programming languages from public repositories) by design.
- Cond-5's `zh` and `es` runs are constrained to `ur`'s succeeded translation idxs → ~4,400 valid translations per language; `ur` has full 5k coverage. See [`analysis/phase-3/cond5-idx-ledger.md`](analysis/phase-3/cond5-idx-ledger.md).

### Target languages

`en` (Latin) · `zh` (CJK Simplified) · `es` (Latin) · `ur` (Nastaliq, RTL). English is the catastrophic-forgetting reference and is included on every condition; the other three span scripts and resource levels. The HF card has the full [Target Languages](https://huggingface.co/datasets/legesher/language-decoded-experiments#target-languages) table.

### Phase 3 evaluation suite

| Benchmark | Format | Phase 2 | Phase 3 |
| --- | --- | :---: | :---: |
| **XNLI** | 3-way NLI (entailment / contradiction / neutral) | ✓ | ✓ |
| **X-CSQA** | 5-way commonsense MC | ✓ | ✓ |
| **SIB-200** | 7-way topic classification (free-form answer) | — | ✓ |
| **Belebele** | 4-way reading-comprehension MC | — | ✓ |
| MGSM | Multilingual math reasoning | ✓ | **dropped** (null result at 3.35B, 250 examples/lang) |

Each Phase 3 session evaluates on a `4 benchmarks × template{1,2} × data_lang × instr_lang` matrix, 3 seeds at 5k, 1 seed at 20k. See [Evaluation Benchmarks](https://huggingface.co/datasets/legesher/language-decoded-experiments#evaluation-benchmarks) on the HF card for selection rationale per benchmark.

---

## Repository layout

```
expedition-tiny-aya/
├── analysis/
│   ├── phase-3/                  # Phase 3 paper-grade analysis (see below)
│   └── evaluation-summary.md     # Phase 2 headline writeup
├── data-pipeline/                # The Stack streaming, filtering, packaging
├── demo/                         # Presentation slides, speaker notes
├── evaluation/
│   └── scripts/
│       ├── reparse_results.py    # Refined post-hoc extractor (Phase 3, paper-grade)
│       ├── rescore_xnli.py
│       ├── build_*.py            # Cross-session table builders (refined-tables)
│       └── evaluate.ipynb        # Evaluation entry-point notebook
├── language-review/              # Per-language reserved-word review and sign-off
├── paper/                        # LaTeX write-up
├── training/                     # QLoRA training notebook (qlora.ipynb)
└── transpilation/                # Batch transpilation scripts and stress tests
```

### Where Phase 3 analysis lives

- **[`analysis/phase-3/phase3-refined-evaluation.md`](analysis/phase-3/phase3-refined-evaluation.md)** — headline Phase 3 writeup (refined-extractor numbers).
- **[`analysis/phase-3/post-refined-action-items.md`](analysis/phase-3/post-refined-action-items.md)** — open action items and their status.
- **[`analysis/phase-3/refined-decision-ledger.md`](analysis/phase-3/refined-decision-ledger.md)** — decisions made during the refined-extractor pass.
- **[`analysis/phase-3/sib200-parser-methodology.md`](analysis/phase-3/sib200-parser-methodology.md)** — SIB-200 refined parser methodology.
- **[`analysis/phase-3/{chinese,spanish,urdu}-surface-forms-review.md`](analysis/phase-3/)** — per-language surface-form audits.
- **[`analysis/phase-3/cond5-idx-ledger.md`](analysis/phase-3/cond5-idx-ledger.md)** — cond-5 idx-allowlist provenance (zh/es constrained to ur's succeeded idxs).
- **[`analysis/phase-3/aggregation-bug-audit.md`](analysis/phase-3/aggregation-bug-audit.md)** — audit of seed-vs-cell inflation across build scripts.
- **[`analysis/phase-3/captions.md`](analysis/phase-3/captions.md)** and **[`tables.tex`](analysis/phase-3/tables.tex)** — paper figure captions and booktabs result tables.

The **refined cross-session tables** themselves (`cells.tsv`, `vs_baseline_cells.tsv`, framework views, `conclusion_flips.tsv`, rollups) live on Hugging Face at [`legesher/language-decoded-experiments/phase3/analysis/refined-tables/`](https://huggingface.co/datasets/legesher/language-decoded-experiments/tree/main/phase3/analysis/refined-tables).

---

## Hugging Face — where the data and results live

All datasets, trained adapters, evaluation outputs, and analysis tables live on Hugging Face under the [`legesher`](https://huggingface.co/legesher) org.

| Repo | Type | Contents |
| --- | --- | --- |
| [`language-decoded-experiments`](https://huggingface.co/datasets/legesher/language-decoded-experiments) | Dataset | **Canonical source of truth.** Per-condition training logs, configs, raw evaluation outputs (`_results_*.json`), inference-time summaries (`_summary_*.json`), refined-extractor summaries (`_summary_reparsed_*.json`), and cross-session analysis tables. |
| [`language-decoded-data`](https://huggingface.co/datasets/legesher/language-decoded-data) | Dataset | Training data for every condition. Configs: `condition-{1,2}-{en,zh,es,ur}-{5k,20k,32k,103k}`, `condition-3-zh-5k`, `condition-4-zh-5k`, `condition-5-{zh,es,ur}-5k`. (32k = Phase 2 size; 20k = Phase 3 size; 103k uploaded but not yet evaluated.) |
| [`language-decoded-lora`](https://huggingface.co/legesher/language-decoded-lora) | Model | LoRA adapters (QLoRA 4-bit via Unsloth) for every trained condition, per-seed. |
| [`language-decoded-community`](https://huggingface.co/datasets/legesher/language-decoded-community) | Dataset | Human-written native-language code samples (cond-4 source corpus). |
| [`legesher-native-code` Space](https://huggingface.co/spaces/legesher/legesher-native-code) | Space | Community-contribution interface feeding the cond-4 corpus pipeline. |
| [`language-decoded` Collection](https://huggingface.co/collections/legesher/language-decoded) | Collection | All Language Decoded artifacts in one place. |

### Per-session file layout (on `language-decoded-experiments`)

Each Phase 3 session directory (e.g. `phase3/conditions/condition-5-ur-5k/seed42/`) contains, per template / benchmark:

- `*_results_*.json` — raw model outputs (frozen historical record).
- `*_summary_*.json` — inference-time extractor accuracies (frozen).
- `*_summary_reparsed_*.json` — **refined extractor accuracies + per-cell deltas. Cite these.**

---

## Reproducing the Phase 3 refined numbers

Top-level path:

1. **Datasets and adapters** are public on Hugging Face. Pulling `legesher/language-decoded-data` and `legesher/language-decoded-lora` is enough to retrain or re-evaluate any condition from scratch.
2. **Evaluation** runs through `evaluation/scripts/evaluate.ipynb` against the four Phase 3 benchmarks ([`facebook/xnli`](https://huggingface.co/datasets/facebook/xnli), [`INK-USC/xcsr`](https://huggingface.co/datasets/INK-USC/xcsr), [`mteb/sib200`](https://huggingface.co/datasets/mteb/sib200), [`facebook/belebele`](https://huggingface.co/datasets/facebook/belebele)) on the full `template{1,2} × data_lang × instr_lang` matrix.
3. **Refined post-hoc scoring** is applied by running [`evaluation/scripts/reparse_results.py`](evaluation/scripts/reparse_results.py) over the `_results_*.json` outputs; this produces the `_summary_reparsed_*.json` siblings and stamps `_extractor_provenance` (with a `content_sha256` of the extractor source) into every reparsed summary.
4. **Cross-session tables** are rebuilt with the `evaluation/scripts/build_*.py` scripts (comparison, framework, vs-baseline, correct-via-constant). Outputs are the `cells.tsv` / `vs_baseline_cells.tsv` / framework views / `conclusion_flips.tsv` files on the HF dataset.

The reproducibility block in [`analysis/phase-3/phase3-refined-evaluation.md`](analysis/phase-3/phase3-refined-evaluation.md#reproducibility) has the end-to-end recipe.

---

## Training setup (all conditions, both phases)

| Parameter | Value |
| --- | --- |
| Base model | [`CohereLabs/tiny-aya-base`](https://huggingface.co/CohereLabs/tiny-aya-base) (3.35B params, 70+ languages, low-resource emphasis) |
| Fine-tuning method | QLoRA 4-bit (NF4), ~5.4 GB VRAM, Unsloth-accelerated |
| Hardware | Kaggle T4 (16 GB) |
| Transpilation tool | [Legesher](https://github.com/legesher/legesher) — **Phase 2: v0.5.1 / v0.6.0**, **Phase 3: v0.7.3** |
| Phase 3 source corpus | [`bigcode/the-stack-v2-dedup`](https://huggingface.co/datasets/bigcode/the-stack-v2-dedup) (Phase 2 used [`bigcode/the-stack`](https://huggingface.co/datasets/bigcode/the-stack) v1) |
| Cond-5 translation model | [`c4ai-aya-expanse-32b`](https://huggingface.co/CohereLabs/aya-expanse-32b) (CC-BY-NC-4.0) via the Cohere API, made possible by Cohere credits awarded to Legesher |

Training configs are identical across conditions within a phase. Configs are checked in to the experiments dataset at [`configs/qlora-base.json`](https://huggingface.co/datasets/legesher/language-decoded-experiments/blob/main/configs/qlora-base.json).

---

## Team

| Person | Role | Owns |
| --- | --- | --- |
| **[Madi Edgar](https://github.com/madiedgar)** | Research lead | Coordination; evaluation pipeline |
| **[Saad Ahmed Bazaz](https://github.com/SaadBazaz)** | Technical lead | Eval pipeline, training configs |
| **[Sarah Jawaid](https://github.com/sarr266)** | Language owner | Chinese (zh); community collection |
| **[Sohaib Ahmed Bazaz](https://github.com/SohaibBazaz)** | Language owner | Spanish (es) |
| **[Rafay Mustafa](https://github.com/rafaym1)** | Language owner | Urdu (ur); evaluation pipeline; community collection |
| **[Khojasteh Mirza](https://github.com/vulcan-332)** | Eval lead | Evaluation pipeline, benchmarks |
| **[Rashik Shahjahan](https://github.com/RashikShahjahan)** | Data engineer | Data packaging, training environment |
| **Tom Sherborne** | Mentor (Cohere) | Research mentorship |

Kaggle compute runs (training and evaluation execution) were operated by Madi, Rafay, Rashik, and Khojasteh.

---

## Key references

- [Aryabumi et al., 2024 — _To Code or Not To Code?_](https://arxiv.org/abs/2408.10914) — English code in pre-training improves downstream reasoning by ~8% (the claim the ladder probes for non-English code).
- [Conneau et al., 2018 — XNLI](https://aclanthology.org/D18-1269/)
- [Lin et al., 2021 — X-CSQA](https://aclanthology.org/2021.acl-long.102/)
- [Adelani et al., 2024 — SIB-200](https://aclanthology.org/2024.eacl-long.14/)
- [Bandarkar et al., 2024 — Belebele](https://aclanthology.org/2024.acl-long.44/)
- [Kocetkov et al., 2022 — The Stack](https://arxiv.org/abs/2211.15533)

---

## Citation

```bibtex
@misc{language-decoded-2026,
  title={Language Decoded: Exploring the Impact of Native Code on Multilingual Models},
  author={Madison Edgar and Saad Ahmed Bazaz and Tom Sherborne and Rashik Shahjahan and Khojasteh Mirza and Sarah Jawaid and Rafay Mustafa and Sohaib Ahmed Bazaz},
  year={2026},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/legesher/language-decoded-experiments}
}
```

## License

Apache 2.0
