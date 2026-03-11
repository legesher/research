# Expedition Tiny Aya

**Language, Decoded: Exploring the Impact of Native-Language Code on Multilingual Models**

Authors: Madi Edgar + Saad | Collaboration: Cohere / Expedition Tiny Aya
Hackathon: March 10–24, 2026

## Research Question

**Primary:** Does multilingual code improve Tiny Aya's reasoning, world knowledge, and instruction following in the language that code is written in?

**Secondary:** Does it matter *how* code is created? Does code natively written by developers who think in a language produce better improvements than code mechanically transpiled from English?

## Experimental Design

Fine-tune Tiny Aya base (3.35B params) with QLoRA under 4 conditions:

| Condition | Data Source | What It Tests |
| --- | --- | --- |
| Baseline | Tiny Aya base, no additional training | Floor |
| 1. English code | English Python from The Stack | Replicates original paper |
| 2. Transpiled code | English code transpiled via Legesher | Token exposure alone |
| 3. NL text (control) | Equivalent token volume of NL text | Code structure vs. more data |
| 4. Native code | Code written by native-language developers | Token exposure + cognitive patterns |

## Target Languages

| Language | Script | Tiny Aya Variant | Why |
| --- | --- | --- | --- |
| Chinese (zh) | CJK (Simplified) | Water | Best-documented identifier paradox |
| Amharic (am) | Ge'ez (Ethiopic) | Earth | Genuinely low-resource, unique script |
| Urdu (ur) | Nastaliq (RTL) | Fire | RTL, cross-lingual transfer test with Hindi |

## Evaluation Benchmarks

- **Reasoning:** XNLI, XStoryCloze in target languages
- **World Knowledge:** MMLU translated subsets, TyDi QA
- **Instruction Following:** LLM-as-judge evaluation
- **Culturally-grounded:** MultiNRC, AI4Math, native speaker evaluation
- **Cross-lingual Transfer:** Does code in one language help related languages?

## Team

| Role | Person | Directory |
| --- | --- | --- |
| Captain / Research Lead | Madi | `transpilation/`, `demo/` |
| Captain / Technical Lead | Saad | `training/`, `evaluation/`, `analysis/` |
| Pipeline Engineer | TBD | `data-pipeline/` |
| NL Curator | TBD | `nl-text/` |
| Language Champions | TBD | `language-review/` |
| Native Code Lead | TBD | `native-code/` |

## Repository Structure

```
expedition-tiny-aya/
├── paper/              # Research write-up (shared across team)
├── demo/               # Presentation slides, speaker notes
├── transpilation/      # Batch transpilation scripts, configs, results
├── training/           # QLoRA training configs, scripts, W&B links
├── evaluation/         # Benchmark scripts, per-condition results
├── analysis/           # Jupyter notebooks, generated figures
├── data-pipeline/      # The Stack streaming, filtering, packaging
├── nl-text/            # CC-100/OSCAR pull, clean, tokenize, package
├── language-review/    # Per-language review notes and sign-off
├── native-code/        # Contributor exercises, guide, quality rubric
├── datasets/           # HuggingFace dataset cards (data lives on HF)
└── models/             # HuggingFace model cards per condition
```

## Key References

- "To Code, or Not To Code?" (Cohere, 2024) — code improves NL reasoning ~8%
- "Unveiling the Impact of Coding Data Instruction Fine-Tuning" (Luo et al., AAAI 2025) — 20-30% improvement
- "Linguistic Relativity and Programming Languages" (Chen, 2018) — Sapir-Whorf applies to code
- "The Stack" (Kocetkov et al., 2022) — 6.4TB code, all English keywords
- "LIMA" (Zhou et al., NeurIPS 2023) — 1,000 curated examples matched 52,000 lower-quality

## Links

- [Linear Project](https://linear.app/legesher-research)
- [Legesher Monorepo](https://github.com/MadiEdgar/legesher) (transpilation tooling)
- HuggingFace Organization: TBD