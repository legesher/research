# Expedition Tiny Aya

Language, Decoded: Exploring the Impact of Native-Language Code on Multilingual Models

> Captains: Madison Edgar (Legesher), Saad Ahmed Bazaz (Grayhat), et al. | Collaboration: Cohere / Expedition Tiny Aya
> Hackathon: March 10–24, 2026

## Research Question

**Primary:** Does multilingual code improve Tiny Aya's reasoning, world knowledge, and instruction following in the language that code is written in?

**Secondary:** Does it matter _how_ code is created? Does code natively written by developers who think in a language produce better improvements than code mechanically transpiled from English?

## Experimental Design

Fine-tune Tiny Aya base (3.35B params) with QLoRA under 4 conditions:

| Condition          | Data Config                               | Data Source                           | What It Tests                       |
| ------------------ | ----------------------------------------- | ------------------------------------- | ----------------------------------- |
| Baseline           | —                                         | Tiny Aya base, no additional training | Floor                               |
| 1. English code    | `condition-1-en-32k`, `condition-1-en-5k` | English Python from The Stack         | Replicates original paper           |
| 2. Transpiled code | `condition-2-{zh,es,ur}-5k`               | English code transpiled via Legesher  | Token exposure alone                |
| 3. Mixed native    | `condition-3-zh-5k`                       | Transpiled + native code blend        | Code structure + native patterns    |
| 4. Combined native | `condition-4-zh-5k`                       | All strictly native code              | Token exposure + cognitive patterns |

## Target Languages

| Language     | Script           | Why                                         |
| ------------ | ---------------- | ------------------------------------------- |
| Chinese (zh) | CJK (Simplified) | Best-documented identifier paradox          |
| Spanish (es) | Latin            | High-resource, cross-lingual transfer test  |
| Urdu (ur)    | Nastaliq (RTL)   | RTL, cross-lingual transfer test with Hindi |

## Evaluation Benchmarks

| Benchmark  | What It Measures                                              | Languages      |
| ---------- | ------------------------------------------------------------- | -------------- |
| **MGSM**   | Multilingual math reasoning (grade-school word problems)      | zh, es, ur, en |
| **XNLI**   | Natural language inference (entailment/contradiction/neutral) | zh, es, ur, en |
| **X-CSQA** | Commonsense reasoning (5-way multiple choice)                 | zh, es, ur, en |

English benchmarks are included to detect catastrophic forgetting.

## Team

| Person             | GitHub                                                 |
| ------------------ | ------------------------------------------------------ |
| Madi Edgar         | [@madiedgar](https://github.com/madiedgar)             |
| Saad A. Bazaz      | [@SaadBazaz](https://github.com/SaadBazaz)             |
| Rashik Shahjahan   | [@RashikShahjahan](https://github.com/RashikShahjahan) |
| Khojasteh Mirza    | [@vulcan-332](https://github.com/vulcan-332)           |
| Rafay Mustafa      | [@rafaym1](https://github.com/rafaym1)                 |
| Sarah Jawaid       | [@sarr266](https://github.com/sarr266)                 |
| Sohaib Ahmed Bazaz | [@SohaibBazaz](https://github.com/SohaibBazaz)         |
| Tom Sherborne      | Cohere                                                 |

## Native Code

Transpiled code is English logic wearing non-English syntax. Native code captures how developers actually think in their language. If native code produces better model improvements than transpiled code, it suggests that linguistic cognition embedded in code matters — not just token exposure. Native code samples are collected in [language-decoded-community](https://huggingface.co/datasets/legesher/language-decoded-community) on HuggingFace.

## Repository Structure

```
expedition-tiny-aya/
├── paper/              # Research write-up (LaTeX)
├── demo/               # Presentation slides, speaker notes
├── transpilation/      # Batch transpilation scripts and stress tests
├── training/           # QLoRA training notebook (qlora.ipynb)
├── evaluation/         # Benchmark scripts and notebooks
├── analysis/           # Evaluation analysis and visualization
├── data-pipeline/      # The Stack streaming, filtering, packaging
└── language-review/    # Per-language keyword review and sign-off
```

## HuggingFace Repos

All datasets, trained adapters, and results live on HuggingFace — not in this repo.

| Repo                                                                                                  | Purpose                                                    |
| ----------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| [language-decoded-data](https://huggingface.co/datasets/legesher/language-decoded-data)               | Training datasets (per-condition configs, 5k and 32k)      |
| [language-decoded-lora](https://huggingface.co/legesher/language-decoded-lora)                        | Trained LoRA adapters (per-condition subfolders)           |
| [language-decoded-experiments](https://huggingface.co/datasets/legesher/language-decoded-experiments) | Eval results, training configs, condition metadata         |
| [language-decoded-community](https://huggingface.co/datasets/legesher/language-decoded-community)     | Human-written native code samples (Condition 3 and 4 data) |

## Key References

- "To Code, or Not To Code?" (Aryabumi et al., 2024) — code improves NL reasoning ~8%
- "Unveiling the Impact of Coding Data Instruction Fine-Tuning" (Luo et al., AAAI 2025) — 20-30% improvement
- "Linguistic Relativity and Programming Languages" (Chen, 2018) — Sapir-Whorf applies to code
- "The Stack" (Kocetkov et al., 2022) — 6.4TB code, all English keywords
- "LIMA" (Zhou et al., NeurIPS 2023) — 1,000 curated examples matched 52,000 lower-quality

## Links

- [Legesher Monorepo](https://github.com/MadiEdgar/legesher) (transpilation tooling)
- [HuggingFace Collection](https://huggingface.co/collections/legesher/language-decoded)
