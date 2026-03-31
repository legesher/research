# Expedition Tiny Aya

Language, Decoded: Investigating Language-Dependent vs. Structure-Dependent Reasoning Benefits of Code

> Authors: Madison Edgar, Saad Ahmed Bazaz, et al. | Collaboration: Cohere / Expedition Tiny Aya
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

| Role                        | Person             | GitHub                                                 |
| --------------------------- | ------------------ | ------------------------------------------------------ |
| Research Lead               | Madi Edgar         | [@madiedgar](https://github.com/madiedgar)             |
| Technical Lead              | Saad A. Bazaz      | [@SaadBazaz](https://github.com/SaadBazaz)             |
| Training Engineer           | Rashik Shahjahan   | [@RashikShahjahan](https://github.com/RashikShahjahan) |
| Evaluation Engineer         | Khojasteh Mirza    | [@vulcan-332](https://github.com/vulcan-332)           |
| Language Reviewer (Urdu)    | Rafay Mustafa      | [@rafaym1](https://github.com/rafaym1)                 |
| Language Reviewer (Urdu)    | Sarah Jawaid       | [@sarr266](https://github.com/sarr266)                 |
| Language Reviewer (Spanish) | Sohaib Ahmed Bazaz | [@SohaibBazaz](https://github.com/SohaibBazaz)         |
| Mentor (Cohere)             | Tom Sherborne      | —                                                      |

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
├── language-review/    # Per-language keyword review and sign-off
└── native-code/        # Native code collection guide and review
```

## HuggingFace Repos

| Repo                                                                                                  | Purpose                                            |
| ----------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| [language-decoded-data](https://huggingface.co/datasets/legesher/language-decoded-data)               | Training datasets (per-condition configs)          |
| [language-decoded-lora](https://huggingface.co/legesher/language-decoded-lora)                        | Trained LoRA adapters (per-condition subfolders)   |
| [language-decoded-experiments](https://huggingface.co/datasets/legesher/language-decoded-experiments) | Eval results, training configs, condition metadata |
| [language-decoded-community](https://huggingface.co/datasets/legesher/language-decoded-community)     | Human-written native code samples                  |

## Key References

- "To Code, or Not To Code?" (Aryabumi et al., 2024) — code improves NL reasoning ~8%
- "Unveiling the Impact of Coding Data Instruction Fine-Tuning" (Luo et al., AAAI 2025) — 20-30% improvement
- "Linguistic Relativity and Programming Languages" (Chen, 2018) — Sapir-Whorf applies to code
- "The Stack" (Kocetkov et al., 2022) — 6.4TB code, all English keywords
- "LIMA" (Zhou et al., NeurIPS 2023) — 1,000 curated examples matched 52,000 lower-quality

## Links

- [Linear Project](https://linear.app/legesher-research)
- [Legesher Monorepo](https://github.com/MadiEdgar/legesher) (transpilation tooling)
- [HuggingFace Collection](https://huggingface.co/collections/legesher/language-decoded)
