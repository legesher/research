# The Stack Dataset: Non-English Content Analysis

**Prepared for:** Expedition Tiny Aya (Cohere Labs)
**Author:** Madison Edgar (Legesher)
**Date:** March 2026

---

## Executive Summary

An analysis of Hugging Face's The Stack dataset and supporting academic research reveals
that while ~4% of Python code comments are in non-English languages (spanning ~40 languages),
**no public dataset contains code where the programming keywords themselves are non-English**.
Developers worldwide actively use their native language in comments (99% of Chinese projects
have non-ASCII comments), but are forced into English for code structure. This gap represents
a unique opportunity: Legesher can generate the first multilingual code datasets by transpiling
Python from The Stack, enabling the Expedition Tiny Aya experiment to test whether multilingual
code transfers reasoning benefits the way English code does.

---

## 1. The Stack Dataset Overview

### The Stack v1 (2022)

- **Size:** 3 TB (v1.0) / 6 TB (v1.1) of permissively-licensed source code
- **Programming languages:** 358
- **Source:** GitHub repositories with permissive licenses
- **Paper:** Kocetkov et al., [arXiv:2211.15533](https://arxiv.org/abs/2211.15533)
- **Dataset:** [bigcode/the-stack](https://huggingface.co/datasets/bigcode/the-stack)

### The Stack v2 (2024)

- **Size:** Significantly larger (Software Heritage archive + GitHub)
- **Programming languages:** 619
- **Additional sources:** Pull requests, Kaggle notebooks, code documentation,
  free programming books
- **Paper:** Lozhkov et al., [arXiv:2402.19173](https://arxiv.org/abs/2402.19173)
- **Dataset:** [bigcode/the-stack-v2](https://huggingface.co/datasets/bigcode/the-stack-v2)
- **Notable:** Non-English Markdown was **actively filtered out** from Pull Requests
  during preprocessing

### The Stack GitHub Issues

- **Size:** 54 GB / ~31M rows
- **Content:** Conversations from GitHub issues and PRs (events, comments, metadata)
- **Language:** Described as "mostly in English" -- no per-language breakdown provided
- **Dataset:** [bigcode/the-stack-github-issues](https://huggingface.co/datasets/bigcode/the-stack-github-issues)

**Critical observation:** In all versions, every one of the 358--619 programming languages
uses **English-based keywords**. The non-English content is entirely in natural language
(comments, docstrings, issues) -- never in the code structure itself.

---

## 2. Natural Language Distribution in Code

### Overall Statistics (The Stack v1, Python subset)

The Stack paper analyzed 10,000 Python files for natural language in comments and docstrings:

- **~96% English**
- **~4% non-English** (across ~40 detected languages)
- Language detection is imperfect -- docstrings often include English code examples
  mixed with non-English prose

### Detected Languages

The following natural languages were detected in comments and docstrings:

> EN, ZH, FR, PT, ES, RU, DE, KO, JA, UZ, IT, ID, RO, AR, FA, CA, HU, ML, NL,
> TR, TE, EL, EO, BN, LV, GL, PL, GU, CEB, IA, KN, SH, MK, UR, SV, LA, JKA,
> MY, SU, CS, MN

### Approximate Per-Language Ranking

Based on the Stanford study's commit message analysis (1.1M GitHub users) and
correlated data, the approximate ranking of non-English languages on GitHub:

| Rank | Language | % of Non-English Commits | Script Type | Source |
| --- | --- | --- | --- | --- |
| 1 | Chinese (ZH) | 28.6% | Logographic | Piech & Abu-El-Haija, 2019 |
| 2 | Spanish (ES) | Not published | Latin | Ranked 2nd in paper |
| 3 | Portuguese (PT) | Not published | Latin | Ranked 3rd in paper |
| 4 | French (FR) | Not published | Latin | Ranked 4th in paper |
| 5 | Japanese (JA) | Not published | Mixed | Ranked 5th in paper |

**Important caveats:**

- Only the Chinese percentage (28.6%) is explicitly stated in the Stanford paper.
  The remaining languages are listed in ranked order but without published percentages.
- These rankings are from **commit message** analysis, not from code comments/docstrings
  in The Stack. The two distributions may differ.
- Exact per-language percentages for The Stack's comments/docstrings have not been
  published. Generating these numbers is one goal of our analysis script
  (see `scripts/analyze_stack_languages.py`).

---

## 3. The Identifier vs. Comments Paradox

The most striking finding comes from the Stanford paper "Human Languages in Source Code"
(Piech & Abu-El-Haija, [arXiv:1909.04556](https://arxiv.org/abs/1909.04556), 2019), which analyzed
**2.9 million Java repositories** from **1.1 million GitHub users**.

### The Core Paradox

Developers who speak non-Latin-script languages overwhelmingly write their **code identifiers**
(variable names, function names, class names) in English -- but write their **comments** in
their native language. The tooling allows native-language expression in comments but not in
code structure.

### By Language Community

| Metric | Chinese | Spanish | Arabic/Hebrew |
| --- | --- | --- | --- |
| Projects with ASCII-only identifiers | 93% | 88% | 99.4% |
| Projects with non-ASCII comments | 99% | 53% | Very low |
| Users who write identifiers in their language | 23.3% | 87.2% | 0.6% |

### Why This Happens

**Latin-script advantage:** Spanish speakers can write `numero` instead of `numero` (drop
the accent) and it still reads naturally. Chinese and Arabic speakers cannot transliterate
-- they must either use full Unicode characters or write in English.

**Technical barriers:** Many languages, tools, and IDEs historically didn't support Unicode
identifiers well. RTL scripts (Arabic, Hebrew) are especially problematic.

**Cultural norm:** English identifiers are the convention, so even developers comfortable
in their native language follow it to maintain readability for collaborators.

### RTL Languages: The Most Severe Barrier

The Stanford study examined 18,961 GitHub users from RTL-language countries with 8,060
Java repositories:

- Only **50 repositories (0.6%)** had Arabic or Hebrew script in identifiers
- Only **a single Java file** in the entire dataset had a single Arabic identifier
- This implies **substantial barriers** for Arabic speakers worldwide

### What This Means

The data proves that developers **want** to express themselves in their native language.
They do it wherever they can (comments: 99% for Chinese). They are **prevented** from
doing so in code structure by the English-only design of programming languages. This is
the problem Legesher solves -- and the gap our Expedition Tiny Aya experiment exploits.

---

## 4. Non-English Programming Languages

These demonstrate that non-English code is both desired and technically viable:

| Language | Based On | Users/Scale | Keywords | Status |
| --- | --- | --- | --- | --- |
| **1C:Enterprise** | Russian | 5M users, 1.5M+ orgs, ~33% of Russian enterprise market (IDC) | Russian OR English (toggle freely) | Production, enterprise |
| **Wenyan** | Classical Chinese | 20.1k GitHub stars | Classical Chinese syntax | Active open-source |
| **Qalb** | Arabic (Lebanese) | Academic/small | Arabic, RTL layout, Lisp-like | Research/art project |
| **Easy PL** | Chinese | Largest non-English PL community (Wikipedia) | Chinese keywords | RAD tool |
| **Rapira** | Russian | Historical (1980s Soviet) | Russian keywords | Educational, historical |
| **Citrine** | Multilingual | Small community | Translatable to any language | Niche |
| **ALGOL 68** | Multi-language | Historical (1968) | Had RU, DE, FR, BG, JA, ZH translations | Historical standard |

### 1C:Enterprise -- The Proof Point

1C:Enterprise deserves special attention as it proves non-English code works at scale:

- **5 million users** writing code with Russian keywords (per founder Boris Nuraliev)
- Used by **80-90% of Russian companies** for accounting and business automation (by
  number of installations; ~33% of market by revenue per IDC, 2014)
- Developers can write in Russian, English, or freely mix both
- Used by **1.5+ million organizations** worldwide
- Full IDE, debugging, and tooling support

This is not hypothetical. Production business software at massive scale already runs on
non-English code. Legesher brings this capability to **open-source Python**.

---

## 5. The Gap: No Multilingual Code Datasets Exist

Summarizing the landscape:

| What Exists | What's Missing |
| --- | --- |
| 3-6 TB of code with English keywords | Code with non-English keywords |
| ~40 languages in comments/docstrings | Keywords in ~40 languages |
| 1C proving non-English code works at scale | Open-source multilingual code |
| Aryabumi et al. proving English code improves English reasoning (8.2%) | Testing if multilingual code improves multilingual reasoning |
| Developers wanting native-language coding (99% Chinese comments) | Tools enabling it for Python |

**Legesher + The Stack = the first multilingual code datasets**, filling a gap that
no other project, dataset, or tool currently addresses.

### Related Work to Note

- **"Automated Python Translation"** (Otten et al., [arXiv:2504.11290](https://arxiv.org/abs/2504.11290),
  April 2025) -- Translates Python library API names (from PyTorch, pandas, TensorFlow,
  NumPy, random) into 7 languages, tested on French, Greek, and Bengali. This is the
  closest prior work, but focuses on **library function names** rather than core language
  keywords (`if`, `for`, `while`, `def`, `class`). It is unclear whether a public dataset
  was released.
- **"To Code or Not to Code"** (Aryabumi et al., Cohere for AI,
  [arXiv:2408.10914](https://arxiv.org/abs/2408.10914), 2024) -- Found up to **8.2%
  relative improvement** in natural language reasoning from including code in pre-training.
  Critically, this paper **tested only English reasoning** -- whether the same transfer
  occurs for non-English languages is exactly our research question.
- **"Bridging the Language Gap"** (IEEE, 2024) -- Enhancing multilingual prompt-based
  code generation via zero-shot cross-lingual transfer. Shows active research interest
  in this space, but does not create non-English code datasets.

---

## 6. How to Get More Data

### Querying The Stack Directly

```python
from datasets import load_dataset

# Stream Python files (no full download needed)
ds = load_dataset(
    "bigcode/the-stack-dedup",
    data_dir="data/python",
    split="train",
    streaming=True,
)

# Sample and analyze
for sample in ds.take(10000):
    content = sample["content"]
    # Extract comments, run language detection
    # See scripts/analyze_stack_languages.py for full implementation
```

### Language Detection Tools

- **langdetect** -- Python port of Nakatani Shuyo's language-detection library (Cybozu Labs,
  originally Java). Pure Python, no model download. Note: the Stanford paper used Google
  Language Detect (Google Translate API), which is a different tool with similar functionality.
- **fasttext** (`lid.176.bin`) -- Facebook's model. Faster, more accurate, but requires
  ~130MB download.
- **multilingual-e5-language-detection** -- HuggingFace model, 98.37% accuracy across
  45 languages.

### Related Datasets and Tools

- [bigcode/the-stack-metadata](https://huggingface.co/datasets/bigcode/the-stack-metadata) --
  lighter weight, pre-computed features
- [bigcode-project/bigcode-analysis](https://github.com/bigcode-project/bigcode-analysis) --
  comment extraction scripts used in the paper
- [bigcode/the-stack-github-issues](https://huggingface.co/datasets/bigcode/the-stack-github-issues) --
  31M issue/PR conversations, mostly English

### Our Analysis Script

See `scripts/analyze_stack_languages.py` in this directory -- streams Python files from
The Stack, extracts comments/docstrings, runs language detection, and outputs per-language
statistics. This will produce the first detailed per-language breakdown of The Stack's
Python subset.

---

## 7. Implications for Expedition Tiny Aya

### The Experiment

1. **Pull Python from The Stack** (50K--100K filtered files)
2. **Transpile via Legesher** into 3+ languages (Arabic, Hindi, Swahili proposed)
3. **LoRA fine-tune Tiny Aya** on 4 conditions: baseline, English code, multilingual code,
   multilingual text (control)
4. **Evaluate** on multilingual reasoning benchmarks (XNLI, XStoryCloze, TyDi QA, MMLU)

### Why This Matters

- **First-of-kind datasets** on HuggingFace -- no one has published multilingual code before
- Tests a fundamental question: is code's reasoning benefit **language-dependent** or
  **structure-dependent**?
- Regardless of outcome (positive, negative, or mixed), the result is publishable and valuable
- The datasets themselves become a resource for future multilingual AI research

### Supporting Data Points for the Proposal

- The majority of the world's ~28-47M developers (SlashData/Evans Data, 2025) are
  non-native English speakers
- 12.7% of GitHub users already write non-English commits
- 5M users prove non-English code works at enterprise scale (1C)
- The ~4% non-English content in The Stack is confined to comments -- no one has addressed
  code structure
- Every major code LLM (StarCoder, CodeLlama, etc.) trained on English-keyword code only

---

## Sources

### Papers

1. Kocetkov et al., "The Stack: 3 TB of permissively licensed source code"
   ([arXiv:2211.15533](https://arxiv.org/abs/2211.15533)), 2022
2. Lozhkov et al., "StarCoder 2 and The Stack v2: The Next Generation"
   ([arXiv:2402.19173](https://arxiv.org/abs/2402.19173)), 2024
3. Piech & Abu-El-Haija, "Human Languages in Source Code: Auto-Translation for
   Localized Instruction" ([arXiv:1909.04556](https://arxiv.org/abs/1909.04556)), 2019
   -- Also published at ACM L@S 2020: [DOI](https://dl.acm.org/doi/10.1145/3386527.3405916)
4. Aryabumi et al., "To Code or Not to Code? Exploring Impact of Code in Pre-training"
   ([arXiv:2408.10914](https://arxiv.org/abs/2408.10914)), Cohere for AI, 2024
5. Otten et al., "Automated Python Translation"
   ([arXiv:2504.11290](https://arxiv.org/abs/2504.11290)), 2025

### Datasets

6. [The Stack v1](https://huggingface.co/datasets/bigcode/the-stack) -- Hugging Face
7. [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2) -- Hugging Face
8. [The Stack GitHub Issues](https://huggingface.co/datasets/bigcode/the-stack-github-issues) --
   54 GB, 31M rows
9. [The Stack Metadata](https://huggingface.co/datasets/bigcode/the-stack-metadata) -- Hugging Face
10. [BigCode Analysis Repository](https://github.com/bigcode-project/bigcode-analysis) --
    comment extraction tools

### Non-English Programming Languages

11. [1C:Enterprise](https://1csoftware.com/) -- Russian-language enterprise platform;
    [Wikipedia](https://en.wikipedia.org/wiki/1C:Enterprise)
12. [Wenyan Lang](https://github.com/wenyan-lang/wenyan) -- Classical Chinese programming language
13. [Qalb](https://github.com/nasser/---) -- Arabic programming language
14. [Easy Programming Language](https://en.wikipedia.org/wiki/Easy_Programming_Language) -- Chinese RAD tool
15. [Non-English-based programming languages](https://en.wikipedia.org/wiki/Non-English-based_programming_languages)
    -- Wikipedia

### Developer Population

16. [SlashData Global Developer Population 2025](https://www.slashdata.co/post/global-developer-population-trends-2025-how-many-developers-are-there)
    -- 47.2M developers worldwide
17. [Evans Data Corporation Developer Population 2025](https://evansdata.com/reports/viewRelease.php?reportID=9)
    -- 28.7M developers worldwide
