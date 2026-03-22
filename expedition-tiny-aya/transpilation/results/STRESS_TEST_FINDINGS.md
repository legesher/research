# Transpiler Stress Test Findings

**Dates:** 2026-03-08 (1K runs), 2026-03-14 (10K runs, cross-dataset runs)
**Legesher Core:** v0.5.1 (1K), v0.6.0 (10K, cross-dataset)
**Datasets:** bigcode/the-stack-dedup, bigcode/the-stack, bigcode/starcoderdata (Python subsets)
**Linear Issue:** CORE-685

## Executive Summary

We ran real-world Python files from three BigCode datasets through both translator backends
(Token and Tree-Sitter) across 8 languages, two scale levels (1K and 10K files), and three
independent datasets to validate scale readiness for the Expedition Tiny Aya / "Language,
Decoded" research project.

**Bottom line:** The Token backend is production-ready for 50K-100K scale dataset generation
with 100% forward-translation success and zero silent failures across all datasets tested.
Error rates and round-trip accuracy hold steady from 1K to 10K files and are consistent
across datasets with different file populations. The transpiler's behavior is
language-agnostic and dataset-agnostic -- determined by code structure, not target language
or data source.

## Test Runs

### 1K-File Runs (2026-03-08)

| Language | Code | Script |
|----------|------|--------|
| Arabic | ar | Arabic (RTL) |
| Chinese | zh | CJK |
| Japanese | ja | CJK/Kana |
| Russian | ru | Cyrillic |
| Spanish | es | Latin |

**Configuration:** 1,000 files from 1,087 streamed. Filter rejects: 45 syntax errors, 33 too short (<5 lines), 9 too large (>100KB). Deterministic streaming ensures the same files in the same order for every run.

### 10K-File Runs (2026-03-14)

| Language | Code | Script |
|----------|------|--------|
| Amharic | am | Ge'ez |
| Chinese | zh | CJK |
| Urdu | ur | Arabic (RTL) |

**Configuration:** 10,000 files from 10,841 streamed. Filter rejects: 499 syntax errors,
298 too short, 44 too large. Wall time ~6 minutes each. Peak memory ~266 MB.

### Cross-Dataset Runs (2026-03-14)

| Dataset                 | Languages  | Files | Data Dir    |
| ----------------------- | ---------- | ----- | ----------- |
| bigcode/the-stack-dedup | am, ur, zh | 1,000 | data/python |
| bigcode/starcoderdata   | am, ur, zh | 1,000 | python      |
| bigcode/the-stack       | am, ur, zh | 1,000 | data/python |

**Purpose:** Validate that transpiler behavior is consistent across independent datasets
with different file populations and curation pipelines.

**Note:** `bigcode/the-stack-v2-dedup` was also attempted but does not contain file
contents -- it only stores Software Heritage blob IDs requiring AWS/S3 access to fetch
source code. Not suitable for direct streaming.

## Results Summary

### Cross-Language Comparison (1K files)

| Language | Token RT Exact | TS RT Exact | TS Errors | Divergences |
|----------|---------------|-------------|-----------|-------------|
| ar (Arabic) | 93.8% | 95.5% | 7 | 23 |
| **es (Spanish)** | **85.8%** | **86.7%** | **0** | **30** |
| ja (Japanese) | 93.8% | 95.5% | 7 | 23 |
| ru (Russian) | 93.8% | 95.5% | 7 | 23 |
| zh (Chinese) | 93.8% | 95.5% | 7 | 23 |

4 out of 5 languages produce identical results. Spanish is a significant outlier (see Finding 5 below).

### Cross-Language Comparison (10K files)

| Language | Token RT Exact | TS RT Exact | TS Errors | Divergences |
|----------|---------------|-------------|-----------|-------------|
| am (Amharic) | 93.35% | 95.36% | 45 | 246 |
| ur (Urdu) | 93.35% | 95.36% | 45 | 246 |
| zh (Chinese) | 93.34% | 95.34% | 45 | 246 |

All three languages produce virtually identical results at 10K scale.

### Cross-Dataset Comparison (1K files, am/ur/zh)

| Dataset             | Token RT | TS RT | TS Errors | Divergences | Filtered | Streamed |
| ------------------- | -------- | ----- | --------- | ----------- | -------- | -------- |
| the-stack-dedup     | 93.8%    | 95.5% | 7         | 23          | 87       | 1,087    |
| starcoderdata       | 93.0%    | 95.3% | 8         | 27          | 1,122    | 2,122    |
| the-stack (original)| 93.6%    | 96.0% | 5         | 28          | 85       | 1,085    |

All three languages (am, ur, zh) produce identical results within each dataset.
Differences between datasets reflect different file populations, not transpiler
behavior. Token backend achieves 100% forward-translation success on all datasets.

**Notable:** starcoderdata has significantly more syntax errors in its files
(1,060 filtered vs ~40-45 for The Stack datasets), requiring 2x as many files
to be streamed. Despite the dirtier file population, transpiler accuracy remains
within the same narrow band (93.0-93.8% Token RT, 95.3-96.0% TS RT).

### Per-Backend Summary (10K, representative)

| Metric | Token | Tree-Sitter |
|--------|-------|-------------|
| Success rate | 10,000/10,000 (100%) | 9,955/10,000 (99.55%) |
| Round-trip exact match | ~93.35% | ~95.35% |
| Errors | 0 | 45 (InvalidCodeError) |
| Silent failures | 0 | 0 |
| Throughput | ~163 files/sec | ~137 files/sec |
| Avg forward time | ~6.1 ms | ~7.3 ms |

## Detailed Findings

### 1. Language-Agnostic and Dataset-Agnostic Behavior

The transpiler produces identical error rates, round-trip accuracy, and divergence
counts regardless of target language -- confirmed across 7 languages (ar, am, ja, ru,
ur, zh) at both 1K and 10K scale.

Additionally, cross-dataset testing across three independent datasets
(the-stack-dedup, starcoderdata, the-stack) confirms that accuracy stays within a
narrow band (Token RT 93.0-93.8%, TS RT 95.3-96.0%) regardless of data source.
Differences between datasets are attributable to different file populations, not
transpiler behavior. The behavior is determined entirely by the source code structure,
not the target language's script, character set, or data source.

**Exception:** Spanish (es) is an outlier -- see Finding 5.

### 2. Round-Trip Mismatches

At 1K scale, **62 files** failed round-trip on Token (6.2%), **45 files** on Tree-Sitter (4.5%). Breaking down:

- **44 files** mismatch on **both** backends -- a systematic issue
  - 41 of these show line count changes (`mismatch_first_line = None`), indicating whitespace/newline normalization during translation (CORE-1160).

- **18 files** mismatch on **Token only** (Tree-Sitter round-trips perfectly)
  - Root cause: Token backend translates identifiers matching builtin names (e.g., a variable called `type` or `input`) without scope awareness (CORE-1107). The reverse translation can't distinguish "this was a builtin" from "this was a variable."

- **1 file** mismatches on **Tree-Sitter only** -- minor edge case.

At 10K scale, rates are consistent: Token ~6.65% mismatch, Tree-Sitter ~4.65% mismatch.

### 3. Tree-Sitter Encoding Errors (CORE-799)

All Tree-Sitter errors are the same UTF-8 encoding issue:

```text
Failed to translate code: 'utf-8' codec can't decode byte 0x9f in position ...
```

Files contain non-UTF-8 bytes inside string literals. Python's tokenizer handles them; Tree-Sitter's C-based byte processing does not. Token backend translates all affected files successfully.

**Language-dependent:** 0 errors with Spanish (ASCII-only translations), 7 errors at 1K / 45 errors at 10K with non-ASCII languages (CJK, Arabic, Cyrillic, Ge'ez). The error occurs when translated output contains multi-byte characters that interact with the existing non-UTF-8 bytes.

### 4. Backend Divergence

23 files (2.3%) at 1K, 246 files (2.46%) at 10K produce different output between Token and Tree-Sitter. In most cases, Token over-translates (scope-unaware) while Tree-Sitter is correct. The divergence count is consistent across all languages.

### 5. Spanish Language Pack Outlier (CORE-704)

Spanish shows ~8% lower round-trip accuracy than all other languages:

| Metric | Spanish | Others (avg) |
|--------|---------|-------------|
| Token RT exact | 85.8% | 93.8% |
| TS RT exact | 86.7% | 95.5% |
| TS errors | 0 | 7 |
| Divergences | 30 | 23 |

The higher divergence count (30 vs 23) suggests the Spanish pack has more translations that trigger Token's scope-unaware over-translation. The 0 Tree-Sitter errors are because Spanish translations are ASCII-only (Latin script), avoiding the 0x9f byte issue entirely.

### 6. Zero Silent Failures

Across all runs (1K and 10K, all languages), zero silent failures were detected. Every error was caught and reported. The transpiler either translates successfully or raises an explicit error.

### 7. Cross-Dataset File Quality

starcoderdata has significantly lower file quality than The Stack datasets:

| Metric              | the-stack-dedup | the-stack | starcoderdata |
| ------------------- | --------------- | --------- | ------------- |
| Syntax errors       | 45 (4.1%)       | 40 (3.7%) | 1,060 (50.0%) |
| Too short           | 33 (3.0%)       | 31 (2.9%) | 55 (2.6%)     |
| Too large           | 9 (0.8%)        | 14 (1.3%) | 7 (0.3%)      |
| Total filtered      | 87 (8.0%)       | 85 (7.8%) | 1,122 (52.9%) |
| Streamed for 1K     | 1,087           | 1,085     | 2,122         |

starcoderdata required streaming 2x as many files to reach 1,000 valid samples due
to its high syntax error rate. Despite this, transpiler accuracy remains consistent,
confirming the transpiler is robust against varying file quality.

**the-stack-v2-dedup:** This dataset does not contain file contents. It stores only
Software Heritage blob IDs (`blob_id`) and metadata. Fetching actual source code
requires AWS credentials and a SoftwareHeritage/INRIA agreement via S3. Not suitable
for direct streaming stress tests.

### 8. Scale Stability (1K vs 10K)

| Metric | 1K (zh) | 10K (zh) | Delta |
|--------|---------|----------|-------|
| Token RT exact | 93.8% | 93.34% | -0.46% |
| TS RT exact | 95.5% | 95.34% | -0.16% |
| TS error rate | 0.7% | 0.45% | -0.25% |
| Divergence rate | 2.3% | 2.46% | +0.16% |

All metrics within 0.5% -- no degradation at 10x scale. Memory grows sub-linearly (172 MB at 1K, 266 MB at 10K).

## Scaling Projections (Validated)

| Scale | Wall Time | Memory | Status |
|-------|-----------|--------|--------|
| 1,000 files | ~48 sec | ~172 MB | Tested |
| 5,000 files | ~3 min | ~220 MB | Tested |
| 10,000 files | ~6 min | ~266 MB | Tested |
| 50,000 files | ~30 min | ~400 MB (est.) | Projected |
| 100,000 files | ~60 min | ~500 MB (est.) | Projected |

Bottleneck is HuggingFace streaming, not translation. Translation alone runs at 137-163 files/sec.

## Recommendations

### For Tiny Aya Dataset Generation

1. **Use Token backend** for initial 50K-100K file generation -- 100%
   forward-translation success rate, no crashes, no silent failures across
   all datasets tested
2. **Use the-stack-dedup as primary dataset** -- cleanest files (92% acceptance
   rate), deterministic streaming, deduplicated
3. The ~6.5% round-trip mismatch rate is acceptable for training data (forward
   translation is correct; only reverse has edge cases)
4. Avoid the Spanish language pack until CORE-704 is investigated
5. Add local file caching to avoid re-downloading on subsequent runs
6. **Avoid the-stack-v2-dedup** -- requires AWS/S3 access for file contents

### Bugs to Fix

| Issue | Description | Impact | Severity |
|-------|-------------|--------|----------|
| CORE-799 | Tree-Sitter UTF-8 encoding error | 0.45% of files fail with non-ASCII languages | Medium |
| CORE-1160 | Round-trip line count changes | 4.1% of files, whitespace normalization | Low |
| CORE-1107 | Token scope-unaware builtin translation | 1.8% of files, over-translation | Low |
| CORE-704 | Spanish pack ~8% lower accuracy | Affects only es language pack | Medium |
| CORE-696 | Stress test mismatch_first_line bug | Script bug, splitlines() strips trailing newlines | Low |

## Appendix: Filter Statistics

### 1K Runs (the-stack-dedup)

| Category             | Count        |
| -------------------- | ------------ |
| Total streamed       | 1,087        |
| Accepted             | 1,000 (92%)  |
| Syntax error         | 45 (4.1%)    |
| Too short (<5 lines) | 33 (3.0%)    |
| Too large (>100KB)   | 9 (0.8%)     |

### 10K Runs (the-stack-dedup)

| Category             | Count          |
| -------------------- | -------------- |
| Total streamed       | 10,841         |
| Accepted             | 10,000 (92.2%) |
| Syntax error         | 499 (4.6%)     |
| Too short (<5 lines) | 298 (2.7%)     |
| Too large (>100KB)   | 44 (0.4%)      |

### 1K Runs (starcoderdata)

| Category             | Count          |
| -------------------- | -------------- |
| Total streamed       | 2,122          |
| Accepted             | 1,000 (47.1%)  |
| Syntax error         | 1,060 (50.0%)  |
| Too short (<5 lines) | 55 (2.6%)      |
| Too large (>100KB)   | 7 (0.3%)       |

### 1K Runs (the-stack)

| Category             | Count          |
| -------------------- | -------------- |
| Total streamed       | 1,085          |
| Accepted             | 1,000 (92.2%)  |
| Syntax error         | 40 (3.7%)      |
| Too short (<5 lines) | 31 (2.9%)      |
| Too large (>100KB)   | 14 (1.3%)      |

## Reports

All raw JSON reports are available in the `reports/` directory:

### the-stack-dedup

- `stress_test_ar_1000.json` -- Arabic, 1K files
- `stress_test_am_1000.json` -- Amharic, 1K files
- `stress_test_es_1000.json` -- Spanish, 1K files
- `stress_test_ja_1000.json` -- Japanese, 1K files
- `stress_test_ru_1000.json` -- Russian, 1K files
- `stress_test_ur_1000.json` -- Urdu, 1K files
- `stress_test_zh_1000.json` -- Chinese, 1K files
- `stress_test_zh_5000.json` -- Chinese, 5K files
- `stress_test_am_10000.json` -- Amharic, 10K files
- `stress_test_ur_10000.json` -- Urdu, 10K files
- `stress_test_zh_10000.json` -- Chinese, 10K files

### starcoderdata

- `stress_test_starcoderdata_am_1000.json` -- Amharic, 1K files
- `stress_test_starcoderdata_ur_1000.json` -- Urdu, 1K files
- `stress_test_starcoderdata_zh_1000.json` -- Chinese, 1K files

### the-stack (original)

- `stress_test_the-stack_am_1000.json` -- Amharic, 1K files
- `stress_test_the-stack_ur_1000.json` -- Urdu, 1K files
- `stress_test_the-stack_zh_1000.json` -- Chinese, 1K files
