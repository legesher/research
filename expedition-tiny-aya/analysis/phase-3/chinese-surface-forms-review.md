# Chinese SIB-200 Surface Forms — Native-Speaker Review

**Purpose:** Our evaluation model, when prompted in Chinese, answers the
SIB-200 topic-classification task in Chinese. Our scoring code currently only
recognizes English category words, so it discards every Chinese answer as
"unparseable." We want to fix that — but only by adding words the model
_actually emitted_, mapped to the _correct_ category.

**What we need from you:** For each word/phrase below, confirm (a) our reading
of what it means, and (b) which of the 7 valid categories it belongs to — or
tell us it belongs to none of them.

**Data source:** Every form below was emitted by the model in the Phase-3
baseline evaluation, in the cells where the dataset and/or instructions were
Chinese (across 2 prompt templates). Counts are approximate totals.

---

## The 7 valid SIB-200 categories

A correct answer must be exactly one of these:

1. **science/technology**
2. **travel**
3. **politics**
4. **sports**
5. **health**
6. **entertainment**
7. **geography**

If a word doesn't clearly mean one of these 7, it should stay marked
"unparseable" (it means the model answered with a category that doesn't
exist — a real model error we want to keep visible, not hide).

---

## Section A — Single-word answers (high confidence)

These appear on their own as the model's whole answer. We believe each maps
cleanly to one category. **Please confirm the meaning and category.**

| #   | Chinese | Pinyin   | Our literal reading           | → Category        | Approx. count | Correct? |
| --- | ------- | -------- | ----------------------------- | ----------------- | ------------- | -------- |
| A1  | 旅行    | lǚxíng   | "travel / a trip"             | **travel**        | ~100          | ☐        |
| A2  | 旅游    | lǚyóu    | "tourism / travel"            | **travel**        | ~5            | ☐        |
| A3  | 政治    | zhèngzhì | "politics"                    | **politics**      | ~30           | ☐        |
| A4  | 娱乐    | yúlè     | "entertainment"               | **entertainment** | ~11           | ☐        |
| A5  | 体育    | tǐyù     | "sports / physical education" | **sports**        | ~6            | ☐        |
| A6  | 地理    | dìlǐ     | "geography"                   | **geography**     | ~2            | ☐        |

**Key question for A1 vs A2:** we are mapping _both_ 旅行 and 旅游 to **travel**
(SIB-200 has no separate "tourism" category). Is that right?

---

## Section B — Science/technology, alternative spellings

We already recognize `科学/技术` ("science/technology" joined with a slash).
The model also writes it joined with a _word_ for "and" instead of a slash —
which our code currently misses. **Please confirm these mean "science and
technology".**

| #   | Chinese    | Pinyin         | Our literal reading                                  | Approx. count | Means science/technology? |
| --- | ---------- | -------------- | ---------------------------------------------------- | ------------- | ------------------------- |
| B1  | 科学和技术 | kēxué hé jìshù | "science **and** technology" (和 = and)              | ~1            | ☐                         |
| B2  | 科学与技术 | kēxué yǔ jìshù | "science **and** technology" (与 = and, more formal) | ~6            | ☐                         |

---

## Section C — Possible science/technology sub-topics

SIB-200's "science/technology" category covers sub-topics including
transportation. The model sometimes answers with a sub-topic. **Confirm
whether this should count as science/technology.**

| #   | Chinese  | Pinyin            | Our literal reading     | → science/technology?                |
| --- | -------- | ----------------- | ----------------------- | ------------------------------------ |
| C1  | 公共交通 | gōnggòng jiāotōng | "public transportation" | ☐ yes ☐ no — it's actually: **\_\_** |

(Note: "transportation" is one of SIB-200's science/technology sub-topics,
so we lean toward yes — but "public transport" could arguably be travel.
Your call.)

---

## Section D — Compound answers (the model names TWO things)

The model sometimes answers with two words joined by "/". We need to know:
**do both halves mean the SAME category, or DIFFERENT categories?**

- Both halves same category → we can safely score it.
- Different categories → the model hedged / couldn't decide → we keep it
  "unparseable" (a real model limitation worth showing in the paper).

| #   | Chinese   | Pinyin            | Half 1 means… | Half 2 means…                       | Same category?     |
| --- | --------- | ----------------- | ------------- | ----------------------------------- | ------------------ |
| D1  | 摄影/技术 | shèyǐng / jìshù   | photography   | technology                          | ☐ same ☐ different |
| D2  | 摄影/摄影 | shèyǐng / shèyǐng | photography   | photography (model repeated itself) | ☐ same ☐ different |

---

## Section E — Words that may not be a SIB-200 category at all

We believe these don't cleanly fit any of the 7 categories. We plan to keep
them marked "unparseable." **Confirm each one is NOT one of the 7 categories**
— or tell us which category it belongs to.

| #   | Chinese | Pinyin  | Our literal reading        | Is it one of the 7?                                                                  |
| --- | ------- | ------- | -------------------------- | ------------------------------------------------------------------------------------ |
| E1  | 戏剧    | xìjù    | "drama / theatre"          | ☐ no ☐ yes: **\_\_** (we suspect _entertainment_)                                    |
| E2  | 赌场    | dǔchǎng | "casino"                   | ☐ no ☐ yes: **\_\_** (we suspect _entertainment_)                                    |
| E3  | 摄影    | shèyǐng | "photography"              | ☐ no ☐ yes: **\_\_**                                                                 |
| E4  | 狮群    | shīqún  | "a pride / group of lions" | ☐ no ☐ yes: **\_\_** (we think this is the model echoing the passage, not answering) |

---

## Summary of what we'll do with the answers

- **Section A, B, C** confirmed → added to the scoring code so these Chinese
  answers are counted correctly.
- **Section D** — entries marked "same" → scored; entries marked "different"
  → kept "unparseable" (model hedged).
- **Section E** confirmed "not a category" → stay "unparseable." If you tell
  us one _is_ a category (e.g. 戏剧 → entertainment), we'll add it.

Every word added is one the model genuinely produced — we are not inventing
translations, only teaching the scorer to read what the model already said.

---

_Reviewer name(s): ********\_\_\_******** Date: ****\_\_\_****_

_Notes / anything we missed:_
