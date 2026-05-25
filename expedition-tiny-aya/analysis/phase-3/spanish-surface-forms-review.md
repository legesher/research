# Spanish SIB-200 Surface Forms — Native-Speaker Review

**Purpose:** Our evaluation model, when prompted in Spanish, answers the
SIB-200 topic-classification task in Spanish. Our scoring code currently only
recognizes English category words, so it discards every Spanish answer as
"unparseable." We want to fix that — but only by adding words the model
_actually emitted_, mapped to the _correct_ category.

**What we need from you:** For each word/phrase below, confirm (a) our reading
of what it means, and (b) which of the 7 valid categories it belongs to — or
tell us it belongs to none of them.

**Data source:** Every form below was emitted by the model in the Phase-3
baseline evaluation, in cells where the dataset and/or instructions were
Spanish (across 2 prompt templates). Counts are approximate totals.

**Note on capitalization:** the scorer lowercases everything before matching,
so `Política` and `política` are treated identically — you only need to
confirm the lowercase form.

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

| #   | Spanish         | Our literal reading        | → Category        | Approx. count | Correct? |
| --- | --------------- | -------------------------- | ----------------- | ------------- | -------- |
| A1  | viajes          | "trips / travels"          | **travel**        | ~85           | ☐        |
| A2  | viaje           | "trip / travel" (singular) | **travel**        | ~5            | ☐        |
| A3  | política        | "politics"                 | **politics**      | ~25           | ☐        |
| A4  | deportes        | "sports"                   | **sports**        | ~19           | ☐        |
| A5  | salud           | "health"                   | **health**        | ~16           | ☐        |
| A6  | entretenimiento | "entertainment"            | **entertainment** | ~6            | ☐        |

---

## Section B — Science/technology

We recognize the English word "technology" but not the Spanish spelling.
**Confirm this means science/technology.**

| #   | Spanish    | Our literal reading | Means science/technology? |
| --- | ---------- | ------------------- | ------------------------- |
| B1  | tecnología | "technology"        | ☐                         |

---

## Section C — Compound answer (two words, same category?)

The model sometimes joins two words. We need to know: **do both halves mean
the SAME category?** If yes, we score it; if they're different categories,
we keep it "unparseable."

| #   | Spanish           | Our literal reading    | Half 1 | Half 2    | Same category?                   |
| --- | ----------------- | ---------------------- | ------ | --------- | -------------------------------- |
| C1  | salud y bienestar | "health and wellbeing" | health | wellbeing | ☐ same (both health) ☐ different |

---

## Section D — Words that may not be a SIB-200 category at all

We believe this doesn't cleanly fit any of the 7 categories. We plan to keep
it marked "unparseable." **Confirm it is NOT one of the 7 categories** — or
tell us which one it belongs to.

| #   | Spanish    | Our literal reading | Is it one of the 7?                                                 |
| --- | ---------- | ------------------- | ------------------------------------------------------------------- |
| D1  | naturaleza | "nature"            | ☐ no ☐ yes: **\_\_** (we considered _geography_ but it's a stretch) |

---

## Summary of what we'll do with the answers

- **Section A, B** confirmed → added to the scoring code so these Spanish
  answers are counted correctly.
- **Section C** — if "same" → scored as health; if "different" → kept
  "unparseable."
- **Section D** confirmed "not a category" → stays "unparseable."

Every word added is one the model genuinely produced — we are not inventing
translations, only teaching the scorer to read what the model already said.

---

## A note on Spanish XNLI (separate from the above)

Spanish is the _easiest_ language for our scorer overall — the model mostly
answers SIB-200 in clean single Spanish words. For the XNLI benchmark
(entailment / contradiction / neutral), the Spanish forms `contradicción`,
`implicación`, `neutro` are **already recognized** by our code. No Spanish
XNLI changes needed — this review is SIB-200 only.

---

_Reviewer name(s): ********\_\_\_******** Date: ****\_\_\_****_

_Notes / anything we missed:_
