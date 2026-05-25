# Refined Decision Ledger — Phase-3 SIB-200 / XNLI / X-CSQA / Belebele

**Purpose (for the paper):** a complete, auditable record of every decision
we made when re-scoring the Phase-3 evaluation outputs. For each benchmark
this document states: the approved answer set, every _kind_ of answer the
model actually produced, and — for each — whether we **counted it as the
model's answer** or **left it as a parse-failure**, with the rationale.

A reader should be able to finish this document and say: "I see exactly what
they accepted, what they rejected, and why — and I can disagree with any
single line."

**Method:** every surface form below was observed in the Phase-3 baseline
evaluation outputs (`legesher/language-decoded-experiments`, 32 cells across
2 prompt templates). We re-score by re-running an extractor over the stored
`raw_output` text — no inference is re-run. We never invent a form; we only
teach the scorer to read forms the model genuinely emitted.

**Status:** `CONFIRMED` = decided. `PENDING REVIEW` = awaiting native-speaker
verification (see `urdu-/chinese-/spanish-surface-forms-review.md`).

---

## Scope finding — where the problem actually is

Re-parsing all 96 non-SIB-200 cells plus the 32 SIB-200 cells shows the
"can't read the model's answer" problem is **overwhelmingly a SIB-200
problem**:

| Benchmark    | Cells with >10% parse-failure          | Verdict                                         |
| ------------ | -------------------------------------- | ----------------------------------------------- |
| **SIB-200**  | most cells where instruction ≠ English | Needs the native-category extension             |
| **XNLI**     | 8 cells — all `template2`              | Needs ONE targeted fix; rest is model behaviour |
| **X-CSQA**   | 0 of 32                                | No change needed                                |
| **Belebele** | 0 of 32                                | No change needed                                |

X-CSQA and Belebele answer with letters (A–E), which are language-neutral —
the scorer already reads them. SIB-200 and (partly) XNLI are where the work is.

---

> **Review status (2026-05-23):** the three native-speaker review docs (Urdu / Chinese / Spanish) have been returned and the proposed mappings confirmed as-is. The two reviewer-decided uncertain entries — `公共交通` → travel, `علم` → science/technology — are applied. All `PENDING REVIEW` rows below have been moved to `CONFIRMED`.

## 1. SIB-200 — topic classification

### Approved answer set (7 categories)

`science/technology` · `travel` · `politics` · `sports` · `health` ·
`entertainment` · `geography`

### ACCEPTED — counted as the model's answer

| Rule                                   | Example raw_output                                                                                                          | Scored as             | Status             | Rationale                                                                                                                 |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | --------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| Exact                                  | `travel`                                                                                                                    | travel                | CONFIRMED          | Verbatim canonical label.                                                                                                 |
| Normalized                             | `"Travel."`, `Science/Technology`                                                                                           | travel / science-tech | CONFIRMED          | Case-fold + quote/punctuation stripping only. No semantic change.                                                         |
| First-line                             | `travel⏎This passage is about…`                                                                                             | travel                | CONFIRMED          | Model answered on line 1, then explained. We read line 1, as the eval always has.                                         |
| Substring                              | `The answer is travel`                                                                                                      | travel                | CONFIRMED          | Canonical token present in a longer first line.                                                                           |
| Alias                                  | `sport` → sports                                                                                                            | sports                | CONFIRMED          | Known English synonym.                                                                                                    |
| Rule A — `science/<X>`                 | `science/AI`, `science/physics`                                                                                             | science/technology    | CONFIRMED (PR #49) | Model invents science sub-topics; all are the science/tech category.                                                      |
| Rule B — native science/tech           | `سائنس/ٹکنالوجی`, `科学/技术`, `ciencia/tecnología`                                                                         | science/technology    | CONFIRMED (PR #49) | Native-script renderings of "science/technology".                                                                         |
| Rule C — bare sub-topic                | `physics`, `chemistry`, `ai`                                                                                                | science/technology    | CONFIRMED (PR #49) | Template-2 strips the `science/` prefix; the bare sub-topic remains.                                                      |
| **Native category (NEW)**              | `سیاست`→politics · `旅行`→travel · `viajes`→travel · `صحت`→health · `娱乐`→entertainment · `体育`→sports · `地理`→geography | per-form              | **CONFIRMED** | Native-language category words. Single-word, exact first-line match only. See the three `*-surface-forms-review.md` docs. |
| **Native science/tech variants (NEW)** | `سائنس/ٹیکنالوجی` (~88×), `科学和技术`, `科学与技术`, `tecnología`                                                          | science/technology    | **CONFIRMED** | Spelling variants of science/technology that current Rule B misses.                                                       |
| **Arabic forms (NEW)**                 | `الرياضة`→sports · `السياسة`→politics                                                                                       | per-form              | **CONFIRMED** | The Urdu-prompted model code-switches to Arabic. Observed, so eligible.                                                   |

### REJECTED — left as a parse-failure (NOT counted)

| Pattern                 | Example raw_output                                                         | Status           | Rationale                                                                                                                                                                                             |
| ----------------------- | -------------------------------------------------------------------------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Cross-category compound | `کھیل/تکنالوجی` (sport/technology), `سیاحت/تفریح` (tourism/entertainment)  | CONFIRMED        | The model named **two different categories** — it hedged. Resolving it to either one manufactures a decision the model never made. We keep it a parse-failure: a real model limitation, kept visible. |
| Off-scheme answer       | `تعلیم` (education), `naturaleza` (nature), `قانون` (law), `戏剧` (drama)? | CONFIRMED        | The model answered with a category that is **not one of the 7**. No rule can fix a wrong-ontology answer; mapping it would be invention. Kept as a parse-failure (a genuine model error).             |
| Passage echo            | `狮群` ("a pride of lions")                                                | CONFIRMED reject | Not a category at all — the model echoed passage content.                                                                                                                                             |
| Empty output            | ``                                                                         | CONFIRMED reject | The model produced nothing.                                                                                                                                                                           |

### Known model-behaviour finding (NOT a scoring decision)

In several Urdu cells the model emits a near-constant `سائنس/ٹیکنالوجی`
("science/technology") regardless of the passage topic. The _identical_
string lands in both the `correct` bucket (when gold happens to be
science/technology) and the `wrong_label` bucket (when it isn't). Those
"correct" rows are not genuine successes — they are a degenerate constant
output landing on gold by chance. **This is a model limitation, not a
scoring decision**, but it must be reported: a raw SIB-200 accuracy for
those cells overstates the model. (See the `correct_ambiguous` flag planned
for the per-row analysis.)

### Paper note — truncation limitation

SIB-200 answers are generated with `max_new_tokens=10`. Long native-script
compounds get cut mid-word — e.g. `کھیل/سائنس/تکنالوجی` is truncated to
`کھیل/سائنس/تکنال`, where `تکنالوجی` (technology) loses its tail and no
longer matches the term map. This is a limitation of the **generation
budget**, not of the extractor: the extractor sees a truncated string and
correctly handles it (the surviving pieces `کھیل`+`سائنس` still trip the
multi-category hedge → parse-failure). Worth a sentence in the paper's
methodology limitations: a tighter token budget can clip multi-word
non-English answers, so some parse-failures reflect truncation rather than
model behaviour or extractor coverage.

### Reviewer decisions on uncertain term-map entries

Two SIB-200 native-term map entries that the surface-form review docs
flagged as uncertain have been decided:

- **`公共交通`** ("public transportation", Chinese) → **`travel`**.
  Provisionally mapped to `science/technology` (transportation is a
  science/tech sub-topic); reviewer decision is that public transport reads
  closer to travel. Map entry changed accordingly.
- **`علم`** ("knowledge", Urdu) → **`science/technology`** — confirmed kept.
  Appears in `علم و ٹیکنالوجی` ("knowledge and technology"); reviewer
  confirms it is a legitimate science/technology rendering.

---

## 2. XNLI — natural language inference

### Approved answer set (3 labels)

`entailment` · `contradiction` · `neutral`

XNLI re-scoring is organized into three **tiers** by how directly the model's
output names a label. Tier 1 is verbatim; Tier 2 embeds the literal English
label inside a native-language frame; Tier 3 describes the relationship in
native prose without ever writing a label word.

### Tier 1 — ACCEPTED: verbatim / native label words

| Rule                              | Example raw_output                                                                                                      | Scored as | Status              | Rationale                                                                         |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | --------- | ------------------- | --------------------------------------------------------------------------------- |
| Exact English                     | `entailment`, `contradiction`, `neutral`                                                                                | the label | CONFIRMED           | Verbatim label. The model answers XNLI in English the large majority of the time. |
| English substring (+ punctuation) | `Neutral.`, `contradiction.`                                                                                            | the label | CONFIRMED           | Trailing punctuation / casing only.                                               |
| Native label                      | `矛盾`→contradiction · `Contradicción`→contradiction · `لازمی`→entailment · `لازم آتی ہے`→entailment · `neutro`→neutral | the label | CONFIRMED (Phase-2) | Native zh/es/ur label words, in the map since the Phase-2 XNLI re-scoring.        |

### Tier 2 — ACCEPTED (positive frames only): English label glued to a CJK frame

The model sometimes lifts the literal English label word out of the prompt's
label list and drops it into a Chinese sentence frame: `假设是entailment。`.
The word-boundary regex `\bentailment\b` fails between a CJK character and a
Latin letter, so these are currently scored as parse-failures even though the
model wrote the label explicitly.

**Empirical finding (baseline, t1+t2 pooled, all 15 CJK-glued forms with
count ≥ 10):** the glued label word is **always `entailment`** — never
`contradiction`, never `neutral`, never a Chinese label word. The model lifts
the English token from the prompt; the Chinese is only sentence scaffolding.
This is consistent with the model's overall `entailment` bias.

The 15 forms split by frame polarity:

| Frame polarity                      | Example                                                             | Chinese frame literally                                                        | Decision                   |
| ----------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------ | -------------------------- |
| **Positive** (12 forms, ≈ 500 rows) | `假设是entailment。` · `假设和前提之间存在entailment。`             | "the hypothesis **is** entailment" · "there **exists** entailment between…"    | **ACCEPT → entailment**    |
| **Negated** (3 forms, ≈ 64 rows)    | `假设和前提之间没有entailment或` · `前提和假设之间没有entailment、` | "there **is no** entailment between…" (trailing `或`/`、` = "or" / list-comma) | **REJECT → parse-failure** |

**The 64 negated rows are a deliberate REJECT** — `没有entailment` ("there is
_no_ entailment") is the model negating the label, not asserting it. The
trailing "or" / list-comma suggests the model is listing the options it rules
out. We cannot faithfully extract a single label from a negation, so these
stay parse-failures. **(Paper note: flag these 64 rows explicitly — they are a
small but real caveat in the Tier-2 rule, and an honest methodology section
should mention the negated-frame exclusion.)**

Status: **CONFIRMED** (positive frames accept, negated frames reject).

### Tier 3 — ACCEPTED: native-prose paraphrase of the relationship

Under `template2`, the model frequently _describes_ the NLI relationship in
native prose without ever writing a label word. The description is precise
enough to imply the label unambiguously:

| Paraphrase (example)                              | Count   | Literal meaning                                         | → Label       |
| ------------------------------------------------- | ------- | ------------------------------------------------------- | ------------- |
| `假设是前提的直接结果。`                          | 1,476   | "the hypothesis is the direct result of the premise"    | entailment    |
| `假设是前提的推论。`                              | 549     | "…is an inference of the premise"                       | entailment    |
| `假设是前提的否定。`                              | 624     | "…is the negation of the premise"                       | contradiction |
| `premise اور hypothesis کے درمیان کوئی تعلق نہیں` | ≈ 1,150 | "no relationship between premise and hypothesis" (Urdu) | neutral       |

Status: **CONFIRMED** — these mappings are an _interpretation_ of native
prose, not a dictionary lookup, so each paraphrase pattern goes to a native
speaker for confirmation (same process as the SIB-200 native forms). Verb-form
(`entails` vs `entailment`) and negation edge cases handled the same faithful
way as Tier 2. **Paper framing:** Tier 3 is explicitly a _lenient, semantic_
mapping of a model that described-instead-of-labelled — reported as such, with
the strict-vs-lenient gap standing as a measure of instruction-following.

#### Negation-guard limitation (verified zero impact)

Tier 3's entailment/contradiction patterns (`直接结果`, `推论`, `等同`, `否定`)
are bare phrase matches — unlike the neutral patterns, which bake in `没有`,
they have no guard against a preceding negation. A hypothetical
`假设不是前提的直接结果` ("is _not_ the direct result of the premise") would
be mis-mapped to entailment.

**Empirical check (baseline t1+t2, 160,320 XNLI rows):** Tier 3 resolves 4,435
rows; **0** of them have a negation marker (`不/没/沒/未/非`) preceding the
matched phrase. The negation trap is structurally real but does not fire on
the evaluation data — likely because `max_new_tokens=10` keeps outputs short,
so the model emits the positive frame and rarely the longer negated form.

**Decision:** do not implement a negation guard (new code, no empirical
benefit, risks introducing its own bugs). Document this as a known latent
limitation: a larger generation budget or a differently-biased model could
surface negated paraphrases, at which point a guard would be warranted.

### REJECTED — left as a parse-failure (NOT counted)

| Pattern              | Example raw_output                                         | Status           | Rationale                                                                                                         |
| -------------------- | ---------------------------------------------------------- | ---------------- | ----------------------------------------------------------------------------------------------------------------- |
| Argument-role only   | `假设` ("hypothesis"), `假设。` ("hypothesis.")            | CONFIRMED reject | The model names only the argument role, never the relationship. No label, and not even a paraphrase to interpret. |
| Negated Tier-2 frame | `假设和前提之间没有entailment` ("there is no entailment…") | CONFIRMED reject | See Tier 2 — the model negated the label without committing to which of the other two it means.                   |
| Empty / off-task     | ``, `1. 2. 3.`, `平衡`("balance"),`无` ("none")            | CONFIRMED reject | No label and no interpretable paraphrase.                                                                         |

### Scope note

`template1` XNLI parses cleanly (parse-failure ≈ 0%). The 8 high-parse-failure
cells are all `template2`. Tiers 2 and 3 recover the rows where the model's
intent is clear; the REJECT rows above are genuine non-answers and stay
visible.

### Known model-behaviour finding (NOT a scoring decision)

XNLI accuracy hovers near the 33% three-way chance level, and the model shows
a heavy `entailment` bias — including the Tier-2 finding that _every_
CJK-glued label form is `entailment`. Same degenerate-constant pattern as
SIB-200's science/technology. Reported as a model finding.

---

## 3. X-CSQA — commonsense multiple choice

### Approved answer set

`A` · `B` · `C` · `D` · `E`

### ACCEPTED

| Rule           | Example raw_output                  | Scored as  | Status    | Rationale                                                                                   |
| -------------- | ----------------------------------- | ---------- | --------- | ------------------------------------------------------------------------------------------- |
| Bare letter    | `A`                                 | A          | CONFIRMED | The whole answer is the choice letter.                                                      |
| Letter in text | `B. office`, `A.`, `C. سوپر مارکیٹ` | the letter | CONFIRMED | Leading `\b[A-E]\b` — the model gave the letter then restated the option (in any language). |
| Answer-prefix  | `ANSWERD`                           | D          | CONFIRMED | Near-vestigial — the bare-letter rule catches almost everything first.                      |

### REJECTED

| Pattern      | Example | Status           | Rationale                                                             |
| ------------ | ------- | ---------------- | --------------------------------------------------------------------- |
| Empty output | ``      | CONFIRMED reject | The model produced nothing. The _only_ parse-failure mode for X-CSQA. |

**No change needed.** X-CSQA parse-failure rate is ≈ 0% in all 32 cells —
choice letters are language-neutral.

---

## 4. Belebele — reading-comprehension multiple choice

### Approved answer set

`A` · `B` · `C` · `D` (4-way, no `E`)

Identical scoring to X-CSQA, restricted to A–D. Parse-failure rate ≈ 0% in
all 32 cells. **No change needed.**

---

## Summary of changes this re-parsing makes

| Benchmark | Change                                                      | Effect                                                                                                                                                       |
| --------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| SIB-200   | Add native-category + native-science-variant + Arabic rules | Recovers the model's in-language answers across politics/sports/travel/health/entertainment/geography. Compounds and off-scheme answers stay parse-failures. |
| XNLI      | Add CJK-glued English-label rule                            | Recovers explicit `entailment`-glued-to-CJK answers; role-token explanations stay parse-failures.                                                            |
| X-CSQA    | None                                                        | Already correct.                                                                                                                                             |
| Belebele  | None                                                        | Already correct.                                                                                                                                             |

Every accepted form is one the model produced. Every rejected form is a real
model limitation we chose to keep visible rather than paper over. The
extractor is applied identically to the baseline and to every fine-tuned
condition — no condition gets a bespoke parser.

---

_This ledger is finalized once the three native-speaker review docs are
returned. PENDING REVIEW rows move to CONFIRMED (or are corrected) at that
point, and this document becomes the methodology appendix for the paper._
