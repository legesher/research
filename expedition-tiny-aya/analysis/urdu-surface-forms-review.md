# Urdu / Arabic SIB-200 Surface Forms — Native-Speaker Review

**Purpose:** Our evaluation model, when prompted in Urdu, answers the SIB-200
topic-classification task in Urdu (and sometimes Arabic). Our scoring code
currently only recognizes English category words, so it throws away every
Urdu answer as "unparseable." We want to fix that — but only by adding words
the model _actually emitted_, mapped to the _correct_ category.

**What we need from you:** For each word/phrase below, confirm (a) our reading
of what it means, and (b) which of the 7 valid categories it belongs to — or
tell us it belongs to none of them.

**Data source:** Every form below was emitted by the model in the Phase-3
baseline evaluation, in the 8 cells where instructions were given in Urdu
(4 dataset languages × 2 prompt templates). Counts are approximate totals
across those 8 cells.

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

If a word doesn't clearly mean one of these 7, it should stay marked "unparseable"
(it means the model answered with a category that doesn't exist — a real model
error we want to keep visible, not paper over).

---

## Section A — Single-word answers (high confidence)

These appear on their own as the model's whole answer. We believe each maps
cleanly to one category. **Please confirm the meaning and category.**

| #   | Urdu  | Our transliteration | Our reading                | → Category        | Approx. count | Correct? |
| --- | ----- | ------------------- | -------------------------- | ----------------- | ------------- | -------- |
| A1  | سیاست | siyāsat             | politics                   | **politics**      | ~230          | ☐        |
| A2  | سفر   | safar               | journey / travel           | **travel**        | ~65           | ☐        |
| A3  | سیاحت | siyāḥat             | tourism                    | **travel**        | ~80           | ☐        |
| A4  | صحت   | ṣiḥḥat              | health                     | **health**        | ~20           | ☐        |
| A5  | تفریح | tafrīḥ              | recreation / entertainment | **entertainment** | ~15           | ☐        |
| A6  | کھیل  | khel                | game / sport               | **sports**        | ~10           | ☐        |

**Key question for A2 vs A3:** we are mapping _both_ سفر and سیاحت to **travel**.
Is that right, or should one of them map elsewhere? (SIB-200 has no separate
"tourism" category — the closest is "travel".)

---

## Section B — Science/technology, alternative spellings

The model writes "science/technology" several ways. We already recognize
`سائنس/ٹکنالوجی` and `سائنس/تکنالوجی`. The forms below we currently MISS.
**Please confirm each one means "science / technology".**

| #   | Urdu            | Our transliteration | Our reading                         | Approx. count | Means science/technology? |
| --- | --------------- | ------------------- | ----------------------------------- | ------------- | ------------------------- |
| B1  | سائنس/ٹیکنالوجی | sā'ins / ṭaiknāloji | science/technology (ٹ+ی+ک spelling) | ~88           | ☐                         |
| B2  | سائنس/تکنولوجی  | sā'ins / teknoloji  | science/technology (و spelling)     | ~6            | ☐                         |
| B3  | علم و ٹیکنالوجی | ʿilm o ṭaiknāloji   | "knowledge and technology"          | ~4            | ☐                         |
| B4  | سائنس/تکنولوجی  | sā'ins / teknolojī  | science/technology (variant)        | ~3            | ☐                         |

**Key question for B3:** علم means "knowledge / science" — is "علم و ٹیکنالوجی"
a legitimate way to say "science and technology", or does it mean something
narrower we should not treat as science/technology?

---

## Section C — Science/technology sub-topics (transliterated English)

SIB-200's "science/technology" category covers sub-topics like physics,
chemistry, transportation. The model sometimes answers with a transliterated
sub-topic instead of the category name. **Confirm each is a science/tech topic.**

| #   | Urdu             | Our transliteration  | Our reading          | → science/technology? |
| --- | ---------------- | -------------------- | -------------------- | --------------------- |
| C1  | انٹرایکٹو ڈیزائن | inṭarayikṭiv ḍizā'in | "interactive design" | ☐                     |
| C2  | سیٹیلائٹ فون     | saiṭalā'iṭ fon       | "satellite phone"    | ☐                     |
| C3  | انٹرنیٹ پراکسی   | inṭarneṭ proksi      | "internet proxy"     | ☐                     |
| C4  | ٹرانسپورٹیشن     | ṭrānsporṭeshan       | "transportation"     | ☐                     |

---

## Section D — Arabic forms (the model code-switches)

When prompted in Urdu, the model occasionally answers in **Arabic** instead
(the scripts overlap). **Confirm these Arabic words and their categories.**

| #   | Arabic      | Our transliteration | Our reading | → Category             | Approx. count | Correct? |
| --- | ----------- | ------------------- | ----------- | ---------------------- | ------------- | -------- |
| D1  | الرياضة     | ar-riyāḍa           | sport       | **sports**             | ~4            | ☐        |
| D2  | السياسة     | as-siyāsa           | politics    | **politics**           | ~2            | ☐        |
| D3  | التكنولوجيا | at-tiknolojiyā      | technology  | **science/technology** | ~2            | ☐        |

(D2 and D3 appeared together as "السياسة/التكنولوجيا" — see Section E on
compounds.)

---

## Section E — Compound answers (the model names TWO things)

The model often answers with two words joined by "/". This is the trickiest
case. We need to know, for each: **do both halves mean the SAME category, or
DIFFERENT categories?**

- If both halves mean the **same** category → we can safely score it.
- If they mean **different** categories → the model hedged / couldn't decide,
  and we'll keep it marked "unparseable" (a real model limitation worth showing).

| #   | Urdu                 | Our transliteration          | Half 1 means…     | Half 2 means…             | Same category?     |
| --- | -------------------- | ---------------------------- | ----------------- | ------------------------- | ------------------ |
| E1  | کھیل/سپورٹس          | khel / sporṭs                | sport (Urdu)      | sports (English loanword) | ☐ same ☐ different |
| E2  | سفر/مسافرت           | safar / musāfarat            | travel            | travel (?)                | ☐ same ☐ different |
| E3  | سفر/مسافر            | safar / musāfir              | travel            | traveller (?)             | ☐ same ☐ different |
| E4  | سیاحت/سفر            | siyāḥat / safar              | tourism           | travel                    | ☐ same ☐ different |
| E5  | کھیل/تکنالوجی        | khel / taknāloji             | sport             | technology                | ☐ same ☐ different |
| E6  | کھیل/ٹیکنالوجی       | khel / ṭaiknāloji            | sport             | technology                | ☐ same ☐ different |
| E7  | کھیل/سائنس/تکنالوجی  | khel / sā'ins / taknāloji    | sport             | science / technology      | ☐ same ☐ different |
| E8  | سیاحت/تفریح          | siyāḥat / tafrīḥ             | tourism           | entertainment             | ☐ same ☐ different |
| E9  | سیاحت/تکنالوجی       | siyāḥat / taknāloji          | tourism           | technology                | ☐ same ☐ different |
| E10 | سیاحت/ٹیکنالوجی      | siyāḥat / ṭaiknāloji         | tourism           | technology                | ☐ same ☐ different |
| E11 | سیاحت/سائنس/تکنالوجی | siyāḥat / sā'ins / taknāloji | tourism           | science / technology      | ☐ same ☐ different |
| E12 | سفر/تکنالوجی         | safar / taknāloji            | travel            | technology                | ☐ same ☐ different |
| E13 | سفر/تفریح            | safar / tafrīḥ               | travel            | entertainment             | ☐ same ☐ different |
| E14 | صحت/تکنالوجی         | ṣiḥḥat / taknāloji           | health            | technology                | ☐ same ☐ different |
| E15 | تفریح/سرگرمی         | tafrīḥ / sargarmī            | entertainment     | activity (?)              | ☐ same ☐ different |
| E16 | فن/تکنالوجی          | fan / taknāloji              | art               | technology                | ☐ same ☐ different |
| E17 | حیات/تکنالوجی        | ḥayāt / taknāloji            | life / biology    | technology                | ☐ same ☐ different |
| E18 | تعلیم/تکنالوجی       | taʿlīm / taknāloji           | education         | technology                | ☐ same ☐ different |
| E19 | صحافت/تکنالوجی       | ṣaḥāfat / taknāloji          | journalism        | technology                | ☐ same ☐ different |
| E20 | السياسة/التكنولوجيا  | as-siyāsa / at-tiknolojiyā   | politics (Arabic) | technology (Arabic)       | ☐ same ☐ different |

---

## Section F — Words that may not be a SIB-200 category at all

We believe these don't cleanly fit any of the 7 categories — the model
answered "off-script." We plan to keep them marked "unparseable." **Confirm
each one is NOT one of the 7 categories** (or tell us if we're wrong).

| #   | Urdu                     | Our transliteration          | Our reading                | Is it one of the 7?  |
| --- | ------------------------ | ---------------------------- | -------------------------- | -------------------- |
| F1  | تعلیم                    | taʿlīm                       | education                  | ☐ no ☐ yes: **\_\_** |
| F2  | صحافت                    | ṣaḥāfat                      | journalism                 | ☐ no ☐ yes: **\_\_** |
| F3  | فن                       | fan                          | art                        | ☐ no ☐ yes: **\_\_** |
| F4  | قانون و ضابطہ            | qānūn o ẓābṭa                | "law and order"            | ☐ no ☐ yes: **\_\_** |
| F5  | حیات                     | ḥayāt                        | life / biology             | ☐ no ☐ yes: **\_\_** |
| F6  | سرگرمی                   | sargarmī                     | activity                   | ☐ no ☐ yes: **\_\_** |
| F7  | سونا                     | sonā                         | gold                       | ☐ no ☐ yes: **\_\_** |
| F8  | فن اور تصویروں کا تخلیقی | fan aur taṣwīroṇ kā taḳhlīqī | "art and creative imagery" | ☐ no ☐ yes: **\_\_** |

---

## Summary of what we'll do with the answers

- **Section A, B, C, D** confirmed → added to the scoring code so these Urdu/
  Arabic answers are counted correctly.
- **Section E** — entries marked "same" → scored; entries marked "different"
  → kept "unparseable" (model hedged).
- **Section F** confirmed "not a category" → stay "unparseable."

Every word added is one the model genuinely produced — we are not inventing
translations, only teaching the scorer to read what the model already said.

---

_Reviewer name(s): ********\_\_\_******** Date: ****\_\_\_****_

_Notes / anything we missed:_
