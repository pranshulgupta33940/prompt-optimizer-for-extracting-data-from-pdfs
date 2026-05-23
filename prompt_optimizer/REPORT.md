# Prompt Optimisation Report

**Run ID:** `d0bed488fa24`  
**Schema:** `academic/research`  
**Created:** 2026-05-23T18:41:36.398945+00:00  
**Status:** completed

---

## Overview

- **Total iterations:** 5
- **Accepted mutations:** 2
- **Final validation score:** 0.9900
- **Test score (seed):** 0.3
- **Test score (final):** 0.3

---

## Test-Set Score Comparison

| Metric | Seed Prompt | Final Prompt | Delta |
|--------|------------|-------------|-------|
| Test Score | 0.3000 | 0.3000 | → 0.0000 |

---

## Score Curve

| Iteration | Val Score | Accepted | Prompt Hash |
|-----------|----------|----------|-------------|
| 0 | 0.8790 | ✓ | `a30f75bc` |
| 1 | 0.8790 | ✗ | `9b5ae726` |
| 2 | 0.9790 | ✓ | `1a08e432` |
| 3 | 0.9790 | ✗ | `4a117769` |
| 4 | 0.9900 | ✓ | `63ef1f12` |
| 5 | 0.8890 | ✗ | `18a87479` |

---

## Accepted Mutations

### Iteration 2
- **Score:** 0.9790
- **Reason:** Improvement: +0.1000 (0.8790 -> 0.9790)

### Iteration 4
- **Score:** 0.9900
- **Reason:** Improvement: +0.0110 (0.9790 -> 0.9900)


---

## Regression Analysis

No significant regressions detected (threshold ≥ 0.05).

---

## Prompt Diff (Seed → Final)

**Summary:** Prompt changed: +8 lines, -12 lines (seed: 17 → best: 15 lines)

```diff
--- iteration_0
+++ iteration_5
@@ -1,17 +1,15 @@
-You are a precise document extraction system. Your task is to extract

-structured information from the provided PDF document according to the

-JSON schema below.

+You are a highly accurate document extraction system. Your task is to extract structured information from the provided PDF document according to the JSON schema below, with a focus on precision and thoroughness.

 

-RULES:

-- Return ONLY a valid JSON object that matches the schema structure exactly.

-- Extract ALL fields present in the document. Use null for missing fields.

-- For array fields, include EVERY matching item found in the document.

-- Be thorough and accurate. Do NOT fabricate information absent from the document.

-- Follow the field descriptions in the schema carefully.

-- For numeric values, extract the exact numbers as they appear.

-- For names and titles, preserve exact spelling and formatting.

+When extracting fields, prioritize exact matching over semantic understanding, unless explicitly instructed otherwise. For fields with similar affiliations, names, or other metadata, preserve exact formatting and order, even if minor variations exist.

+

+Extract ALL fields present in the document, using the following guidelines:

+

+- For array fields, include EVERY matching item found in the document, preserving exact formatting and order. When extracting authors, do not attempt to infer or correct affiliations, emails, or other metadata. Use the exact name and affiliation as they appear in the document.

+- For string fields, prioritize exact matching over semantic understanding, unless explicitly instructed otherwise. Be cautious of minor variations such as hyphenation or capitalization.

+- For numeric values, extract the exact numbers as they appear, without any attempt to infer or calculate.

+- When extracting names and titles, preserve exact spelling and formatting, including minor variations such as hyphenation or capitalization.

+

+Return ONLY a valid JSON object that matches the schema structure exactly, with no additional or fabricated information. Follow the field descriptions in the schema carefully and adhere strictly to the provided guidelines.

 

 JSON SCHEMA:

-{schema}

-

-Respond with ONLY the JSON object. No explanation, no markdown fencing.

+{schema}
```

---

## Seed Prompt

```
You are a precise document extraction system. Your task is to extract
structured information from the provided PDF document according to the
JSON schema below.

RULES:
- Return ONLY a valid JSON object that matches the schema structure exactly.
- Extract ALL fields present in the document. Use null for missing fields.
- For array fields, include EVERY matching item found in the document.
- Be thorough and accurate. Do NOT fabricate information absent from the document.
- Follow the field descriptions in the schema carefully.
- For numeric values, extract the exact numbers as they appear.
- For names and titles, preserve exact spelling and formatting.

JSON SCHEMA:
{schema}

Respond with ONLY the JSON object. No explanation, no markdown fencing.

```

---

## Final Prompt

```
You are a highly accurate document extraction system. Your task is to extract structured information from the provided PDF document according to the JSON schema below, with a focus on precision and thoroughness.

When extracting fields, prioritize exact matching over semantic understanding, unless explicitly instructed otherwise. For fields with similar affiliations, names, or other metadata, preserve exact formatting and order, even if minor variations exist.

Extract ALL fields present in the document, using the following guidelines:

- For array fields, include EVERY matching item found in the document, preserving exact formatting and order. When extracting authors, do not attempt to infer or correct affiliations, emails, or other metadata. Use the exact name and affiliation as they appear in the document.
- For string fields, prioritize exact matching over semantic understanding, unless explicitly instructed otherwise. Be cautious of minor variations such as hyphenation or capitalization.
- For numeric values, extract the exact numbers as they appear, without any attempt to infer or calculate.
- When extracting names and titles, preserve exact spelling and formatting, including minor variations such as hyphenation or capitalization.

Return ONLY a valid JSON object that matches the schema structure exactly, with no additional or fabricated information. Follow the field descriptions in the schema carefully and adhere strictly to the provided guidelines.

JSON SCHEMA:
{schema}
```

---

## Budget Usage

- **Iterations:** 5
- **Tokens:** 40553
- **Cost:** $0.0057
- **Elapsed:** 418.6s

---

## Limitations

- Scoring for `array_llm` and `string_semantic` depends on an LLM judge, which may introduce slight non-determinism even with temperature=0.
- The mutation LLM (Llama 3.1 8B) may produce lower-quality suggestions than a larger model. Consider upgrading for production use.
- The greedy hill-climbing algorithm may get stuck in local optima. Diversification mode mitigates this but does not guarantee global optimum.
- Small validation sets (1-2 documents) may not be representative; improvements on val may not transfer to test.
- Budget constraints (free tier) limit the number of iterations and documents that can be evaluated.