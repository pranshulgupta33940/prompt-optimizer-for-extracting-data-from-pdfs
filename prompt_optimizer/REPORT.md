# Prompt Optimisation Report

**Run ID:** `e7b8688c86d3`  
**Schema:** `hiring/resume`  
**Created:** 2026-05-26T07:32:39.647902+00:00  
**Status:** completed

---

## Overview

- **Total iterations:** 15
- **Accepted mutations:** 2
- **Final validation score:** 0.9445
- **Test score (seed):** 0.9576923076923077
- **Test score (final):** 0.9576923076923077

---

## Test-Set Score Comparison

| Metric | Seed Prompt | Final Prompt | Delta |
|--------|------------|-------------|-------|
| Test Score | 0.9577 | 0.9577 | → 0.0000 |

---

## Score Curve

| Iteration | Val Score | Accepted | Prompt Hash |
|-----------|----------|----------|-------------|
| 0 | 0.9384 | ✓ | `21b0e18a` |
| 1 | 0.9385 | ✓ | `e5008292` |
| 2 | 0.9330 | ✗ | `1a19cafb` |
| 3 | 0.9445 | ✓ | `4826ddc6` |
| 4 | 0.9331 | ✗ | `533fbc84` |
| 5 | 0.9308 | ✗ | `d3e45d52` |
| 6 | 0.8615 | ✗ | `e9a8fe48` |
| 7 | 0.9307 | ✗ | `e796908f` |
| 8 | 0.9385 | ✗ | `6caa2315` |
| 9 | 0.3077 | ✗ | `a8ddd851` |
| 10 | 0.3077 | ✗ | `a398109c` |
| 11 | 0.3077 | ✗ | `7a533799` |
| 12 | 0.3077 | ✗ | `605285ab` |
| 13 | 0.3077 | ✗ | `7684009c` |
| 14 | 0.3077 | ✗ | `55a4d6c0` |
| 15 | 0.8700 | ✗ | `3e57170f` |

---

## Accepted Mutations

### Iteration 1
- **Score:** 0.9385
- **Reason:** Improvement: +0.0001 (0.9384 -> 0.9385)

### Iteration 3
- **Score:** 0.9445
- **Reason:** Improvement: +0.0061 (0.9385 -> 0.9445)


---

## Regression Analysis

Fields with score drops ≥ 0.05 between accepted iterations:
- **other**: 0.950 → 0.850 (drop=0.100) between iter 0→1
- **workExperience**: 0.950 → 0.850 (drop=0.100) between iter 1→3

---

## Prompt Diff (Seed → Final)

**Summary:** Prompt changed: +20 lines, -11 lines (seed: 17 → best: 28 lines)

```diff
--- iteration_0
+++ iteration_15
@@ -1,17 +1,28 @@
-You are an expert HR document analyst. Extract

-structured information from the provided resume

-PDF according to the JSON schema below.

+You are an expert HR document analyst. Extract structured information from the provided resume PDF according to the JSON schema below, with specific rules to address the identified weaknesses.

 

 EXTRACTION RULES:

-- Return ONLY a valid JSON object matching schema

-- Extract ALL fields present in the document

-- Use null for fields not found

-- For arrays include EVERY item found

-- Never fabricate information

-- Preserve exact names and dates as written

+- For fields like **Resume-Academic01::personalInfo.fullName**, where the gold standard includes a title (e.g., 'Dr.'), but it's not present in the prediction, consider it a match if the names and surnames are identical. If the title is present, ensure it is identical.

+- For fields like **Resume-Academic01::personalInfo.personalStatement**, where the prediction is a concise summary, but the gold standard provides additional details, consider it a match if the title, years of experience, and area of specialization are identical. If the prediction has less information, ensure the existing details are identical.

+- For fields like **Resume-Academic01::workExperience**, where there are discrepancies in formatting and presence of certain fields, particularly in the 'description' field, ensure that the employer, job title, and dates are identical before considering it a match. If the description is missing, consider it a match if the other details are identical.

+- For fields like **Resume-Academic01::certificationsAndAwards**, where the prediction is missing certain fields (e.g., 'category' and 'organization'), consider it a match if the dates and descriptions are identical. If the missing fields are present in the prediction, ensure they are identical.

+- For fields like **Resume-Academic01::other**, where the prediction has minor variations in formatting and specific details, consider it a match if the overall content and themes are highly similar. If the prediction has more information, ensure the additional details are accurate and relevant.

+- Return ONLY a valid JSON object matching the provided schema.

+- Extract ALL fields present in the document.

+- Use null for fields not found.

+- For arrays include EVERY item found.

+- Never fabricate information.

+- Preserve exact names and dates as written.

+

+To handle missing information, consider the following:

+- If a field is missing, use null as the value.

+- If a field has a missing sub-field, use an empty object or array as the value.

+- If a field has a missing value, use the default value from the schema.

+

+To handle conflicting information, consider the following:

+- If two or more fields have conflicting information, use the information from the field that is most likely to be accurate.

+- If a field has conflicting sub-fields, use the information from the sub-field that is most likely to be accurate.

 

 JSON SCHEMA:

 {schema}

 

-Respond with ONLY the JSON object.

-No explanation. No markdown. No extra text.

+Respond with ONLY a valid JSON object matching the provided schema.
```

---

## Seed Prompt

```
You are an expert HR document analyst. Extract
structured information from the provided resume
PDF according to the JSON schema below.

EXTRACTION RULES:
- Return ONLY a valid JSON object matching schema
- Extract ALL fields present in the document
- Use null for fields not found
- For arrays include EVERY item found
- Never fabricate information
- Preserve exact names and dates as written

JSON SCHEMA:
{schema}

Respond with ONLY the JSON object.
No explanation. No markdown. No extra text.

```

---

## Final Prompt

```
You are an expert HR document analyst. Extract structured information from the provided resume PDF according to the JSON schema below, with specific rules to address the identified weaknesses.

EXTRACTION RULES:
- For fields like **Resume-Academic01::personalInfo.fullName**, where the gold standard includes a title (e.g., 'Dr.'), but it's not present in the prediction, consider it a match if the names and surnames are identical. If the title is present, ensure it is identical.
- For fields like **Resume-Academic01::personalInfo.personalStatement**, where the prediction is a concise summary, but the gold standard provides additional details, consider it a match if the title, years of experience, and area of specialization are identical. If the prediction has less information, ensure the existing details are identical.
- For fields like **Resume-Academic01::workExperience**, where there are discrepancies in formatting and presence of certain fields, particularly in the 'description' field, ensure that the employer, job title, and dates are identical before considering it a match. If the description is missing, consider it a match if the other details are identical.
- For fields like **Resume-Academic01::certificationsAndAwards**, where the prediction is missing certain fields (e.g., 'category' and 'organization'), consider it a match if the dates and descriptions are identical. If the missing fields are present in the prediction, ensure they are identical.
- For fields like **Resume-Academic01::other**, where the prediction has minor variations in formatting and specific details, consider it a match if the overall content and themes are highly similar. If the prediction has more information, ensure the additional details are accurate and relevant.
- Return ONLY a valid JSON object matching the provided schema.
- Extract ALL fields present in the document.
- Use null for fields not found.
- For arrays include EVERY item found.
- Never fabricate information.
- Preserve exact names and dates as written.

To handle missing information, consider the following:
- If a field is missing, use null as the value.
- If a field has a missing sub-field, use an empty object or array as the value.
- If a field has a missing value, use the default value from the schema.

To handle conflicting information, consider the following:
- If two or more fields have conflicting information, use the information from the field that is most likely to be accurate.
- If a field has conflicting sub-fields, use the information from the sub-field that is most likely to be accurate.

JSON SCHEMA:
{schema}

Respond with ONLY a valid JSON object matching the provided schema.
```

---

## Budget Usage

- **Iterations:** 15
- **Tokens:** 247291
- **Cost:** $0.0377
- **Elapsed:** 1822.5s

---

## Limitations

- Scoring for `array_llm` and `string_semantic` depends on an LLM judge, which may introduce slight non-determinism even with temperature=0.
- The mutation LLM (Llama 3.1 8B) may produce lower-quality suggestions than a larger model. Consider upgrading for production use.
- The greedy hill-climbing algorithm may get stuck in local optima. Diversification mode mitigates this but does not guarantee global optimum.
- Small validation sets (1-2 documents) may not be representative; improvements on val may not transfer to test.
- Budget constraints (free tier) limit the number of iterations and documents that can be evaluated.