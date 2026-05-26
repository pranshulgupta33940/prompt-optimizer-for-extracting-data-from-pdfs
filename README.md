# Prompt Optimizer — Automated Prompt Optimisation for Structured Extraction

An automated system that improves LLM prompts for structured 
JSON extraction from PDF documents using the 
[ExtractBench](https://github.com/ContextualAI/extract-bench) dataset.

---

## Quick Start

### 1. Prerequisites
- Python 3.10+
- [Google AI Studio API key](https://aistudio.google.com/) — free, no credit card
- [Groq API key](https://console.groq.com/) — free, no credit card

### 2. Install
```bash
cd prompt_optimizer
pip install -e ".[dev]"
```

### 3. Set API Keys

**Windows PowerShell:**
```powershell
$env:GOOGLE_API_KEY = "your-google-api-key"
$env:GROQ_API_KEY   = "your-groq-api-key"
```

**Linux / macOS:**
```bash
export GOOGLE_API_KEY="your-google-api-key"
export GROQ_API_KEY="your-groq-api-key"
```

**Optional — Multiple Google keys for quota rotation:**
```powershell
$env:GOOGLE_API_KEY_2 = "your-second-google-key"
$env:GOOGLE_API_KEY_3 = "your-third-google-key"
$env:GOOGLE_API_KEY_4 = "your-fourth-google-key"
```

The system automatically rotates to the next key when 
the free tier daily quota (20 requests/day) is exhausted.
A single key is sufficient for dry runs and short runs
under 20 iterations.

### 4. Clone the Dataset
```bash
git clone --depth 1 https://github.com/ContextualAI/extract-bench.git
```

### 5. Run
```bash
# Dry run — 2 docs per split, near-zero cost, tests pipeline
python -m src.main --config config/default.yaml --dry-run

# Full run — academic/research schema
python -m src.main --config config/default.yaml

# Full run — finance/credit_agreement schema
python -m src.main --config config/alternate_schema.yaml

# Full run — hiring/resume schema
python -m src.main --config config/hiring_resume.yaml

# Any custom schema — copy a config and change dataset.schema
python -m src.main --config config/my_schema.yaml
```

### 6. Run Tests
```bash
pytest tests/ -v
```

---

## Architecture
prompt_optimizer/
├── config/
│   ├── default.yaml           # academic/research schema
│   ├── alternate_schema.yaml  # finance/credit_agreement schema
│   └── hiring_resume.yaml     # hiring/resume schema
├── src/
│   ├── main.py                # CLI entry point
│   ├── config_loader.py       # YAML config loading + validation
│   ├── data/
│   │   ├── loader.py          # PDF + gold JSON loading + splits
│   │   └── schema.py          # Schema parsing + $ref resolution
│   ├── scoring/
│   │   ├── scorer.py          # Main scoring entry point
│   │   ├── metrics.py         # All 9 evaluation metric types
│   │   ├── alignment.py       # Hungarian algorithm array alignment
│   │   └── cache.py           # SQLite cache for stochastic metrics
│   ├── optimizer/
│   │   ├── loop.py            # Greedy hill-climbing loop
│   │   ├── mutator.py         # LLM-based prompt mutation
│   │   ├── budget.py          # Budget tracking and enforcement
│   │   └── state.py           # Run state + resumability
│   ├── llm/
│   │   ├── client.py          # Gemini + Groq clients, key rotation
│   │   └── logger.py          # JSONL logging for every LLM call
│   └── observability/
│       ├── diff.py            # Prompt diff between iterations
│       └── report.py          # REPORT.md generation
├── tests/
│   ├── test_scoring.py        # Unit tests — all 9 metric types
│   └── test_alignment.py      # Unit tests — Hungarian alignment
├── pyproject.toml
├── README.md
└── REPORT.md                  # Auto-generated after each run

---

## How to Retarget to a New Schema

Switching schemas requires only a config file change.
No code edits needed.

1. Copy an existing config:
```bash
cp config/default.yaml config/my_schema.yaml
```

2. Edit `config/my_schema.yaml`:
```yaml
dataset:
  schema: "finance/10kq"
seed_prompt: |
  You are an expert analyst...
```

3. Run:
```bash
python -m src.main --config config/my_schema.yaml
```

See `config/hiring_resume.yaml` for a complete 
retargeting example on a different domain.

---

## Split Policy

All runs use a deterministic train/val/test split:

1. Sort all document IDs alphabetically
2. Shuffle with `random.Random(42)`
3. Allocate 70% train / 15% val / 15% test
   (minimum 1 document per split)

| Schema | Total | Train | Val | Test |
|--------|-------|-------|-----|------|
| academic/research | 6 | 4 | 1 | 1 |
| finance/10kq | 7 | 5 | 1 | 1 |
| finance/credit_agreement | 10 | 7 | 1 | 2 |
| hiring/resume | 7 | 5 | 1 | 1 |
| sport/swimming | 5 | 3 | 1 | 1 |

Same seed and ratios across all schemas and configs.
Running the same config twice always produces 
identical splits.

---

## Array Alignment Policy

**Hungarian Algorithm** via `scipy.optimize.linear_sum_assignment`.

Finds the globally optimal one-to-one assignment between 
predicted and gold array items. Superior to greedy or 
positional matching because:

- Handles reordered arrays correctly
- Finds the best possible alignment not just locally good
- Fully deterministic — same input always gives same result
- Time complexity O(n³) acceptable for typical array sizes

For arrays of objects: pairwise similarity is the average 
field score across all evaluation-config fields.
For arrays of primitives: fuzzy string matching is used.

---

## Evaluation Config Types

| Metric | Input | How It Works |
|--------|-------|--------------|
| `string_exact` | String | Case-sensitive exact match |
| `string_semantic` | String | LLM judge via Groq, cached |
| `string_fuzzy` | String | Levenshtein ratio (SequenceMatcher) |
| `string_case_insensitive` | String | Lowercase then exact match |
| `integer_exact` | Integer | Exact match after int coercion |
| `number_tolerance` | Number | Relative error within tolerance |
| `number_exact` | Number | Exact match after float coercion |
| `boolean_exact` | Boolean | Exact match after bool coercion |
| `array_llm` | Array | LLM judge for array comparison |

---

## Optimization Algorithm

**Greedy hill-climbing with stall detection:**

Evaluate seed prompt on validation split → baseline
For each iteration until budget exhausted:
a. Mutator LLM proposes an improved prompt
b. Evaluate candidate on validation split
c. Accept if score improves, reject otherwise
d. After 3 consecutive rejections → diversification mode
e. Skip duplicate prompts via hash + similarity check
f. Every 3 iterations → check val vs test gap
Evaluate best prompt on held-out test split
Generate REPORT.md with full trajectory


**Pathological case handling:**

| Case | Handling |
|------|----------|
| Regression | Reject, log reason, keep current best |
| Stall (3 rejections) | Diversification mode — instruct mutator to try fundamentally different approach |
| Duplicate prompt | Skip via SHA-256 hash check |
| Overfitting | Warn when val/test gap exceeds threshold |

---

## Caching Strategy

All stochastic operations cached to SQLite on disk:

| Cache | Key | Purpose |
|-------|-----|---------|
| Metric cache | SHA-256(metric_id, predicted, gold) | Never score same pair twice |
| Extraction cache | SHA-256(doc_path, prompt) | Never re-extract same doc+prompt |

Interrupted runs resume without re-spending any API calls.

---

## Budget Controls

All limits configurable in YAML:

```yaml
budget:
  max_iterations: 10
  max_tokens: 500000
  max_cost_usd: 1.00
  max_wall_clock_seconds: 5400
```

`--dry-run` limits to 2 docs per split for 
near-zero cost pipeline testing.

---

## LLM Providers

| Role | Provider | Model | Free Tier |
|------|----------|-------|-----------|
| PDF Extraction | Google AI Studio | gemini-2.5-flash | ✓ no card |
| Prompt Mutation | Groq | llama-3.1-8b-instant | ✓ no card |
| Scoring Judge | Groq | llama-3.1-8b-instant | ✓ no card |

---

## Results

### hiring/resume

| Metric | Seed | Final | Delta |
|--------|------|-------|-------|
| Val Score | 0.9384 | 0.9445 | +0.0061 |
| Test Score | 0.9577 | 0.9577 | 0.000 |

- 2 mutations accepted out of 15 iterations
- Val score improved from 0.9384 to 0.9445
- Test score: 0.9577 — strong generalization
- Total cost: $0.0387
- Total tokens: 252,842

---

## Observability

Every run produces in `runs/<schema>/<run_id>/`:
run_state.json       — full iteration history and state
llm_calls.jsonl      — every LLM call with input, output,
tokens, cost, latency
extraction_cache.db  — cached PDF extractions
metric_cache.db      — cached stochastic metric scores
REPORT.md            — score curve, prompt diff,
regression analysis

---

## Known Limitations

1. Free tier quota (20 req/day per key) limits iteration 
   count. Use multiple keys or paid tier for longer runs.
2. Small val splits (1 doc) cause val/test divergence.
   Cross-validation would give more stable signal.
3. Mutation LLM (Llama 3.1 8B) has limited prompt 
   engineering capability. A larger model would propose
   better mutations.
4. Greedy hill-climbing may miss global optimum.
   Population-based search would explore more broadly.
