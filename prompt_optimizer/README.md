# Prompt Optimizer — Automated Prompt Optimisation for Structured Extraction

An automated system that improves LLM prompts for structured JSON extraction
from PDF documents using the [ExtractBench](https://github.com/ContextualAI/extract-bench) dataset.

## Quick Start

### 1. Prerequisites

- Python 3.10+
- [Google AI Studio API key](https://aistudio.google.com/) (free, no credit card)
- [Groq API key](https://console.groq.com/) (free, no credit card)

### 2. Install

```bash
cd prompt_optimizer
pip install -e ".[dev]"
```

### 3. Set API Keys

```bash
# Windows PowerShell
$env:GOOGLE_API_KEY = "your-google-api-key"
$env:GROQ_API_KEY = "your-groq-api-key"

# Linux / macOS
export GOOGLE_API_KEY="your-google-api-key"
export GROQ_API_KEY="your-groq-api-key"
```

### 4. Clone the Dataset

```bash
git clone --depth 1 https://github.com/ContextualAI/extract-bench.git
```

### 5. Run

```bash
# Full run (default config: academic/research schema)
python -m src.main --config config/default.yaml

# Dry run (2 docs per split, near-zero cost)
python -m src.main --config config/default.yaml --dry-run

# Alternate schema (finance/credit_agreement)
python -m src.main --config config/alternate_schema.yaml
```

### 6. Run Tests

```bash
pytest tests/ -v
```

---

## Architecture

```
prompt_optimizer/
├── config/
│   ├── default.yaml              # Default config (academic/research + Gemini + Groq)
│   └── alternate_schema.yaml     # Finance/credit_agreement config
├── src/
│   ├── main.py                   # CLI entry point
│   ├── config_loader.py          # YAML config loading + validation
│   ├── data/
│   │   ├── loader.py             # PDF + gold JSON loading + deterministic splits
│   │   └── schema.py             # Schema parsing + $ref resolution + eval configs
│   ├── scoring/
│   │   ├── scorer.py             # Main scoring entry point (independent of loop)
│   │   ├── metrics.py            # All 9 evaluation metric implementations
│   │   ├── alignment.py          # Hungarian algorithm for array alignment
│   │   └── cache.py              # SQLite caching for stochastic metrics
│   ├── optimizer/
│   │   ├── loop.py               # Main optimisation loop (greedy hill-climbing)
│   │   ├── mutator.py            # LLM-based prompt mutation
│   │   ├── budget.py             # Budget tracking (iterations, tokens, cost, time)
│   │   └── state.py              # Run state persistence + resumability
│   ├── llm/
│   │   ├── client.py             # Gemini + Groq API wrappers with retry logic
│   │   └── logger.py             # JSONL logging for every LLM call
│   └── observability/
│       ├── diff.py               # Prompt diff between iterations
│       └── report.py             # REPORT.md generation
├── tests/
│   ├── test_scoring.py           # Unit tests for all 9 metric types
│   └── test_alignment.py         # Unit tests for Hungarian alignment
├── pyproject.toml
├── README.md
└── REPORT.md                     # Auto-generated after each run
```

---

## Split Policy

Documents are split **deterministically** into train / val / test:

1. **Sort** all document IDs alphabetically.
2. **Shuffle** with `random.Random(42)`.
3. **Allocate** 70% train, 15% val, 15% test (minimum 1 per split).

| Schema | Total | Train | Val | Test |
|--------|-------|-------|-----|------|
| academic/research | 6 | 4 | 1 | 1 |
| finance/10kq | 7 | 5 | 1 | 1 |
| finance/credit_agreement | 10 | 8 | 1 | 1 |
| hiring/resume | 7 | 5 | 1 | 1 |
| sport/swimming | 5 | 3 | 1 | 1 |

---

## Array Alignment Policy

**Hungarian Algorithm** (optimal bipartite matching) via
`scipy.optimize.linear_sum_assignment`.

**Rationale:** The Hungarian algorithm finds the globally optimal one-to-one
assignment between predicted and gold array items, minimising total mismatch.
This is superior to greedy or order-dependent matching because:

- It handles **reordered** arrays correctly.
- It finds the **best possible** alignment, not just a locally good one.
- It is **deterministic** — same input always yields same assignment.
- Time complexity: O(n³), which is acceptable for typical array sizes.

For arrays of objects, pairwise similarity is computed as the average field
score across all evaluation-config-bearing fields. For arrays of primitives,
fuzzy string matching is used.

---

## Evaluation Config Types

| Metric ID | Input Type | How It Works |
|-----------|-----------|--------------|
| `string_exact` | String | Case-sensitive exact match |
| `string_semantic` | String | LLM judge (Groq/Llama), cached |
| `string_fuzzy` | String | Levenshtein ratio (SequenceMatcher) |
| `string_case_insensitive` | String | Lowercase both, then exact match |
| `integer_exact` | Integer | Exact match after int coercion |
| `number_tolerance` | Number | `|pred-gold|/|gold| ≤ tolerance` |
| `number_exact` | Number | Exact match after float coercion |
| `boolean_exact` | Boolean | Exact match after bool coercion |
| `array_llm` | Array | LLM judge for array-level comparison |

---

## Caching Strategy

All stochastic operations are cached to disk to ensure:

1. **Zero wasted API calls** — interrupted runs resume without re-spending.
2. **Deterministic re-runs** — same (prediction, gold) pair → same score.

| Cache | Key | Store |
|-------|-----|-------|
| Metric Cache | SHA-256(metric_id, predicted, gold) | SQLite (WAL mode) |
| Extraction Cache | SHA-256(doc_path, prompt) | SQLite (WAL mode) |

---

## How to Retarget

Switching to a different schema requires **only config changes** — no code edits:

1. Copy `config/default.yaml` to `config/my_schema.yaml`.
2. Change `dataset.schema` to the target (e.g. `finance/10kq`).
3. Update `seed_prompt` with domain-appropriate instructions.
4. Run: `python -m src.main --config config/my_schema.yaml`

See `config/alternate_schema.yaml` for a complete example.

---

## Budget Controls

All limits are configurable in YAML and enforced at runtime:

```yaml
budget:
  max_iterations: 5        # Stop after N iterations
  max_tokens: 100000       # Stop after total token usage
  max_cost_usd: 0.50       # Stop after estimated cost
  max_wall_clock_seconds: 1200  # Stop after 20 minutes
```

The `--dry-run` flag limits each split to 2 documents for near-zero cost testing.

---

## LLM Providers

| Role | Provider | Model | Free Tier |
|------|----------|-------|-----------|
| PDF Extraction | Google AI Studio | gemini-1.5-flash | ✓ (no credit card) |
| Prompt Mutation | Groq | llama-3.1-8b-instant | ✓ (no credit card) |
| Scoring Judge | Groq | llama-3.1-8b-instant | ✓ (no credit card) |
