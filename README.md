# PIBIT Project

This repository contains the Prompt Optimizer project for automated LLM prompt optimization for structured JSON extraction from PDF documents.

## Project Structure

```
pibit-project/
├── prompt_optimizer/          # Main project directory
│   ├── config/               # Configuration files
│   ├── src/                  # Source code
│   ├── tests/                # Unit tests
│   ├── extract-bench/        # ExtractBench dataset (submodule)
│   ├── pyproject.toml        # Python project configuration
│   ├── README.md             # Project README
│   └── REPORT.md             # Auto-generated report
└── .gitignore               # Git ignore rules

```

## Quick Start

See [prompt_optimizer/README.md](prompt_optimizer/README.md) for detailed instructions.

```bash
cd prompt_optimizer
pip install -e ".[dev]"
python -m src.main --config config/default.yaml
```

## Components

- **Prompt Optimizer**: Automated system that improves LLM prompts for structured JSON extraction
- **ExtractBench**: Benchmark dataset for evaluating extraction quality
- **Evaluation Metrics**: Comprehensive metrics for assessing extraction accuracy

## Requirements

- Python 3.10+
- Google AI Studio API key
- Groq API key

## License

See individual project directories for license information.
