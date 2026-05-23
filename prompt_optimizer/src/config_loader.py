"""YAML configuration loader with validation.

Loads a YAML config file into a plain dict, validates required keys,
and resolves relative paths against the project root.
"""

import os
from pathlib import Path
from typing import Any

import yaml


_REQUIRED_KEYS = [
    "dataset",
    "seed_prompt",
    "extraction_llm",
    "mutation_llm",
]

_REQUIRED_LLM_KEYS = ["provider", "model", "api_key_env"]


def load_config(config_path: str) -> dict[str, Any]:
    """Load and validate a YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Validated configuration dict.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If required keys are missing.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    _validate(config)
    _set_defaults(config)
    _resolve_paths(config, path.parent)

    return config


def _validate(config: dict) -> None:
    """Check that all required keys are present."""
    for key in _REQUIRED_KEYS:
        if key not in config:
            raise ValueError(f"Missing required config key: '{key}'")

    for section in ("extraction_llm", "mutation_llm"):
        if section in config:
            for key in _REQUIRED_LLM_KEYS:
                if key not in config[section]:
                    raise ValueError(
                        f"Missing key '{key}' in '{section}' config"
                    )

    ds = config.get("dataset", {})
    if "path" not in ds or "schema" not in ds:
        raise ValueError("'dataset' must include 'path' and 'schema' keys")


def _set_defaults(config: dict) -> None:
    """Fill in default values for optional keys."""
    ds = config.setdefault("dataset", {})
    ds.setdefault("split_seed", 42)
    ds.setdefault("split_ratios", [0.7, 0.15, 0.15])

    for section in ("extraction_llm", "mutation_llm", "scoring_llm"):
        if section in config:
            config[section].setdefault("temperature", 0)
            config[section].setdefault("max_output_tokens", 8192)

    if "scoring_llm" not in config:
        config["scoring_llm"] = dict(config["mutation_llm"])

    budget = config.setdefault("budget", {})
    budget.setdefault("max_iterations", 5)
    budget.setdefault("max_tokens", 100_000)
    budget.setdefault("max_cost_usd", 0.50)
    budget.setdefault("max_wall_clock_seconds", 1200)

    opt = config.setdefault("optimizer", {})
    opt.setdefault("stall_threshold", 3)
    opt.setdefault("duplicate_similarity_threshold", 0.95)
    opt.setdefault("overfitting_check_interval", 3)
    opt.setdefault("overfitting_gap_threshold", 0.1)

    config.setdefault("output", {}).setdefault("run_dir", "./runs")

    dr = config.setdefault("dry_run", {})
    dr.setdefault("enabled", False)
    dr.setdefault("docs_per_split", 2)


def _resolve_paths(config: dict, config_dir: Path) -> None:
    """Resolve relative paths against the config file's directory.

    This ensures the system works regardless of the current working
    directory by anchoring all paths to the config file location.
    """
    ds = config.get("dataset", {})
    if "path" in ds:
        ds_path = Path(ds["path"])
        if not ds_path.is_absolute():
            ds["path"] = str((config_dir / ds_path).resolve())

    out = config.get("output", {})
    if "run_dir" in out:
        run_path = Path(out["run_dir"])
        if not run_path.is_absolute():
            out["run_dir"] = str((config_dir / run_path).resolve())
