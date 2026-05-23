"""Run-state persistence and crash-safe resumability.

The state file (``run_state.json``) is written atomically after every
iteration so that a killed process can resume from the last completed
iteration without re-executing any LLM calls.
"""

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class PromptRecord:
    """Record of a single optimisation iteration."""

    iteration: int
    prompt: str
    validation_score: float
    test_score: float | None = None
    accepted: bool = False
    reason: str = ""
    timestamp: str = ""
    prompt_hash: str = ""
    field_scores_summary: dict[str, float] = field(default_factory=dict)


@dataclass
class RunState:
    """Full state of an optimisation run — persisted to disk."""

    run_id: str
    schema_name: str
    seed_prompt: str
    current_best_prompt: str
    current_best_score: float
    current_iteration: int
    status: str  # "running" | "completed" | "interrupted"
    budget_used: dict[str, Any] = field(default_factory=dict)
    prompt_history: list[PromptRecord] = field(default_factory=list)
    test_score_seed: float | None = None
    test_score_final: float | None = None
    created_at: str = ""
    updated_at: str = ""

    # -- Persistence -------------------------------------------------------

    def save(self, state_path: Path) -> None:
        """Atomically write state to disk (write-then-rename).

        Args:
            state_path: Path for the ``run_state.json`` file.
        """
        self.updated_at = _now_iso()
        state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = state_path.with_suffix(".tmp")

        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(self._to_dict(), fh, indent=2, default=str)

        _atomic_rename(tmp_path, state_path)

    @classmethod
    def load(cls, state_path: Path) -> "RunState":
        """Load state from disk.

        Args:
            state_path: Path to the ``run_state.json`` file.

        Returns:
            A ``RunState`` instance.

        Raises:
            FileNotFoundError: If the state file does not exist.
        """
        with open(state_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls._from_dict(data)

    @classmethod
    def exists(cls, state_path: Path) -> bool:
        """Check whether a state file exists at the given path."""
        return state_path.exists()

    # -- Helpers -----------------------------------------------------------

    def add_iteration(self, record: PromptRecord) -> None:
        """Append an iteration record and update best if accepted.

        Args:
            record: The ``PromptRecord`` for this iteration.
        """
        if not record.timestamp:
            record.timestamp = _now_iso()
        self.prompt_history.append(record)
        self.current_iteration = record.iteration

        if record.accepted:
            self.current_best_prompt = record.prompt
            self.current_best_score = record.validation_score

    @staticmethod
    def create_new(schema_name: str, seed_prompt: str) -> "RunState":
        """Create a fresh run state for a new optimisation run.

        Args:
            schema_name: The target schema identifier.
            seed_prompt: The starting prompt text.

        Returns:
            A new ``RunState`` instance.
        """
        return RunState(
            run_id=uuid.uuid4().hex[:12],
            schema_name=schema_name,
            seed_prompt=seed_prompt,
            current_best_prompt=seed_prompt,
            current_best_score=0.0,
            current_iteration=0,
            status="running",
            created_at=_now_iso(),
        )

    def _to_dict(self) -> dict:
        """Serialise the state to a plain dict."""
        data = asdict(self)
        data["prompt_history"] = [asdict(r) for r in self.prompt_history]
        return data

    @classmethod
    def _from_dict(cls, data: dict) -> "RunState":
        """Deserialise a state from a plain dict."""
        history_raw = data.pop("prompt_history", [])
        history = [PromptRecord(**r) for r in history_raw]
        state = cls(**{k: v for k, v in data.items() if k != "prompt_history"})
        state.prompt_history = history
        return state


# ---------------------------------------------------------------------------
#  Utilities
# ---------------------------------------------------------------------------

def get_run_dir(base_dir: str, schema_name: str, run_id: str) -> Path:
    """Construct the directory path for a run's artefacts.

    Layout: ``{base_dir}/{schema_name_slug}/{run_id}/``

    Args:
        base_dir: Root runs directory.
        schema_name: Schema identifier (e.g. ``'academic/research'``).
        run_id: Unique run identifier.

    Returns:
        Path to the run directory (created if needed).
    """
    slug = schema_name.replace("/", "_")
    run_dir = Path(base_dir) / slug / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _atomic_rename(src: Path, dst: Path) -> None:
    """Rename *src* to *dst* atomically (best-effort on Windows)."""
    try:
        os.replace(str(src), str(dst))
    except OSError:
        if dst.exists():
            dst.unlink()
        src.rename(dst)
