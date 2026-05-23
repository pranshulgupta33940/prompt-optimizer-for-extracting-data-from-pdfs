"""Budget tracking and enforcement.

Tracks iterations, tokens, cost (USD), and wall-clock time against
configurable limits.  Any single limit being exceeded causes the budget
to be considered exhausted.
"""

import time
from dataclasses import dataclass, field


@dataclass
class BudgetConfig:
    """Configurable budget limits.  ``None`` means unlimited."""

    max_iterations: int | None = 5
    max_tokens: int | None = 100_000
    max_cost_usd: float | None = 0.50
    max_wall_clock_seconds: float | None = 1200.0


@dataclass
class BudgetUsage:
    """Snapshot of how much budget has been consumed so far."""

    iterations: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    elapsed_seconds: float = 0.0


class BudgetTracker:
    """Tracks resource consumption against configured limits."""

    def __init__(self, config: BudgetConfig) -> None:
        """Create a tracker with the given limits.

        Args:
            config: A ``BudgetConfig`` with the caps to enforce.
        """
        self._config = config
        self._iterations = 0
        self._tokens = 0
        self._cost = 0.0
        self._start_time = time.time()

    # -- Recording usage ---------------------------------------------------

    def record_iteration(self) -> None:
        """Increment the iteration counter by one."""
        self._iterations += 1

    def record_tokens(self, count: int) -> None:
        """Add *count* tokens to the running total."""
        self._tokens += count

    def record_cost(self, amount: float) -> None:
        """Add *amount* USD to the running cost."""
        self._cost += amount

    def sync_from_logger(self, total_tokens: int, total_cost: float) -> None:
        """Bulk-update totals from the LLM call logger.

        Args:
            total_tokens: Cumulative tokens from the logger.
            total_cost: Cumulative cost from the logger.
        """
        self._tokens = total_tokens
        self._cost = total_cost

    # -- Querying ----------------------------------------------------------

    def is_exhausted(self) -> bool:
        """Return ``True`` if **any** configured limit has been reached."""
        cfg = self._config
        if cfg.max_iterations is not None and self._iterations >= cfg.max_iterations:
            return True
        if cfg.max_tokens is not None and self._tokens >= cfg.max_tokens:
            return True
        if cfg.max_cost_usd is not None and self._cost >= cfg.max_cost_usd:
            return True
        if cfg.max_wall_clock_seconds is not None:
            if self._elapsed() >= cfg.max_wall_clock_seconds:
                return True
        return False

    def exhaustion_reason(self) -> str | None:
        """Return a human-readable reason if exhausted, else ``None``."""
        cfg = self._config
        if cfg.max_iterations is not None and self._iterations >= cfg.max_iterations:
            return f"Iteration limit reached: {self._iterations}/{cfg.max_iterations}"
        if cfg.max_tokens is not None and self._tokens >= cfg.max_tokens:
            return f"Token limit reached: {self._tokens}/{cfg.max_tokens}"
        if cfg.max_cost_usd is not None and self._cost >= cfg.max_cost_usd:
            return f"Cost limit reached: ${self._cost:.4f}/${cfg.max_cost_usd}"
        if cfg.max_wall_clock_seconds is not None:
            elapsed = self._elapsed()
            if elapsed >= cfg.max_wall_clock_seconds:
                return f"Time limit reached: {elapsed:.0f}s/{cfg.max_wall_clock_seconds:.0f}s"
        return None

    def get_usage(self) -> BudgetUsage:
        """Return a snapshot of current usage."""
        return BudgetUsage(
            iterations=self._iterations,
            tokens=self._tokens,
            cost_usd=round(self._cost, 6),
            elapsed_seconds=round(self._elapsed(), 2),
        )

    def remaining_summary(self) -> str:
        """Return a compact summary of remaining budget."""
        cfg = self._config
        parts: list[str] = []

        if cfg.max_iterations is not None:
            parts.append(f"iter {self._iterations}/{cfg.max_iterations}")
        if cfg.max_tokens is not None:
            parts.append(f"tok {self._tokens}/{cfg.max_tokens}")
        if cfg.max_cost_usd is not None:
            parts.append(f"${self._cost:.4f}/${cfg.max_cost_usd}")
        if cfg.max_wall_clock_seconds is not None:
            parts.append(f"{self._elapsed():.0f}s/{cfg.max_wall_clock_seconds:.0f}s")

        return " | ".join(parts)

    # -- Serialisation (for state persistence) -----------------------------

    def to_dict(self) -> dict:
        """Serialise current usage to a dict (for JSON persistence)."""
        return {
            "iterations": self._iterations,
            "tokens": self._tokens,
            "cost_usd": self._cost,
            "elapsed_seconds": self._elapsed(),
        }

    def restore_from_dict(self, data: dict) -> None:
        """Restore usage counters from a previously persisted dict.

        Args:
            data: Dict produced by ``to_dict()``.
        """
        self._iterations = data.get("iterations", 0)
        self._tokens = data.get("tokens", 0)
        self._cost = data.get("cost_usd", 0.0)

    # -- Private -----------------------------------------------------------

    def _elapsed(self) -> float:
        return time.time() - self._start_time
