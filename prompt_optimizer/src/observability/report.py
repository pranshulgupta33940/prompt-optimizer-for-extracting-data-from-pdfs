"""Report generator — produces ``REPORT.md`` from a completed run's state.

Includes: seed vs final prompt, test-set comparison, score curve,
regression detection, per-section breakdown, and prompt diff.
"""

from pathlib import Path
from typing import Any

from src.observability.diff import prompt_diff, prompt_diff_summary
from src.optimizer.state import RunState


class ReportGenerator:
    """Generates a Markdown report from an optimisation run state."""

    def __init__(self, state: RunState, run_dir: Path) -> None:
        self._state = state
        self._run_dir = run_dir

    def generate(self, output_path: Path | None = None) -> str:
        """Build the full Markdown report.

        Args:
            output_path: If set, write the report to this file path.

        Returns:
            The report as a Markdown string.
        """
        sections = [
            self._header(),
            self._overview(),
            self._score_comparison(),
            self._score_curve(),
            self._accepted_mutations(),
            self._regression_analysis(),
            self._prompt_diff_section(),
            self._seed_prompt_section(),
            self._final_prompt_section(),
            self._budget_summary(),
            self._limitations(),
        ]

        report = "\n\n---\n\n".join(s for s in sections if s)

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as fh:
                fh.write(report)

        return report

    # -- Sections ----------------------------------------------------------

    def _header(self) -> str:
        return (
            f"# Prompt Optimisation Report\n\n"
            f"**Run ID:** `{self._state.run_id}`  \n"
            f"**Schema:** `{self._state.schema_name}`  \n"
            f"**Created:** {self._state.created_at}  \n"
            f"**Status:** {self._state.status}"
        )

    def _overview(self) -> str:
        s = self._state
        n_accepted = sum(1 for r in s.prompt_history if r.accepted and r.iteration > 0)
        n_total = len(s.prompt_history) - 1  # exclude seed
        return (
            f"## Overview\n\n"
            f"- **Total iterations:** {n_total}\n"
            f"- **Accepted mutations:** {n_accepted}\n"
            f"- **Final validation score:** {s.current_best_score:.4f}\n"
            f"- **Test score (seed):** {s.test_score_seed if s.test_score_seed is not None else 'N/A'}\n"
            f"- **Test score (final):** {s.test_score_final if s.test_score_final is not None else 'N/A'}"
        )

    def _score_comparison(self) -> str:
        s = self._state
        if s.test_score_seed is None or s.test_score_final is None:
            return ""

        delta = s.test_score_final - s.test_score_seed
        direction = "↑" if delta > 0 else ("↓" if delta < 0 else "→")

        return (
            f"## Test-Set Score Comparison\n\n"
            f"| Metric | Seed Prompt | Final Prompt | Delta |\n"
            f"|--------|------------|-------------|-------|\n"
            f"| Test Score | {s.test_score_seed:.4f} | {s.test_score_final:.4f} | {direction} {abs(delta):.4f} |"
        )

    def _score_curve(self) -> str:
        lines = ["## Score Curve\n"]
        lines.append("| Iteration | Val Score | Accepted | Prompt Hash |")
        lines.append("|-----------|----------|----------|-------------|")

        for r in self._state.prompt_history:
            status = "✓" if r.accepted else "✗"
            lines.append(
                f"| {r.iteration} | {r.validation_score:.4f} | {status} | `{r.prompt_hash[:8]}` |"
            )

        return "\n".join(lines)

    def _accepted_mutations(self) -> str:
        accepted = [
            r for r in self._state.prompt_history
            if r.accepted and r.iteration > 0
        ]
        if not accepted:
            return "## Accepted Mutations\n\nNo mutations were accepted."

        lines = ["## Accepted Mutations\n"]
        for r in accepted:
            lines.append(
                f"### Iteration {r.iteration}\n"
                f"- **Score:** {r.validation_score:.4f}\n"
                f"- **Reason:** {r.reason}\n"
            )

        return "\n".join(lines)

    def _regression_analysis(self) -> str:
        """Detect fields where score dropped between consecutive accepted iterations."""
        accepted = [
            r for r in self._state.prompt_history if r.accepted
        ]
        if len(accepted) < 2:
            return ""

        regressions: list[str] = []
        for i in range(1, len(accepted)):
            prev = accepted[i - 1]
            curr = accepted[i]

            for field, prev_score in prev.field_scores_summary.items():
                curr_score = curr.field_scores_summary.get(field, 0.0)
                drop = prev_score - curr_score
                if drop >= 0.05:
                    regressions.append(
                        f"- **{field}**: {prev_score:.3f} → {curr_score:.3f} "
                        f"(drop={drop:.3f}) between iter {prev.iteration}→{curr.iteration}"
                    )

        if not regressions:
            return "## Regression Analysis\n\nNo significant regressions detected (threshold ≥ 0.05)."

        header = "## Regression Analysis\n\nFields with score drops ≥ 0.05 between accepted iterations:\n"
        return header + "\n".join(regressions)

    def _prompt_diff_section(self) -> str:
        diff_text = prompt_diff(self._state, 0)
        summary = prompt_diff_summary(self._state)

        return (
            f"## Prompt Diff (Seed → Final)\n\n"
            f"**Summary:** {summary}\n\n"
            f"```diff\n{diff_text}\n```"
        )

    def _seed_prompt_section(self) -> str:
        return (
            f"## Seed Prompt\n\n"
            f"```\n{self._state.seed_prompt}\n```"
        )

    def _final_prompt_section(self) -> str:
        return (
            f"## Final Prompt\n\n"
            f"```\n{self._state.current_best_prompt}\n```"
        )

    def _budget_summary(self) -> str:
        b = self._state.budget_used
        if not b:
            return ""

        return (
            f"## Budget Usage\n\n"
            f"- **Iterations:** {b.get('iterations', 0)}\n"
            f"- **Tokens:** {b.get('tokens', 0)}\n"
            f"- **Cost:** ${b.get('cost_usd', 0):.4f}\n"
            f"- **Elapsed:** {b.get('elapsed_seconds', 0):.1f}s"
        )

    def _limitations(self) -> str:
        return (
            "## Limitations\n\n"
            "- Scoring for `array_llm` and `string_semantic` depends on an LLM judge, "
            "which may introduce slight non-determinism even with temperature=0.\n"
            "- The mutation LLM (Llama 3.1 8B) may produce lower-quality suggestions "
            "than a larger model. Consider upgrading for production use.\n"
            "- The greedy hill-climbing algorithm may get stuck in local optima. "
            "Diversification mode mitigates this but does not guarantee global optimum.\n"
            "- Small validation sets (1-2 documents) may not be representative; "
            "improvements on val may not transfer to test.\n"
            "- Budget constraints (free tier) limit the number of iterations and "
            "documents that can be evaluated."
        )
