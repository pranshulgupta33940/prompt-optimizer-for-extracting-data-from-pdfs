"""CLI entry point for the prompt optimisation system.

Usage::

    python -m src.main --config config/default.yaml
    python -m src.main --config config/default.yaml --dry-run
"""

import argparse
import signal
import sys
from pathlib import Path

from src.config_loader import load_config
from src.observability.report import ReportGenerator
from src.optimizer.loop import optimize
from src.optimizer.state import RunState


def main() -> None:
    """Parse arguments and run the optimisation pipeline."""
    parser = argparse.ArgumentParser(
        description="Automated Prompt Optimisation for Structured PDF Extraction",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run with only 2 documents per split for testing",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    if args.dry_run:
        config["dry_run"]["enabled"] = True
        print("[DRY-RUN] Using at most 2 documents per split.")

    # -- Register signal handlers for graceful shutdown -------------------
    _register_signals()

    # -- Run optimisation -------------------------------------------------
    print("=" * 60)
    print("  Automated Prompt Optimisation")
    print(f"  Schema: {config['dataset']['schema']}")
    print(f"  Config: {args.config}")
    print("=" * 60)

    try:
        state = optimize(config)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Saving state and exiting...")
        sys.exit(1)

    # -- Generate report --------------------------------------------------
    run_dir = Path(config["output"]["run_dir"])
    schema_slug = config["dataset"]["schema"].replace("/", "_")

    # Find the run directory
    run_base = run_dir / schema_slug
    report_path: Path | None = None
    if run_base.exists():
        for child in sorted(run_base.iterdir(), reverse=True):
            if (child / "run_state.json").exists():
                report_path = child / "REPORT.md"
                actual_run_dir = child
                break

    if report_path is not None:
        generator = ReportGenerator(state, actual_run_dir)
        generator.generate(report_path)
        print(f"\n[REPORT] Generated: {report_path}")

        # Also copy to project root
        root_report = Path(args.config).parent.parent / "REPORT.md"
        try:
            root_report.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"[REPORT] Copied to: {root_report}")
        except Exception:
            pass

    print("\nDone.")


def _register_signals() -> None:
    """Register signal handlers for graceful shutdown."""
    def handler(signum, frame):
        print(f"\n[SIGNAL] Received signal {signum}, shutting down gracefully...")
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
    except (OSError, ValueError):
        pass


if __name__ == "__main__":
    main()
