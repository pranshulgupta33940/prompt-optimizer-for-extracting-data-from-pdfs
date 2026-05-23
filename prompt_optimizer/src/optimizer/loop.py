"""Main optimisation loop — greedy hill-climbing with stall detection.

Algorithm:
  1. Evaluate the seed prompt on the validation split → baseline score.
  2. For each iteration (until budget exhausted):
     a. Call the mutator to generate a candidate prompt.
     b. Evaluate the candidate on the validation split.
     c. Accept if score improves; reject otherwise.
     d. After N consecutive rejections → enter diversification mode.
     e. Skip duplicate prompts (similarity > threshold).
     f. Periodically evaluate on test split for overfitting detection.
     g. Save state after every iteration.
  3. Evaluate the final best prompt on the test split.
  4. Generate the final report.
"""

import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from src.data.loader import DataSplit, DatasetLoader, Document
from src.data.schema import SchemaInfo, SchemaParser
from src.llm.client import LLMClient, create_client
from src.llm.logger import LLMCallLogger
from src.optimizer.budget import BudgetConfig, BudgetTracker
from src.optimizer.mutator import PromptMutator, prompt_hash
from src.optimizer.state import PromptRecord, RunState, get_run_dir
from src.scoring.cache import ExtractionCache, MetricCache
from src.scoring.scorer import ScoringResult, score


# ---------------------------------------------------------------------------
#  Public interface
# ---------------------------------------------------------------------------

def optimize(config: dict) -> RunState:
    """Run the full prompt-optimisation loop.

    Args:
        config: Parsed configuration dict (from YAML).

    Returns:
        The final ``RunState`` with all iteration history.
    """
    # -- Unpack configuration ---------------------------------------------
    ds_cfg = config["dataset"]
    budget_cfg = _make_budget_config(config.get("budget", {}))
    opt_cfg = config.get("optimizer", {})
    output_cfg = config.get("output", {})
    dry_run_cfg = config.get("dry_run", {})

    schema_name: str = ds_cfg["schema"]
    seed_prompt: str = config["seed_prompt"]
    stall_threshold: int = opt_cfg.get("stall_threshold", 3)
    dup_sim_threshold: float = opt_cfg.get("duplicate_similarity_threshold", 0.95)
    overfit_interval: int = opt_cfg.get("overfitting_check_interval", 3)
    overfit_gap: float = opt_cfg.get("overfitting_gap_threshold", 0.1)

    max_docs = dry_run_cfg.get("docs_per_split") if dry_run_cfg.get("enabled") else None

    # -- Load data --------------------------------------------------------
    loader = DatasetLoader(ds_cfg["path"], schema_name)
    data_split = loader.load(
        split_seed=ds_cfg.get("split_seed", 42),
        split_ratios=ds_cfg.get("split_ratios"),
        max_docs_per_split=max_docs,
    )
    schema_info = SchemaParser.parse(loader.get_schema_path())
    extraction_schema = SchemaParser.get_extraction_schema(schema_info)
    schema_text = json.dumps(extraction_schema, indent=2)

    print(f"[INIT] {data_split.summary()}")

    # -- Initialise run directory -----------------------------------------
    state_path, run_dir = _init_run_dir(output_cfg, schema_name, seed_prompt)

    # -- Resume support ---------------------------------------------------
    if RunState.exists(state_path):
        state = RunState.load(state_path)
        if state.status == "completed":
            print(f"[SKIP] Run {state.run_id} already completed.")
            return state
        print(f"[RESUME] Resuming run {state.run_id} from iter {state.current_iteration + 1}")
        start_iter = state.current_iteration + 1
    else:
        state = RunState.create_new(schema_name, seed_prompt)
        start_iter = 0

    # -- Create LLM clients -----------------------------------------------
    call_logger = LLMCallLogger(run_dir / "llm_calls.jsonl")
    extraction_client = _create_llm(config["extraction_llm"], call_logger)
    mutation_client = _create_llm(config["mutation_llm"], call_logger)
    scoring_client = _create_llm(config.get("scoring_llm", config["mutation_llm"]), call_logger)

    mutator = PromptMutator(mutation_client)
    budget = BudgetTracker(budget_cfg)
    if state.budget_used:
        budget.restore_from_dict(state.budget_used)

    extraction_cache = ExtractionCache(run_dir / "extraction_cache.db")
    llm_judge_fn = _make_llm_judge(scoring_client)

    # -- Evaluate seed prompt ---------------------------------------------
    if start_iter == 0:
        print("[ITER 0] Evaluating seed prompt on validation set...")
        seed_score = _evaluate_prompt(
            state.current_best_prompt, data_split.val,
            schema_info, extraction_client, extraction_cache,
            run_dir, llm_judge_fn, schema_text,
        )
        state.current_best_score = seed_score.aggregate_score

        record = PromptRecord(
            iteration=0,
            prompt=seed_prompt,
            validation_score=seed_score.aggregate_score,
            accepted=True,
            reason="Seed prompt baseline",
            prompt_hash=prompt_hash(seed_prompt),
            field_scores_summary={
                k: v for k, v in seed_score.subtree_scores.items()
            },
        )
        state.add_iteration(record)
        state.budget_used = budget.to_dict()
        state.save(state_path)
        budget.sync_from_logger(call_logger.total_tokens, call_logger.total_cost)

        print(f"[ITER 0] Seed score: {seed_score.aggregate_score:.4f}")
        start_iter = 1

    # -- Main loop --------------------------------------------------------
    consecutive_rejections = 0
    seen_hashes: set[str] = {prompt_hash(state.current_best_prompt)}
    last_scoring_result = None

    for iteration in range(start_iter, budget_cfg.max_iterations + 1 if budget_cfg.max_iterations else 999):
        if budget.is_exhausted():
            print(f"[STOP] Budget exhausted: {budget.exhaustion_reason()}")
            break

        budget.record_iteration()
        diversify = consecutive_rejections >= stall_threshold

        if diversify:
            print(f"[ITER {iteration}] DIVERSIFICATION MODE (stall={consecutive_rejections})")
        else:
            print(f"[ITER {iteration}] Generating candidate...")

        # Build mutation history for the mutator
        recent_history = [
            {
                "iteration": r.iteration,
                "accepted": r.accepted,
                "score": r.validation_score,
                "reason": r.reason,
            }
            for r in state.prompt_history[-5:]
        ]

        # Get the last scoring result for mutation context
        if last_scoring_result is None:
            last_scoring_result = _evaluate_prompt(
                state.current_best_prompt, data_split.val,
                schema_info, extraction_client, extraction_cache,
                run_dir, llm_judge_fn, schema_text,
            )

        # Generate candidate
        try:
            candidate = mutator.mutate(
                current_prompt=state.current_best_prompt,
                scoring_result=last_scoring_result,
                schema_text=schema_text,
                history=recent_history,
                diversify=diversify,
            )
        except Exception as exc:
            print(f"[ITER {iteration}] Mutation failed: {exc}")
            consecutive_rejections += 1
            continue

        # Duplicate check
        c_hash = prompt_hash(candidate)
        if c_hash in seen_hashes:
            print(f"[ITER {iteration}] Duplicate prompt (exact hash), skipping.")
            consecutive_rejections += 1
            continue

        sim = SequenceMatcher(None, state.current_best_prompt, candidate).ratio()
        if sim > dup_sim_threshold:
            print(f"[ITER {iteration}] Too similar to current best (sim={sim:.3f}), skipping.")
            consecutive_rejections += 1
            continue

        seen_hashes.add(c_hash)

        # Evaluate candidate
        try:
            candidate_score = _evaluate_prompt(
                candidate, data_split.val,
                schema_info, extraction_client, extraction_cache,
                run_dir, llm_judge_fn, schema_text,
            )
        except Exception as exc:
            print(f"[ITER {iteration}] Evaluation failed: {exc}")
            consecutive_rejections += 1
            continue

        # Accept / reject
        improved = candidate_score.aggregate_score > state.current_best_score
        delta = candidate_score.aggregate_score - state.current_best_score

        if improved:
            reason = f"Improvement: +{delta:.4f} ({state.current_best_score:.4f} -> {candidate_score.aggregate_score:.4f})"
            print(f"[ITER {iteration}] ACCEPTED -- {reason}")
            consecutive_rejections = 0
            last_scoring_result = candidate_score
        else:
            reason = f"Regression: {delta:.4f} ({state.current_best_score:.4f} -> {candidate_score.aggregate_score:.4f})"
            print(f"[ITER {iteration}] REJECTED -- {reason}")
            consecutive_rejections += 1

        # Periodic test-set evaluation for overfitting detection
        test_score_val = None
        if improved and iteration % overfit_interval == 0 and data_split.test:
            test_result = _evaluate_prompt(
                candidate if improved else state.current_best_prompt,
                data_split.test, schema_info, extraction_client,
                extraction_cache, run_dir, llm_judge_fn, schema_text,
            )
            test_score_val = test_result.aggregate_score
            gap = candidate_score.aggregate_score - test_score_val
            if gap > overfit_gap:
                print(f"[ITER {iteration}] WARNING Overfitting detected: val={candidate_score.aggregate_score:.3f}, test={test_score_val:.3f}, gap={gap:.3f}")

        record = PromptRecord(
            iteration=iteration,
            prompt=candidate,
            validation_score=candidate_score.aggregate_score,
            test_score=test_score_val,
            accepted=improved,
            reason=reason,
            prompt_hash=c_hash,
            field_scores_summary={
                k: v for k, v in candidate_score.subtree_scores.items()
            },
        )
        state.add_iteration(record)
        budget.sync_from_logger(call_logger.total_tokens, call_logger.total_cost)
        state.budget_used = budget.to_dict()
        state.save(state_path)

        print(f"[BUDGET] {budget.remaining_summary()}")

    # -- Final test-set evaluation ----------------------------------------
    print("\n[FINAL] Evaluating best prompt on test set...")
    if data_split.test:
        final_test = _evaluate_prompt(
            state.current_best_prompt, data_split.test,
            schema_info, extraction_client, extraction_cache,
            run_dir, llm_judge_fn, schema_text,
        )
        state.test_score_final = final_test.aggregate_score
        print(f"[FINAL] Test score: {final_test.aggregate_score:.4f}")

        # Also evaluate seed on test for comparison
        seed_test = _evaluate_prompt(
            state.seed_prompt, data_split.test,
            schema_info, extraction_client, extraction_cache,
            run_dir, llm_judge_fn, schema_text,
        )
        state.test_score_seed = seed_test.aggregate_score
        print(f"[FINAL] Seed test score: {seed_test.aggregate_score:.4f}")
    else:
        print("[FINAL] No test documents available.")

    state.status = "completed"
    state.budget_used = budget.to_dict()
    state.save(state_path)

    extraction_cache.close()

    usage = call_logger.get_usage_summary()
    print(f"\n[DONE] Run {state.run_id} completed.")
    print(f"  Best val score: {state.current_best_score:.4f}")
    print(f"  Total LLM calls: {usage['total_calls']}")
    print(f"  Total tokens: {usage['total_tokens']}")
    print(f"  Total cost: ${usage['total_cost_usd']:.4f}")

    return state


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _evaluate_prompt(
    prompt_text: str,
    documents: list[Document],
    schema_info: SchemaInfo,
    extraction_client: LLMClient,
    extraction_cache: ExtractionCache,
    cache_dir: Path,
    llm_judge_fn: Any,
    schema_text: str,
) -> ScoringResult:
    """Evaluate a prompt across a set of documents and return aggregate score."""
    all_field_scores: dict[str, Any] = {}
    total_score = 0.0

    rendered_prompt = prompt_text.replace("{schema}", schema_text)

    for doc in documents:
        # Check extraction cache first
        cached = extraction_cache.get(doc.pdf_path, prompt_text)
        if cached is not None:
            predicted = cached
        else:
            try:
                predicted = extraction_client.extract(rendered_prompt, doc.pdf_path)
            except Exception as exc:
                print(f"  [WARN] Extraction failed for {doc.doc_id}: {exc}")
                predicted = {}
            extraction_cache.put(doc.pdf_path, prompt_text, predicted)

        result = score(
            predicted=predicted,
            gold=doc.gold_data,
            schema=schema_info.raw_schema,
            cache_dir=cache_dir,
            llm_judge_fn=llm_judge_fn,
        )

        total_score += result.aggregate_score
        for path, fs in result.field_scores.items():
            key = f"{doc.doc_id}::{path}"
            all_field_scores[key] = fs

    n_docs = max(len(documents), 1)
    avg_score = total_score / n_docs

    # Build aggregate result
    from src.scoring.scorer import ScoringResult as SR

    return SR(
        aggregate_score=avg_score,
        aggregate_precision=avg_score,
        aggregate_recall=avg_score,
        aggregate_f1=avg_score,
        field_scores=all_field_scores,
        subtree_scores=_avg_subtree_scores(all_field_scores),
        total_fields=len(all_field_scores),
        matched_fields=sum(1 for fs in all_field_scores.values() if fs.passed),
        missing_fields=sum(1 for fs in all_field_scores.values()
                         if fs.predicted_value is None and fs.gold_value is not None),
        extra_fields=0,
    )


def _avg_subtree_scores(
    field_scores: dict[str, Any],
) -> dict[str, float]:
    """Compute average score per top-level key across all documents."""
    groups: dict[str, list[float]] = {}
    for path, fs in field_scores.items():
        parts = path.split("::")
        field_path = parts[-1] if len(parts) > 1 else parts[0]
        top_key = field_path.split(".")[0].split("[")[0]
        groups.setdefault(top_key, []).append(fs.score)

    return {k: sum(v) / len(v) for k, v in groups.items() if v}


def _make_budget_config(budget: dict) -> BudgetConfig:
    """Create a BudgetConfig from the config dict."""
    return BudgetConfig(
        max_iterations=budget.get("max_iterations", 5),
        max_tokens=budget.get("max_tokens", 100_000),
        max_cost_usd=budget.get("max_cost_usd", 0.50),
        max_wall_clock_seconds=budget.get("max_wall_clock_seconds", 1200),
    )


def _create_llm(llm_cfg: dict, logger: LLMCallLogger) -> LLMClient:
    """Create an LLM client from a config subsection."""
    return create_client(
        provider=llm_cfg["provider"],
        model=llm_cfg["model"],
        api_key_env=llm_cfg["api_key_env"],
        temperature=llm_cfg.get("temperature", 0),
        max_output_tokens=llm_cfg.get("max_output_tokens", 8192),
        logger=logger,
    )


def _make_llm_judge(scoring_client: LLMClient):
    """Create an LLM judge function for semantic metrics."""

    def judge(predicted: Any, gold: Any) -> tuple[float, str]:
        prompt = (
            "Compare the following two values and rate their semantic similarity "
            "on a scale from 0.0 (completely different) to 1.0 (equivalent).\n\n"
            f"Value A (predicted): {json.dumps(predicted, default=str)}\n"
            f"Value B (gold): {json.dumps(gold, default=str)}\n\n"
            "Respond with ONLY a JSON object: "
            '{\"score\": <float>, \"reason\": \"<brief explanation>\"}'
        )
        try:
            result = scoring_client.judge(prompt)
            sc = float(result.get("score", 0))
            reason = str(result.get("reason", "LLM judge"))
            return max(0.0, min(1.0, sc)), reason
        except Exception as exc:
            return 0.0, f"LLM judge error: {exc}"

    return judge


def _init_run_dir(
    output_cfg: dict, schema_name: str, seed_prompt: str,
) -> tuple[Path, Path]:
    """Create/resolve the run directory and return (state_path, run_dir)."""
    import uuid

    base_dir = output_cfg.get("run_dir", "./runs")
    slug = schema_name.replace("/", "_")

    # Check for existing incomplete run
    base_path = Path(base_dir) / slug
    if base_path.exists():
        for child in sorted(base_path.iterdir()):
            state_file = child / "run_state.json"
            if state_file.exists():
                try:
                    st = RunState.load(state_file)
                    if st.status != "completed":
                        return state_file, child
                except Exception:
                    pass

    run_id = uuid.uuid4().hex[:12]
    run_dir = get_run_dir(base_dir, schema_name, run_id)
    return run_dir / "run_state.json", run_dir
