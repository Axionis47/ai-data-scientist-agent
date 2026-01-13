#!/usr/bin/env python3
"""
Eval runner for golden scenario regression testing.

Runs scenarios from evals/scenarios.json and checks expected outputs.
Uses fake LLM clients for deterministic, offline testing.

Usage:
    python scripts/run_evals.py                    # Run all scenarios
    python scripts/run_evals.py --scenario route_causal_effect  # Run specific scenario
    python scripts/run_evals.py --verbose          # Show detailed output
    python scripts/run_evals.py --json             # Output JSON results
"""

import argparse
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from packages.agent.fake_clients import FakeEmbeddingsClient, FakeLLMClient
from packages.agent.graph import run_agent


@dataclass
class EvalResult:
    """Result of a single scenario evaluation."""
    scenario_id: str
    passed: bool
    checks: list[dict] = field(default_factory=list)
    error: str | None = None


def setup_test_fixtures(tmpdir: Path, use_causal_dataset: bool = False, skip_dataset: bool = False):
    """Set up test fixtures (doc and optionally dataset)."""
    storage_dir = tmpdir / "contexts"
    storage_dir.mkdir(parents=True)
    datasets_dir = tmpdir / "datasets"
    datasets_dir.mkdir()

    # Create a minimal doc for RAG
    doc_id = "eval-doc"
    doc_dir = storage_dir / doc_id
    doc_dir.mkdir()

    # Create fake embeddings with correct dimension (256 for FakeEmbeddingsClient)
    import random
    random.seed(42)
    embedding = [random.uniform(-1, 1) for _ in range(256)]

    (doc_dir / "chunks.json").write_text(json.dumps([
        {"chunk_index": 0, "text": "This is a test dataset with customer data including age, income, and purchase history."}
    ]))
    (doc_dir / "embeddings.json").write_text(json.dumps([
        {"chunk_index": 0, "text": "Test chunk", "embedding": embedding}
    ]))

    dataset_id = None
    if not skip_dataset:
        # Create dataset
        if use_causal_dataset:
            # Use the balanced causal dataset for evals (passes balance check)
            fixture_path = PROJECT_ROOT / "evals/balanced_causal.csv"
            if fixture_path.exists():
                dataset_id = "eval-causal-dataset"
                ds_dir = datasets_dir / dataset_id
                ds_dir.mkdir()
                shutil.copy(fixture_path, ds_dir / "data.csv")
                # Create metadata
                (ds_dir / "metadata.json").write_text(json.dumps({
                    "dataset_id": dataset_id,
                    "filename": "data.csv",
                    "row_count": 40,
                    "column_count": 5,
                    "columns": ["treatment", "outcome", "age", "income", "prior_purchases"],
                    "inferred_types": {
                        "treatment": "int64",
                        "outcome": "float64",
                        "age": "int64",
                        "income": "float64",
                        "prior_purchases": "int64"
                    }
                }))
        else:
            # Create a simple dataset
            dataset_id = "eval-dataset"
            ds_dir = datasets_dir / dataset_id
            ds_dir.mkdir()
            (ds_dir / "data.csv").write_text("age,income,region\n25,50000,North\n35,75000,South\n45,60000,East\n")
            (ds_dir / "metadata.json").write_text(json.dumps({
                "dataset_id": dataset_id,
                "filename": "data.csv",
                "row_count": 3,
                "column_count": 3,
                "columns": ["age", "income", "region"],
                "inferred_types": {"age": "int64", "income": "int64", "region": "object"}
            }))

    return storage_dir, datasets_dir, doc_id, dataset_id


def check_expectation(key: str, expected_value, actual_value) -> dict:
    """Check a single expectation and return result dict."""
    passed = actual_value == expected_value
    return {
        "check": key,
        "expected": expected_value,
        "actual": actual_value,
        "passed": passed
    }


def run_scenario(scenario: dict) -> EvalResult:  # noqa: PLR0915
    """Run a single scenario and check expectations."""
    scenario_id = scenario["id"]
    question = scenario["question"]
    expected = scenario.get("expected", {})
    use_causal = scenario.get("use_causal_dataset", False)
    skip_dataset = scenario.get("skip_dataset", False)
    confirmations = scenario.get("causal_confirmations")
    spec_override = scenario.get("causal_spec_override")

    # Default spec override for causal scenarios with causal dataset
    if use_causal and not spec_override:
        spec_override = {
            "treatment": "treatment",
            "outcome": "outcome",
            "confounders": ["age", "income", "prior_purchases"]
        }

    checks = []

    try:
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            storage_dir, datasets_dir, doc_id, dataset_id = setup_test_fixtures(
                tmpdir, use_causal_dataset=use_causal, skip_dataset=skip_dataset
            )

            # Run agent
            result = run_agent(
                question=question,
                doc_id=doc_id,
                embeddings_client=FakeEmbeddingsClient(),
                llm_client=FakeLLMClient(),
                storage_dir=storage_dir,
                dataset_id=dataset_id,
                datasets_dir=datasets_dir,
                causal_spec_override=spec_override,
                causal_confirmations=confirmations,
            )

            # Check route (final state)
            if "route" in expected:
                checks.append(check_expectation(
                    "route", expected["route"], result.get("route")
                ))

            # Check initial_route from trace events (the router's decision before any gate)
            if "initial_route" in expected:
                trace_events = result.get("trace_events", [])
                routed_event = next((e for e in trace_events if e.get("event_type") == "ROUTED"), None)
                actual_initial = routed_event.get("payload", {}).get("route") if routed_event else None
                checks.append(check_expectation(
                    "initial_route", expected["initial_route"], actual_initial
                ))

            # Check route confidence minimum
            if "route_confidence_min" in expected:
                actual = result.get("route_confidence", 0)
                passed = actual >= expected["route_confidence_min"]
                checks.append({
                    "check": "route_confidence_min",
                    "expected": f">= {expected['route_confidence_min']}",
                    "actual": actual,
                    "passed": passed
                })

            # Check playbook
            if "playbook" in expected:
                checks.append(check_expectation(
                    "playbook", expected["playbook"], result.get("playbook")
                ))

            # Check causal readiness status (exact match)
            if "causal_readiness_status" in expected:
                checks.append(check_expectation(
                    "causal_readiness_status",
                    expected["causal_readiness_status"],
                    result.get("causal_readiness_status")
                ))

            # Check causal readiness status (any of list)
            if "causal_readiness_status_in" in expected:
                actual = result.get("causal_readiness_status")
                allowed = expected["causal_readiness_status_in"]
                passed = actual in allowed
                checks.append({
                    "check": "causal_readiness_status_in",
                    "expected": f"one of {allowed}",
                    "actual": actual,
                    "passed": passed
                })

            # Check for causal estimate
            if "has_causal_estimate" in expected:
                has_estimate = result.get("causal_estimate") is not None
                checks.append(check_expectation(
                    "has_causal_estimate", expected["has_causal_estimate"], has_estimate
                ))

            # Check estimate method
            if "estimate_method" in expected:
                estimate = result.get("causal_estimate", {})
                actual_method = estimate.get("method") if estimate else None
                checks.append(check_expectation(
                    "estimate_method", expected["estimate_method"], actual_method
                ))

            # Check artifact types include
            if "artifact_types_include" in expected:
                actual_types = [a.get("type") for a in result.get("artifacts", [])]
                for expected_type in expected["artifact_types_include"]:
                    passed = expected_type in actual_types
                    checks.append({
                        "check": f"artifact_type_{expected_type}",
                        "expected": f"includes {expected_type}",
                        "actual": actual_types,
                        "passed": passed
                    })

            # Check trace event types include
            if "trace_event_types_include" in expected:
                actual_types = [e.get("event_type") for e in result.get("trace_events", [])]
                for expected_type in expected["trace_event_types_include"]:
                    passed = expected_type in actual_types
                    checks.append({
                        "check": f"trace_event_{expected_type}",
                        "expected": f"includes {expected_type}",
                        "actual": actual_types,
                        "passed": passed
                    })

            # Check LLM response minimum length
            if "llm_response_min_length" in expected:
                response = result.get("llm_response", "")
                actual_len = len(response) if response else 0
                passed = actual_len >= expected["llm_response_min_length"]
                checks.append({
                    "check": "llm_response_min_length",
                    "expected": f">= {expected['llm_response_min_length']}",
                    "actual": actual_len,
                    "passed": passed
                })

        all_passed = all(c["passed"] for c in checks)
        return EvalResult(scenario_id=scenario_id, passed=all_passed, checks=checks)

    except Exception as e:
        return EvalResult(scenario_id=scenario_id, passed=False, error=str(e))


def load_scenarios(scenario_filter: str | None = None) -> list[dict]:
    """Load scenarios from evals/scenarios.json."""
    scenarios_path = PROJECT_ROOT / "evals" / "scenarios.json"
    with open(scenarios_path) as f:
        data = json.load(f)

    scenarios = data.get("scenarios", [])
    if scenario_filter:
        scenarios = [s for s in scenarios if s["id"] == scenario_filter]
    return scenarios


def main():
    parser = argparse.ArgumentParser(description="Run eval scenarios for regression testing")
    parser.add_argument("--scenario", help="Run only a specific scenario by ID")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    scenarios = load_scenarios(args.scenario)
    if not scenarios:
        print("❌ No scenarios found" + (f" matching '{args.scenario}'" if args.scenario else ""))
        sys.exit(1)

    results: list[EvalResult] = []
    passed_count = 0
    failed_count = 0

    if not args.json:
        print(f"\n🧪 Running {len(scenarios)} eval scenario(s)...\n")

    for scenario in scenarios:
        result = run_scenario(scenario)
        results.append(result)

        if result.passed:
            passed_count += 1
            if not args.json:
                print(f"  ✅ {result.scenario_id}")
        else:
            failed_count += 1
            if not args.json:
                print(f"  ❌ {result.scenario_id}")
                if result.error:
                    print(f"     Error: {result.error}")
                else:
                    for check in result.checks:
                        if not check["passed"]:
                            print(f"     - {check['check']}: expected {check['expected']}, got {check['actual']}")

    if args.json:
        output = {
            "total": len(results),
            "passed": passed_count,
            "failed": failed_count,
            "results": [
                {
                    "scenario_id": r.scenario_id,
                    "passed": r.passed,
                    "checks": r.checks,
                    "error": r.error
                }
                for r in results
            ]
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"\n{'=' * 40}")
        print(f"Results: {passed_count}/{len(results)} passed")
        if failed_count > 0:
            print(f"❌ {failed_count} scenario(s) failed")
            sys.exit(1)
        else:
            print("✅ All scenarios passed!")
            sys.exit(0)


if __name__ == "__main__":
    main()

