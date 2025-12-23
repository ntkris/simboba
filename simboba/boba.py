"""
Boba - Simple eval tracking.

Usage:
    from simboba import Boba

    boba = Boba()

    # Single eval
    boba.eval(input="Hello", output="Hi there!", expected="Should greet the user")

    # Dataset eval
    boba.run(agent_fn, dataset="my-dataset")
"""

from datetime import datetime
from typing import Callable, Optional

from simboba import storage

# Type alias for metadata checker function
MetadataChecker = Callable[[Optional[dict], Optional[dict]], bool]


class Boba:
    """Simple eval tracking."""

    def __init__(self):
        """Initialize Boba."""
        self._warned_simple_judge = False

    def _get_judge(self, warn: bool = True):
        """Get the judge function.

        Args:
            warn: Whether to print a warning if falling back to simple judge

        Returns:
            Judge function
        """
        try:
            from simboba.judge import create_judge
            model = storage.get_setting("model")
            return create_judge(model=model)
        except Exception:
            if warn and not self._warned_simple_judge:
                print("\n  No API key found. Using simple keyword-matching judge.")
                print("   For better results, set ANTHROPIC_API_KEY in your environment.")
                print("   See: https://console.anthropic.com/\n")
                self._warned_simple_judge = True
            from simboba.judge import create_simple_judge
            return create_simple_judge()

    # Fixed ID for ad-hoc evals (not part of a dataset)
    ADHOC_DATASET_ID = "_adhoc"

    def eval(
        self,
        input: str,
        output: str,
        expected: str,
        name: Optional[str] = None,
        expected_metadata: Optional[dict] = None,
        actual_metadata: Optional[dict] = None,
        metadata_checker: Optional[MetadataChecker] = None,
    ) -> dict:
        """
        Evaluate a single input/output pair.

        Args:
            input: The input message
            output: The actual output from your agent
            expected: What the output should do/contain
            name: Optional name for this eval
            expected_metadata: Expected metadata (citations, tool_calls, etc.)
            actual_metadata: Actual metadata from the agent response
            metadata_checker: Optional function(expected, actual) -> bool for
                            deterministic metadata checking. If not provided,
                            metadata is passed to the LLM judge.

        Returns:
            dict with: passed, reasoning, run_id
        """
        # Create run record
        run = {
            "dataset_id": self.ADHOC_DATASET_ID,
            "dataset_name": "_adhoc",
            "eval_name": name or "single-eval",
            "status": "running",
            "total": 1,
            "passed": 0,
            "failed": 0,
            "started_at": datetime.now().isoformat(),
            "results": {},
        }

        # Judge the output (always pass metadata to LLM when provided)
        judge_fn = self._get_judge()
        inputs = [{"role": "user", "message": input}]
        output_passed, reasoning = judge_fn(
            inputs, expected, output,
            expected_metadata=expected_metadata,
            actual_metadata=actual_metadata,
        )

        # If metadata_checker provided, run deterministic check as additional gate
        if metadata_checker is not None:
            metadata_passed = metadata_checker(expected_metadata, actual_metadata)
            passed = output_passed and metadata_passed
        else:
            metadata_passed = None
            passed = output_passed

        # Create result
        case_id = storage.generate_id()
        run["results"][case_id] = {
            "case_id": case_id,
            "inputs": inputs,
            "expected_outcome": expected,
            "passed": passed,
            "actual_output": output,
            "judgment": "PASS" if passed else "FAIL",
            "reasoning": reasoning,
            "expected_metadata": expected_metadata,
            "actual_metadata": actual_metadata,
            "metadata_passed": metadata_passed,
            "created_at": datetime.now().isoformat(),
        }

        # Update run
        run["status"] = "completed"
        run["passed"] = 1 if passed else 0
        run["failed"] = 0 if passed else 1
        run["score"] = 100.0 if passed else 0.0
        run["completed_at"] = datetime.now().isoformat()

        # Save run using the ad-hoc dataset ID
        saved_run = storage.save_run(self.ADHOC_DATASET_ID, run)

        return {
            "passed": passed,
            "reasoning": reasoning,
            "run_id": saved_run["filename"],
        }

    def run(
        self,
        agent: Callable[[str], str],
        dataset: str,
        name: Optional[str] = None,
        metadata_checker: Optional[MetadataChecker] = None,
    ) -> dict:
        """
        Run an agent against a dataset.

        Args:
            agent: Function that takes a message string and returns a response string
            dataset: Name of the dataset to run against
            name: Optional name for this run
            metadata_checker: Optional function(expected, actual) -> bool for
                            deterministic metadata checking. If not provided,
                            metadata is passed to the LLM judge.

        Returns:
            dict with: passed, failed, total, score, run_id, regressions, fixes
        """
        # Load dataset
        ds = storage.get_dataset(dataset)
        if not ds:
            raise ValueError(f"Dataset '{dataset}' not found")

        dataset_id = ds["id"]
        dataset_name = ds["name"]

        cases = ds.get("cases", [])
        if not cases:
            raise ValueError(f"Dataset '{dataset}' has no cases")

        # Create run record
        run = {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "eval_name": name or f"eval-{dataset_name}",
            "status": "running",
            "total": len(cases),
            "passed": 0,
            "failed": 0,
            "started_at": datetime.now().isoformat(),
            "results": {},
        }

        # Save initial run state (using dataset ID)
        run = storage.save_run(dataset_id, run)

        # Get judge
        judge_fn = self._get_judge()

        # Run each case
        passed_count = 0
        failed_count = 0

        for case in cases:
            case_id = case.get("id", storage.generate_id())

            # Get input message (last user message)
            inputs = case.get("inputs", [])
            if inputs and len(inputs) > 0:
                last_message = inputs[-1].get("message", "")
            else:
                last_message = ""

            # Call agent
            try:
                output = agent(last_message)
                error_message = None
            except Exception as e:
                output = None
                error_message = str(e)

            # Judge if no error
            expected_metadata = case.get("expected_metadata")
            actual_metadata = None  # TODO: agent could return metadata in future

            if error_message:
                passed = False
                reasoning = f"Error: {error_message}"
                metadata_passed = None
            else:
                # Always pass metadata to LLM when provided
                output_passed, reasoning = judge_fn(
                    inputs,
                    case.get("expected_outcome", ""),
                    output,
                    expected_metadata=expected_metadata,
                    actual_metadata=actual_metadata,
                )

                # If metadata_checker provided, run deterministic check as additional gate
                if metadata_checker is not None:
                    metadata_passed = metadata_checker(expected_metadata, actual_metadata)
                    passed = output_passed and metadata_passed
                else:
                    metadata_passed = None
                    passed = output_passed

            # Create result
            run["results"][case_id] = {
                "case_id": case_id,
                "passed": passed,
                "actual_output": str(output) if output else None,
                "judgment": "PASS" if passed else "FAIL",
                "reasoning": reasoning,
                "error_message": error_message,
                "expected_metadata": expected_metadata,
                "actual_metadata": actual_metadata,
                "metadata_passed": metadata_passed,
                "created_at": datetime.now().isoformat(),
                "case": {
                    "id": case_id,
                    "name": case.get("name"),
                    "inputs": inputs,
                    "expected_outcome": case.get("expected_outcome", ""),
                },
            }

            if passed:
                passed_count += 1
            else:
                failed_count += 1

            # Update run incrementally
            run["passed"] = passed_count
            run["failed"] = failed_count

            # Save run after each case (atomic write)
            storage.save_run(dataset_id, run)

            # Print progress
            status = "+" if passed else "x"
            case_name = case.get("name") or f"Case {case_id[:8]}"
            print(f"  {status} {case_name}")

        # Finalize run
        run["status"] = "completed"
        run["score"] = (passed_count / len(cases) * 100) if cases else 0.0
        run["completed_at"] = datetime.now().isoformat()

        storage.save_run(dataset_id, run)

        # Compare to baseline (using dataset ID)
        baseline = storage.get_baseline(dataset_id)
        comparison = storage.compare_run_to_baseline(run, baseline)

        # Print results
        print(f"\nResults: {passed_count}/{len(cases)} passed ({run['score']:.1f}%)")

        if comparison["has_baseline"]:
            if comparison["regressions"]:
                print(f"\n  REGRESSIONS: {len(comparison['regressions'])} cases now failing")
                for case_id in comparison["regressions"][:5]:
                    result = run["results"].get(case_id, {})
                    case_name = result.get("case", {}).get("name") or case_id[:8]
                    print(f"    - {case_name}")
                if len(comparison["regressions"]) > 5:
                    print(f"    ... and {len(comparison['regressions']) - 5} more")

            if comparison["fixes"]:
                print(f"\n  FIXES: {len(comparison['fixes'])} cases now passing")
        else:
            print("\n  No baseline found. Run 'boba baseline' to save current results as baseline.")

        return {
            "passed": passed_count,
            "failed": failed_count,
            "total": len(cases),
            "score": run["score"],
            "run_id": run["filename"],
            "regressions": comparison["regressions"],
            "fixes": comparison["fixes"],
        }
