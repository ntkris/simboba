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

from simboba.database import init_db, get_session_factory
from simboba.models import Dataset, EvalRun, EvalResult, Settings


class Boba:
    """Simple eval tracking."""

    def __init__(self):
        """Initialize Boba and connect to database."""
        init_db()
        self._session_factory = get_session_factory()

    def _get_session(self):
        return self._session_factory()

    def _get_judge(self, db):
        """Get the judge function."""
        try:
            from simboba.judge import create_judge
            model = Settings.get(db, "model")
            return create_judge(model=model)
        except Exception:
            from simboba.judge import create_simple_judge
            return create_simple_judge()

    def eval(
        self,
        input: str,
        output: str,
        expected: str,
        name: Optional[str] = None,
    ) -> dict:
        """
        Evaluate a single input/output pair.

        Args:
            input: The input message
            output: The actual output from your agent
            expected: What the output should do/contain
            name: Optional name for this eval

        Returns:
            dict with: passed, reasoning, run_id
        """
        db = self._get_session()

        try:
            # Create run record
            run = EvalRun(
                dataset_id=None,
                eval_name=name or "single-eval",
                status="running",
                total=1,
            )
            db.add(run)
            db.flush()

            # Judge the result
            judge_fn = self._get_judge(db)
            inputs = [{"role": "user", "message": input}]
            passed, reasoning = judge_fn(inputs, expected, output)

            # Create result record
            result = EvalResult(
                run_id=run.id,
                case_id=None,
                inputs=inputs,
                expected_outcome=expected,
                passed=passed,
                actual_output=output,
                judgment="PASS" if passed else "FAIL",
                reasoning=reasoning,
            )
            db.add(result)

            # Update run
            run.status = "completed"
            run.passed = 1 if passed else 0
            run.failed = 0 if passed else 1
            run.score = 100.0 if passed else 0.0
            run.completed_at = datetime.utcnow()

            db.commit()

            return {
                "passed": passed,
                "reasoning": reasoning,
                "run_id": run.id,
            }

        finally:
            db.close()

    def run(
        self,
        agent: Callable[[str], str],
        dataset: str,
        name: Optional[str] = None,
    ) -> dict:
        """
        Run an agent against a dataset.

        Args:
            agent: Function that takes a message string and returns a response string
            dataset: Name of the dataset to run against
            name: Optional name for this run

        Returns:
            dict with: passed, failed, total, score, run_id
        """
        db = self._get_session()

        try:
            # Load dataset
            ds = db.query(Dataset).filter(Dataset.name == dataset).first()
            if not ds:
                raise ValueError(f"Dataset '{dataset}' not found")

            cases = ds.cases
            if not cases:
                raise ValueError(f"Dataset '{dataset}' has no cases")

            # Create run record
            run = EvalRun(
                dataset_id=ds.id,
                eval_name=name or f"eval-{dataset}",
                status="running",
                total=len(cases),
            )
            db.add(run)
            db.flush()

            # Get judge
            judge_fn = self._get_judge(db)

            # Run each case
            passed_count = 0
            failed_count = 0

            for case in cases:
                # Get input message (last user message)
                inputs = case.inputs
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
                if error_message:
                    passed = False
                    reasoning = f"Error: {error_message}"
                else:
                    passed, reasoning = judge_fn(inputs, case.expected_outcome, output)

                # Create result record
                result = EvalResult(
                    run_id=run.id,
                    case_id=case.id,
                    passed=passed,
                    actual_output=str(output) if output else None,
                    judgment="PASS" if passed else "FAIL",
                    reasoning=reasoning,
                    error_message=error_message,
                )
                db.add(result)

                if passed:
                    passed_count += 1
                else:
                    failed_count += 1

                # Update run incrementally
                run.passed = passed_count
                run.failed = failed_count
                db.flush()

                # Print progress
                status = "✓" if passed else "✗"
                case_name = case.name or f"Case {case.id}"
                print(f"  {status} {case_name}")

            # Finalize run
            run.status = "completed"
            run.score = (passed_count / len(cases) * 100) if cases else 0.0
            run.completed_at = datetime.utcnow()

            db.commit()

            print(f"\nResults: {passed_count}/{len(cases)} passed ({run.score:.1f}%)")

            return {
                "passed": passed_count,
                "failed": failed_count,
                "total": len(cases),
                "score": run.score,
                "run_id": run.id,
            }

        finally:
            db.close()
