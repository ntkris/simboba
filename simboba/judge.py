"""LLM-based judge for evaluating outputs against expected outcomes."""

from typing import Tuple, Optional

from simboba.utils import LLMClient
from simboba.prompts import build_judge_prompt


def create_judge(model: Optional[str] = None):
    """Create a judge function that uses an LLM.

    Args:
        model: Model to use for judging (e.g., "gpt-4", "claude-sonnet-4-20250514")
               If not specified, uses LLMClient's default.

    Returns:
        A function(inputs, expected, actual) -> (passed: bool, reasoning: str)
    """
    client = LLMClient(model=model)

    def judge(inputs: list, expected_outcome: str, actual_output: str) -> Tuple[bool, str]:
        """Judge whether the actual output meets the expected outcome."""
        prompt = build_judge_prompt(inputs, expected_outcome, actual_output)

        try:
            result = client.generate_json(prompt, max_tokens=1024)
            passed = bool(result.get("passed", False))
            reasoning = result.get("reasoning", "No reasoning provided")
            return passed, reasoning
        except Exception as e:
            # If we can't parse, try to extract intent from raw response
            response_text = str(e)
            passed = "passed" in response_text.lower() and "true" in response_text.lower()
            return passed, f"Could not parse judge response: {response_text}"

    return judge


def create_simple_judge():
    """Create a simple string-matching judge for testing without API calls.

    Returns:
        A function(inputs, expected, actual) -> (passed: bool, reasoning: str)
    """
    def judge(inputs: list, expected_outcome: str, actual_output: str) -> Tuple[bool, str]:
        """Simple judge that checks if expected keywords are in the output."""
        # Extract key terms from expected outcome
        expected_lower = expected_outcome.lower()
        actual_lower = actual_output.lower()

        # Very basic check - just see if main words overlap
        expected_words = set(expected_lower.split())
        actual_words = set(actual_lower.split())

        # Remove common words
        stop_words = {"the", "a", "an", "is", "are", "should", "must", "will", "and", "or", "to", "be"}
        expected_words -= stop_words
        actual_words -= stop_words

        if not expected_words:
            return True, "No specific requirements to check"

        overlap = expected_words & actual_words
        overlap_ratio = len(overlap) / len(expected_words)

        passed = overlap_ratio >= 0.3  # At least 30% overlap
        reasoning = f"Found {len(overlap)}/{len(expected_words)} expected terms in output"

        return passed, reasoning

    return judge
