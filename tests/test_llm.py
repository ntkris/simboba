"""Tests for LLM integration with LiteLLM.

Run with: pytest tests/test_llm.py -v
"""

from simboba import storage
from simboba.utils import LLMClient


def test_llm_connection(tmp_path, monkeypatch):
    """Test that LLM API calls work with the configured model."""
    # Set up temp storage
    evals_dir = tmp_path / "boba-evals"
    evals_dir.mkdir()
    (evals_dir / "datasets").mkdir()
    (evals_dir / "baselines").mkdir()
    (evals_dir / "runs").mkdir()
    (evals_dir / "files").mkdir()

    monkeypatch.setattr(storage, "get_evals_dir", lambda: evals_dir)

    model = storage.get_setting("model") or LLMClient.DEFAULT_MODEL

    print(f"\nTesting model: {model}")

    client = LLMClient(model=model)
    response = client.generate("Reply with exactly one word: hello")

    assert response is not None
    assert len(response.strip()) > 0
    print(f"Response: {response.strip()}")
