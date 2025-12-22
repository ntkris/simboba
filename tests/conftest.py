"""Pytest fixtures for simboba tests."""

import os
import pytest
from pathlib import Path
from dotenv import load_dotenv
from fastapi.testclient import TestClient

from simboba.server import create_app
from simboba import storage

# Load .env file for API keys
load_dotenv()


@pytest.fixture
def evals_dir(tmp_path):
    """Create a temporary boba-evals directory."""
    evals = tmp_path / "boba-evals"
    evals.mkdir()
    (evals / "datasets").mkdir()
    (evals / "baselines").mkdir()
    (evals / "runs").mkdir()
    (evals / "files").mkdir()
    return evals


@pytest.fixture
def client(evals_dir, monkeypatch):
    """Create a test client with isolated storage.

    Monkeypatches the storage module to use the temp directory.
    """
    # Monkeypatch storage.get_evals_dir to return our temp directory
    monkeypatch.setattr(storage, "get_evals_dir", lambda: evals_dir)

    app = create_app()

    with TestClient(app) as test_client:
        yield test_client
