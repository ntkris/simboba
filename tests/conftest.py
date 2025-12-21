"""Pytest fixtures for simboba tests."""

import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from simboba.database import create_db_engine, Base
from simboba.server import create_app, get_db


@pytest.fixture
def db_path(tmp_path):
    """Create a temporary database file."""
    return tmp_path / "test.db"


@pytest.fixture
def client(db_path):
    """Create a test client with isolated database.

    Also sets SIMBOBA_DB_PATH so that the Boba class uses the same database.
    """
    # Set environment variable so Boba class uses the same database
    old_db_path = os.environ.get("SIMBOBA_DB_PATH")
    os.environ["SIMBOBA_DB_PATH"] = str(db_path)

    engine = create_db_engine(db_path)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def override_get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    engine.dispose()

    # Restore original environment
    if old_db_path is not None:
        os.environ["SIMBOBA_DB_PATH"] = old_db_path
    else:
        del os.environ["SIMBOBA_DB_PATH"]
