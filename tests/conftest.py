"""Shared pytest fixtures providing clean, temporary database isolation."""

import pytest
import tempfile
from pathlib import Path

from daemon.app import app, repository, vector_store, assistant, enricher
from daemon.db.repository import DatabaseRepository
from daemon.rag.vector_store import VectorStore


@pytest.fixture(autouse=True)
def isolated_test_database(tmp_path, monkeypatch):
    """
    Ensure every test uses a temporary, isolated SQLite database
    so the live foresight.db is never polluted.
    """
    test_db = tmp_path / "test_foresight_isolated.db"
    orig_repo_db = repository.db_path
    orig_vs_db = vector_store.db_path

    # Point repository and vector_store to temporary test database
    repository.db_path = str(test_db)
    vector_store.db_path = str(test_db)

    yield test_db

    # Restore original paths after test
    repository.db_path = orig_repo_db
    vector_store.db_path = orig_vs_db
