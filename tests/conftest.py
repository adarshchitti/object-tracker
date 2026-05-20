"""Shared pytest fixtures for the API test suite."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from api import storage
from api.db import Base, get_db
from api.main import app

FIXTURE_VIDEO = Path(__file__).parent / "fixtures" / "sample_short.mp4"


@pytest.fixture
def storage_root(tmp_path, monkeypatch) -> Path:
    """Redirect api.storage paths into a tmp directory for the duration of a test."""
    root = tmp_path / "storage"
    tasks = root / "tasks"
    tasks.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(storage, "STORAGE_ROOT", root)
    monkeypatch.setattr(storage, "TASKS_ROOT", tasks)
    return root


@pytest.fixture
def db_session(tmp_path) -> Iterator[Session]:
    """Yield a session bound to a fresh temp SQLite DB."""
    db_url = f"sqlite:///{tmp_path}/test.db"
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = TestSession()
    session._TestSession = TestSession  # stash factory for the client override
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(db_session, storage_root, monkeypatch) -> Iterator[TestClient]:
    """TestClient with get_db overridden to point at the per-test SQLite DB.

    Stubs process_video so background tasks don't kick off the heavy pipeline
    during fast API tests.
    """
    TestSession = db_session._TestSession

    def _override_get_db():
        s = TestSession()
        try:
            yield s
        finally:
            s.close()

    def _noop_process_video(*args, **kwargs):
        return None

    monkeypatch.setattr("api.routes.process_video", _noop_process_video)
    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture
def sample_video_bytes() -> bytes:
    """Bytes of the small fixture video used for end-to-end tests."""
    if not FIXTURE_VIDEO.exists():
        pytest.skip(f"Fixture video missing at {FIXTURE_VIDEO}")
    return FIXTURE_VIDEO.read_bytes()
