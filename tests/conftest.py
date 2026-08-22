"""Shared test setup.

Two things every test needs and none should have to remember:

* No network. Uploads archive to Supabase in production; a test suite that
  reaches a live bucket is slow, flaky, and writes junk into real storage.
* No background threads. Batches and match runs are dispatched to threads in
  production, which makes assertions racy. Inline execution keeps them
  deterministic without changing the code under test.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def no_remote_storage(monkeypatch):
    """Uploads write locally only. The archiving path has its own tests."""
    from app import storage
    from app.config import settings

    monkeypatch.setattr(settings, "supabase_url", "", raising=False)
    monkeypatch.setattr(settings, "supabase_key", "", raising=False)
    monkeypatch.setattr(storage, "_client", None, raising=False)
    yield
    monkeypatch.setattr(storage, "_client", None, raising=False)


@pytest.fixture
def inline_threads(monkeypatch):
    """Run dispatched work synchronously.

    Patches the app's own dispatch seam, NOT threading.Thread: that name is
    global, and replacing it also breaks the test client's worker threads,
    which deadlocks every request.
    """
    from app import main as main_mod

    def inline(target, *args):
        target(*args)

    monkeypatch.setattr(main_mod, "run_in_background", inline)
    return inline
