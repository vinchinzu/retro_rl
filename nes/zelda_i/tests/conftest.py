"""zelda_i test-wide fixtures."""

from __future__ import annotations

import pytest

from zelda_i.walk import live_env


@pytest.fixture(autouse=True)
def _unbind_live_env():
    """Leave no emulator bound to ``live_env`` after a test.

    A ROM test's stage runner binds the process's env and then closes it; a
    later test's ``room_step`` falls back to ``live_env.current()`` and read
    that dead core (a segfault in ``test_level7_pond`` after the ROM tier's
    earlier tests, 2026-09-28).
    """
    yield
    live_env.bind(None)
