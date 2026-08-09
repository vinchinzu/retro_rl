"""Core-install import boundaries for optional subsystems."""

from __future__ import annotations

import subprocess
import sys


def test_fighters_facade_does_not_import_ml_stack() -> None:
    code = """
import sys
import types
integrations = types.SimpleNamespace(
    add_custom_path=lambda *_args, **_kwargs: None,
    CUSTOM_ONLY='CUSTOM_ONLY',
    ALL='ALL',
)
sys.modules['stable_retro'] = types.SimpleNamespace(
    RetroEnv=object,
    State=types.SimpleNamespace(NONE='NONE'),
    Actions=types.SimpleNamespace(ALL='ALL'),
    data=types.SimpleNamespace(Integrations=integrations),
)
import retro_harness.fighters
for name in ('gymnasium', 'cv2', 'torch', 'stable_baselines3'):
    assert name not in sys.modules, name
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
