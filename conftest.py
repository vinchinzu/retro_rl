"""Repo-level test stubs for optional emulator dependencies."""

from __future__ import annotations

import sys
import types
from pathlib import Path

# Games live under snes/ and nes/ but keep package names (alttp, smb, …).
# Nested package roots (harvest, hals_golf, …) are discovered by layout after
# the stable_retro stub is installed (package import may touch the emulator).
_ROOT = Path(__file__).resolve().parent
for _extra in (_ROOT, _ROOT / "snes", _ROOT / "nes"):
    if _extra.is_dir():
        _text = str(_extra)
        if _text not in sys.path:
            sys.path.insert(0, _text)


try:
    import stable_retro  # noqa: F401
except ModuleNotFoundError:
    integrations = types.SimpleNamespace(
        add_custom_path=lambda *_args, **_kwargs: None,
        CUSTOM_ONLY="CUSTOM_ONLY",
        ALL="ALL",
    )
    sys.modules["stable_retro"] = types.SimpleNamespace(
        RetroEnv=object,
        State=types.SimpleNamespace(NONE="NONE"),
        Actions=types.SimpleNamespace(ALL="ALL"),
        data=types.SimpleNamespace(Integrations=integrations),
        make=lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("stable_retro stub cannot create environments")
        ),
    )


from retro_harness.repo import ensure_import_paths  # noqa: E402

ensure_import_paths(root=_ROOT)


def pytest_collection_modifyitems(items):
    """Attach ownership markers from stable repository boundaries.

    Game suites are intentionally discovered only by the explicit game tier,
    while this marker makes omissions and CI reporting visible.
    """
    import pytest

    for item in items:
        try:
            relative = Path(str(item.path)).resolve().relative_to(_ROOT)
        except (OSError, ValueError):
            continue
        if relative.parts and relative.parts[0] in {"nes", "snes"}:
            item.add_marker(pytest.mark.game)
