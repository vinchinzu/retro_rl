"""Repo-level test stubs for optional emulator dependencies."""

from __future__ import annotations

import sys
import types


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
