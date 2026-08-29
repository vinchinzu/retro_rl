"""FCEUX HappyLee oracle checkpoints (timing source of truth for 8-3 sync).

Use original HappyLee #1715M under FCEUX 2.6.6 as the only timing oracle.
stable-retro/fceumm is the *target* runtime for control-relative replay.

See ``extract_fceux_checkpoints`` and ``compare_fceumm_chain``.

Lazy exports avoid runpy warnings when modules are executed as ``-m``.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ORACLE_DIR",
    "ORACLE_EVIDENCE_DIR",
    "DEFAULT_FM2",
    "DEFAULT_ROM",
    "extract_checkpoints",
    "run_fceux_dump",
    "compare_chain_to_oracle",
]


def __getattr__(name: str) -> Any:
    if name in {
        "ORACLE_DIR",
        "ORACLE_EVIDENCE_DIR",
        "DEFAULT_FM2",
        "DEFAULT_ROM",
        "extract_checkpoints",
        "run_fceux_dump",
    }:
        from smb.tas.oracle import extract_fceux_checkpoints as m

        return getattr(m, name)
    if name == "compare_chain_to_oracle":
        from smb.tas.oracle.compare_fceumm_chain import compare_chain_to_oracle

        return compare_chain_to_oracle
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
