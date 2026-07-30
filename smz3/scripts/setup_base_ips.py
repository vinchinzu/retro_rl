"""Fetch the SMZ3 combo base IPS into smz3/refs/ (gitignored)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from smz3.rom_builder import ensure_base_ips, load_base_ips  # noqa: E402


def main() -> None:
    path = ensure_base_ips()
    ips = load_base_ips(path)
    print(f"Base IPS ready: {path} ({len(ips)} bytes after gunzip)")


if __name__ == "__main__":
    main()
