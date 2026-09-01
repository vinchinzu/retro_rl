#!/usr/bin/env python3
"""Read-only health check: is this retro_rl checkout worth driving?"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parents[4], Path.cwd(), *here.parents]:
        pyproject = candidate / "pyproject.toml"
        if pyproject.is_file() and 'name = "retro_rl"' in pyproject.read_text(
            encoding="utf-8"
        ):
            return candidate
    raise SystemExit("doctor: could not find retro_rl pyproject.toml")


def _python_ok() -> tuple[bool, str]:
    version = ".".join(str(part) for part in sys.version_info[:3])
    return sys.version_info[:2] == (3, 12), version


def main() -> int:
    root = _repo_root()
    python_ok, python_version = _python_ok()
    qt_platform = os.environ.get("QT_QPA_PLATFORM", "")
    display = os.environ.get("DISPLAY", "")

    harness_error = ""
    editors: list[str] = []
    try:
        sys.path[:0] = [str(root), str(root / "snes"), str(root / "nes")]
        import retro_harness  # noqa: F401
        from retro_harness.editor_registry import registered_editor_projects

        editors = [project.project_id for project in registered_editor_projects()]
        harness_ok = True
    except Exception as exc:  # noqa: BLE001 — doctor must report any import failure
        harness_ok = False
        harness_error = f"{type(exc).__name__}: {exc}"

    manifests = sorted((root / "docs" / "manifests").glob("*.yaml"))
    roms = {
        "super_metroid": (root / "roms" / "SuperMetroid.sfc").is_file(),
        "harvest": (root / "snes" / "harvest" / "custom_integrations" / "HarvestMoon-Snes" / "rom.sfc").is_file(),
        "smb": (root / "roms" / "Super Mario Bros..nes").is_file(),
    }

    ok = python_ok and harness_ok and bool(editors)
    report = {
        "ok": ok,
        "root": str(root),
        "python": python_version,
        "python_ok": python_ok,
        "harness_ok": harness_ok,
        "harness_error": harness_error,
        "editors": editors,
        "manifest_count": len(manifests),
        "qt_platform": qt_platform,
        "display": display,
        "roms": roms,
    }
    print(json.dumps(report, indent=2))
    for key, value in report.items():
        if key == "roms":
            for name, present in roms.items():
                print(f"rom_{name}={str(present).lower()}")
            continue
        if key == "editors":
            print(f"editors={','.join(editors) if editors else ''}")
            continue
        if key == "harness_error" and not value:
            continue
        print(f"{key}={value if not isinstance(value, bool) else str(value).lower()}")
    if not python_ok:
        print("fail=python must be 3.12 via `uv run` (.python-version; stable-retro has no 3.13+ wheels)", file=sys.stderr)
    if not harness_ok:
        print(f"fail=retro_harness import failed: {harness_error}", file=sys.stderr)
    if not editors:
        print("fail=no registered editors (expected at least harvest)", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
