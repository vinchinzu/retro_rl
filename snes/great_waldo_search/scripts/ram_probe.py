"""Differential RAM probe for Waldo cursor X/Y and scene candidates.

Loads Scene1.state when present (else boots briefly), pulses d-pad axes, and
writes ranked candidates to recordings/ plus docs/ram_map.md notes via stdout
JSON report.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from great_waldo_search.paths import (
    DOCS_DIR,
    GAME,
    GAME_DIR,
    INTEGRATION_DIR,
    RECORDINGS_DIR,
)
from great_waldo_search.ram import (
    deltas_for_move,
    filter_byte_range,
    ram_copy,
    rank_axis_candidates,
)
from retro_harness.env import get_available_states, make_env
from retro_harness.actions import buttons, idle_action

def _configure_headless() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("SDL_SOFTWARE_RENDERER", "1")

def _save_png(obs: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(obs).save(path)

def _idle(env: object, frames: int) -> np.ndarray:
    obs = None
    for _ in range(frames):
        obs, *_rest = env.step(idle_action())  # type: ignore[attr-defined]
    assert obs is not None
    return obs

def _hold(env: object, *names: str, frames: int = 8) -> np.ndarray:
    action = buttons(*names)
    obs = None
    for _ in range(frames):
        obs, *_rest = env.step(action)  # type: ignore[attr-defined]
    assert obs is not None
    return obs

def _probe_axis(
    env: object,
    direction: str,
    *,
    pulses: int = 5,
    hold: int = 6,
    settle: int = 4,
) -> list:
    """Pulse one d-pad direction and collect per-pulse RAM deltas."""
    groups = []
    for _ in range(pulses):
        before = ram_copy(np.asarray(env.get_ram(), dtype=np.uint8))  # type: ignore[attr-defined]
        _hold(env, direction, frames=hold)
        _idle(env, settle)
        after = ram_copy(np.asarray(env.get_ram(), dtype=np.uint8))  # type: ignore[attr-defined]
        deltas = filter_byte_range(deltas_for_move(before, after, limit=None))
        # Keep only small steps typical of cursor motion.
        deltas = [d for d in deltas if 1 <= abs(d.delta) <= 16]
        groups.append(deltas)
    return groups

def _render_ram_map_md(report: dict) -> str:
    lines = [
        "# Great Waldo Search — RAM map",
        "",
        "Segment-first discovery notes. Addresses below are **candidates**",
        "from differential probes; confirm before trusting in policies.",
        "",
        f"Probe state: `{report.get('state', '')}`",
        f"Frames probed: `{report.get('pulses_per_axis', '')}` pulses/axis",
        "",
        "## Cursor X candidates",
        "",
        "| Address | Hits | Last before → after |",
        "|---------|------|---------------------|",
    ]
    for row in report.get("cursor_x", [])[:15]:
        lines.append(
            f"| `0x{row['address']:04X}` ({row['address']}) | {row['hits']} | "
            f"{row['last_before']} → {row['last_after']} |"
        )
    lines.extend(
        [
            "",
            "## Cursor Y candidates",
            "",
            "| Address | Hits | Last before → after |",
            "|---------|------|---------------------|",
        ]
    )
    for row in report.get("cursor_y", [])[:15]:
        lines.append(
            f"| `0x{row['address']:04X}` ({row['address']}) | {row['hits']} | "
            f"{row['last_before']} → {row['last_after']} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Prefer addresses that change by 1–8 per d-pad pulse and stay",
            "  in a screen-like range (often 0–255 for 8-bit cursor).",
            "- Scene / mode IDs are not yet isolated; re-probe across scene",
            "  transitions once Scene1 → Scene2 clears work.",
            "- Update `custom_integrations/GreatWaldoSearch-Snes/data.json`",
            "  only after a candidate survives freeze/correlation checks.",
            "",
        ]
    )
    return "\n".join(lines)

def run_ram_probe(
    *,
    state: str | None = None,
    pulses: int = 5,
    write_docs: bool = True,
) -> dict:
    """Run axis probes and write candidate report + ram_map.md."""
    _configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    available = get_available_states(GAME, GAME_DIR)
    chosen = state
    if chosen is None:
        if "Scene1" in available:
            chosen = "Scene1"
        elif available:
            chosen = available[0]
        else:
            chosen = "NONE"

    env = make_env(
        game=GAME,
        state=chosen,
        game_dir=GAME_DIR,
        render_mode="rgb_array",
    )
    try:
        obs, _info = env.reset()
        _save_png(obs, RECORDINGS_DIR / f"ram_probe_{chosen}_start.png")
        # Warm up a few idle frames.
        obs = _idle(env, 20)

        right_groups = _probe_axis(env, "RIGHT", pulses=pulses)
        left_groups = _probe_axis(env, "LEFT", pulses=pulses)
        down_groups = _probe_axis(env, "DOWN", pulses=pulses)
        up_groups = _probe_axis(env, "UP", pulses=pulses)

        x_cands = rank_axis_candidates(right_groups + left_groups, axis="x")
        y_cands = rank_axis_candidates(down_groups + up_groups, axis="y")

        obs = _idle(env, 2)
        _save_png(obs, RECORDINGS_DIR / f"ram_probe_{chosen}_end.png")

        report = {
            "state": chosen,
            "available_states": available,
            "pulses_per_axis": pulses,
            "integration_dir": str(INTEGRATION_DIR),
            "cursor_x": [
                {
                    "address": c.address,
                    "hits": c.hits,
                    "last_before": c.last_before,
                    "last_after": c.last_after,
                }
                for c in x_cands[:30]
            ],
            "cursor_y": [
                {
                    "address": c.address,
                    "hits": c.hits,
                    "last_before": c.last_before,
                    "last_after": c.last_after,
                }
                for c in y_cands[:30]
            ],
        }
        report_path = RECORDINGS_DIR / "ram_probe_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[ram_probe] report={report_path}")
        print(f"[ram_probe] top X: {report['cursor_x'][:5]}")
        print(f"[ram_probe] top Y: {report['cursor_y'][:5]}")

        if write_docs:
            DOCS_DIR.mkdir(parents=True, exist_ok=True)
            md_path = DOCS_DIR / "ram_map.md"
            md_path.write_text(_render_ram_map_md(report), encoding="utf-8")
            print(f"[ram_probe] wrote {md_path}")
        return report
    finally:
        env.close()

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default=None, help="Save state name or NONE")
    parser.add_argument("--pulses", type=int, default=5)
    parser.add_argument("--no-docs", action="store_true")
    return parser

def main(argv: list[str] | None = None) -> int:
    """CLI entry for the RAM cursor probe."""
    args = _build_parser().parse_args(argv)
    run_ram_probe(
        state=args.state,
        pulses=args.pulses,
        write_docs=not args.no_docs,
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
