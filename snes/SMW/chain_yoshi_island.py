"""Replay YI2→YI4 clears with overworld walks; save chained entry + Iggy.

Natural-entry YI4 (big Mario) uses the chained clear seed — package YI4
recordings die on ``Chained_YoshiIsland4``. Rebuild::

    uv run python -m SMW chain-yi

YI2: ``recording_001.json`` (idle-padded). YI3: hillclimb ``recording_001.json``.
YI4: ``recording_004_chained_clear.json`` (from Chained_YoshiIsland4 play +
1 free-frame pad for Evaluator resync).

Live chain entry has no stable-retro free frame; after each level enter we
inject one idle frame so seeds recorded from ``make_env``/play stay aligned.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
GAME = "SuperMarioWorld-Snes-v0"
RUNS = ROOT / "optimizer" / "runs"

UP, DOWN, LEFT, RIGHT, Y = 4, 5, 6, 7, 1
OW_MODES = {0x0B, 0x0C, 0x0D, 0x0E}
# Death fade often hits 0x0B; real clears settle on 0x0C–0x0E.
OW_CLEAR_MODES = {0x0C, 0x0D, 0x0E}

YI2_CLEAR = RUNS / "smw_yoshi_island_2" / "recording_001.json"
YI3_CLEAR = RUNS / "smw_yoshi_island_3" / "recording_001.json"
YI4_CLEAR = RUNS / "smw_yoshi_island_4" / "recording_004_chained_clear.json"

IGGY_TRANS = 0x25  # Yoshi Island castle


def _info(env) -> dict:
    r = env.get_ram()
    return {
        "mode": int(r[0x100]),
        "trans": int(r[0x13BF]),
        "owx": int(r[0x1F17]),
        "owy": int(r[0x1F19]),
        "px": int(r[0xD1]) | (int(r[0xD2]) << 8),
        "py": int(r[0xD3]) | (int(r[0xD4]) << 8),
        "power": int(r[0x19]),
        "lives": int(r[0xDBE]),
        "exits": int(r[0x1F2E]),
    }


def _step(env, act, n: int = 1):
    a = np.asarray(act, dtype=np.int8)
    obs = None
    for _ in range(n):
        obs, *_ = env.step(a)
    return obs


def _replay(env, path: Path, label: str, *, live_entry_pad: bool = False) -> dict:
    """Replay raw buttons. Optional 1-frame pad after live level enter."""
    from retro_harness.platformer.bk2_extract import load_raw_buttons

    btns = load_raw_buttons(path)
    if not btns:
        raise SystemExit(f"No raw_buttons in {path}")
    if live_entry_pad:
        # Match make_env free-frame so seeds from play/file load stay in sync.
        _step(env, [0] * 12, 1)
    print(f"  replay {label}: {path.name} ({len(btns)} frames"
          f"{', +1 live pad' if live_entry_pad else ''})")
    for a in btns:
        _step(env, a)
    return _info(env)


def _wait_ow(env, *, lives_before: int, max_f: int = 600) -> dict:
    """Wait for a real clear OW — not death (lives drop / mode 0x0B only)."""
    nop = [0] * 12
    for _ in range(max_f):
        _step(env, nop)
        inf = _info(env)
        if inf["lives"] < lives_before:
            return inf  # death; caller checks
        if inf["mode"] in OW_CLEAR_MODES:
            for _ in range(100):
                _step(env, nop)
                inf = _info(env)
                if inf["mode"] == 0x0E:
                    return inf
            return inf
    return _info(env)


def _pulse(env, btn: int, times: int, hold: int = 22, gap: int = 12) -> dict:
    nop = [0] * 12
    for _ in range(times):
        act = [0] * 12
        act[btn] = 1
        _step(env, act, hold)
        _step(env, nop, gap)
        if _info(env)["mode"] not in (0x0D, 0x0E):
            return _info(env)
    return _info(env)


def _enter(env) -> dict:
    act = [0] * 12
    act[Y] = 1
    for _ in range(100):
        _step(env, act)
        if _info(env)["mode"] not in (0x0D, 0x0E):
            break
    nop = [0] * 12
    for _ in range(400):
        _step(env, nop)
        inf = _info(env)
        if inf["mode"] == 0x14:
            for _ in range(60):
                _step(env, nop)
                if _info(env)["mode"] != 0x14:
                    break
            return _info(env)
    return _info(env)


def _require_clear(inf: dict, lives_before: int, label: str) -> None:
    if inf["lives"] < lives_before:
        raise SystemExit(
            f"{label} died during clear (lives {lives_before}→{inf['lives']}): {inf}"
        )
    if inf["mode"] not in OW_MODES:
        raise SystemExit(f"{label} did not reach overworld: {inf}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stop-after",
        choices=("yi3", "yi4", "iggy"),
        default="iggy",
        help="Stop after this stage (default: full chain through Iggy entry)",
    )
    args = parser.parse_args(argv)

    required = [(YI2_CLEAR, "YI2"), (YI3_CLEAR, "YI3")]
    if args.stop_after in ("yi4", "iggy"):
        required.append((YI4_CLEAR, "YI4"))
    for p, name in required:
        if not p.is_file():
            print(f"Missing {name} clear: {p}", file=sys.stderr)
            sys.exit(1)

    from retro_harness.env import make_env, save_state

    env = make_env(game=GAME, state="YoshiIsland2", game_dir=ROOT, render_mode="rgb_array")
    env.reset()

    try:
        print("=== YI2 ===")
        lives0 = _info(env)["lives"]
        _replay(env, YI2_CLEAR, "yi2")
        inf = _wait_ow(env, lives_before=lives0)
        print(f"  OW {inf}")
        _require_clear(inf, lives0, "YI2")

        print("=== OW → YI3 ===")
        _pulse(env, UP, 8)
        inf = _enter(env)
        print(f"  entry {inf}")
        if inf["mode"] != 0x14 or inf["trans"] != 0x27:
            raise SystemExit(f"Expected YI3 (trans 0x27), got {inf}")
        path = save_state(env, ROOT, GAME, "Chained_YoshiIsland3")
        print(f"  saved {path}")

        if args.stop_after == "yi3":
            return

        print("=== YI3 ===")
        lives0 = inf["lives"]
        _replay(env, YI3_CLEAR, "yi3", live_entry_pad=True)
        inf = _wait_ow(env, lives_before=lives0)
        print(f"  OW {inf}")
        _require_clear(inf, lives0, "YI3")

        print("=== OW → YI4 ===")
        _pulse(env, RIGHT, 15)
        inf = _enter(env)
        print(f"  entry {inf}")
        if inf["mode"] != 0x14 or inf["trans"] != 0x26:
            raise SystemExit(f"Expected YI4 (trans 0x26), got {inf}")
        path = save_state(env, ROOT, GAME, "Chained_YoshiIsland4")
        print(f"  saved {path}")

        print("=== YI4 (chained natural-entry clear) ===")
        lives0 = inf["lives"]
        # recording_004_chained_clear already leads with 1 idle (Evaluator free-
        # frame pad). Live enter needs no extra pad (extra idle desyncs).
        _replay(env, YI4_CLEAR, "yi4", live_entry_pad=False)
        inf = _wait_ow(env, lives_before=lives0)
        print(f"  OW {inf}")
        _require_clear(inf, lives0, "YI4")
        path = save_state(env, ROOT, GAME, "Chained_AfterYI4_OW")
        print(f"  saved {path}")

        if args.stop_after == "yi4":
            return

        print("=== OW → Iggy's Castle ===")
        print(f"  pre-enter {inf}")
        inf = _enter(env)
        print(f"  entry {inf}")
        if inf["mode"] != 0x14 or inf["trans"] != IGGY_TRANS:
            raise SystemExit(
                f"Expected Iggy (trans {IGGY_TRANS:#x}), got {inf}. "
                "May need manual OW walk from Chained_AfterYI4_OW."
            )
        path = save_state(env, ROOT, GAME, "IggysCastle")
        print(f"  saved {path}")
        print("\nDone. Record Iggy with:")
        print("  uv run python -m SMW -l iggy play")
    finally:
        env.close()


if __name__ == "__main__":
    main()
