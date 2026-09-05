"""Recon: poke Whistle on pond 0x42, drain, enter L7, pin Level7Entrance.

Not a route claim. ADDR_WHISTLE poke is recon-only (not on the Survival spine;
ASSIST_CONTRACT still forbids it there). Geometry walk is the already-green
OverworldToLevel7PondController from PostSwordStart.

    # first run (~1 min): walk 0x77→0x42, save OW_L7Pond, poke, drain, pin
    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/pond/probe_l7_pond_drain.py --tag drain_v1

    # iterate drain/stairs from the geometry leftover (~15 s)
    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/pond/probe_l7_pond_drain.py \
        --from-state OW_L7Pond --tag drain_v2
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.ops import mem_write
from zelda_i.level7.overworld import OverworldToLevel7PondController
from zelda_i.paths import GAME, GAME_DIR, INTEGRATION_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_FOOD,
    ADDR_SELECTED_ITEM,
    ADDR_WHISTLE,
    PLAY_MODE,
    read_snapshot,
    read_u8,
)
from zelda_i.runner import add_common_args, make_assist, write_report
from zelda_i.screen_glance import leftover_from_snapshot

POND_SCREEN = 0x42
LEVEL7 = 7
WHISTLE_B_SLOT = 5  # recorder; matches level5.boss_path.WHISTLE_B_SLOT
# drain_v1 leftover: pond drained, stairs visible LEFT of Link at (120,134).
# Center (120,141) is sand (tile 149), not the hole. Sweep the dry bed.
STAIR_CANDIDATES = (
    (104, 128),
    (96, 128),
    (112, 128),
    (104, 136),
    (96, 136),
    (112, 136),
    (104, 120),
    (88, 128),
    (120, 128),
)
POND_STATE = "OW_L7Pond"
ENTRANCE_STATE = "Level7Entrance"
WALK_MAX = 30_000
DRAIN_MAX = 4_000


def _glance(env) -> dict[str, Any]:
    ram = env.get_ram()
    snap = read_snapshot(ram)
    leftover = leftover_from_snapshot(snap)
    leftover.update(
        {
            "level": int(snap.level),
            "food": int(read_u8(ram, ADDR_FOOD)),
            "whistle": int(read_u8(ram, ADDR_WHISTLE)),
            "selected_item": int(read_u8(ram, ADDR_SELECTED_ITEM)),
            "sword": int(snap.sword),
            "rupees": int(snap.rupees),
        }
    )
    return leftover


def _shot(obs, tag: str, frame: int, snap) -> Path:
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    png = RECORDINGS_DIR / f"{tag}_f{frame}_L{snap.level}_s{snap.screen:02x}_m{snap.mode}.png"
    save_rgb_png(obs, png)
    return png


def _step(env, assist, total: list[int], action, obs_box: list):
    obs, *_ = env.step(action)
    obs_box[0] = obs
    total[0] += 1
    if assist is not None:
        assist.apply_env(env, frame=total[0])
    return read_snapshot(env.get_ram())


def _walk_to_pond(env, assist, total: list[int], obs_box: list, tag: str) -> dict:
    ctl = OverworldToLevel7PondController()
    last_screen = None
    notes: list[str] = []
    for _ in range(WALK_MAX):
        snap = read_snapshot(env.get_ram())
        if last_screen is None or snap.screen != last_screen:
            _shot(obs_box[0], tag, total[0], snap)
            last_screen = snap.screen
            notes.append(f"screen_0x{snap.screen:02x}@{total[0]}")
        action = ctl.step(snap)
        snap = _step(env, assist, total, action.action, obs_box)
        if ctl.success or ctl.failed:
            break
    leftover = _glance(env)
    return {
        "success": bool(ctl.success),
        "failed": bool(ctl.failed),
        "phase": getattr(getattr(ctl, "phase", None), "name", None),
        "notes": notes[-12:],
        "controller": ctl.report(),
        "leftover": leftover,
    }


def _poke_whistle(env) -> dict[str, Any]:
    ram = env.get_ram()
    from_w = int(read_u8(ram, ADDR_WHISTLE))
    from_b = int(read_u8(ram, ADDR_SELECTED_ITEM))
    wmsg = mem_write(env, ADDR_WHISTLE, 1)
    bmsg = mem_write(env, ADDR_SELECTED_ITEM, WHISTLE_B_SLOT)
    ram = env.get_ram()
    return {
        "whistle": {
            "address": ADDR_WHISTLE,
            "from": from_w,
            "to": int(read_u8(ram, ADDR_WHISTLE)),
            "write": wmsg,
        },
        "selected_item": {
            "address": ADDR_SELECTED_ITEM,
            "from": from_b,
            "to": int(read_u8(ram, ADDR_SELECTED_ITEM)),
            "write": bmsg,
            "want": WHISTLE_B_SLOT,
        },
        "progression_writes": 0,
        "capacity_writes": 0,
        "recon": True,
        "spine": False,
    }


def _seek(env, assist, total: list[int], obs_box: list, tx: int, ty: int, *, max_f: int) -> Any:
    snap = read_snapshot(env.get_ram())
    for _ in range(max_f):
        if snap.level == LEVEL7 or snap.mode in (9, 11, 16):
            return snap
        dx, dy = tx - snap.link_x, ty - snap.link_y
        if abs(dx) <= 4 and abs(dy) <= 4:
            return snap
        if abs(dx) >= abs(dy):
            act = nes_action("RIGHT" if dx > 0 else "LEFT")
        else:
            act = nes_action("DOWN" if dy > 0 else "UP")
        snap = _step(env, assist, total, act, obs_box)
    return snap


def _l7_play(snap) -> bool:
    return (
        int(snap.level) == LEVEL7
        and int(snap.mode) == PLAY_MODE
        and not snap.transitioning
    )


def _blow_and_enter(env, assist, total: list[int], obs_box: list, tag: str) -> dict:
    samples: list[dict] = []
    notes: list[str] = []
    shots: list[str] = []

    def sample(reason: str) -> dict:
        snap = read_snapshot(env.get_ram())
        rec = {
            "frame": total[0],
            "reason": reason,
            "level": int(snap.level),
            "mode": int(snap.mode),
            "screen": int(snap.screen),
            "xy": [int(snap.link_x), int(snap.link_y)],
            "tile": int(snap.colliding_tile),
            "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
            "selected": int(read_u8(env.get_ram(), ADDR_SELECTED_ITEM)),
        }
        samples.append(rec)
        notes.append(
            f"{reason} f{total[0]} L{snap.level} m{snap.mode} "
            f"s0x{snap.screen:02x} ({snap.link_x},{snap.link_y}) tile={snap.colliding_tile}"
        )
        return rec

    sample("pre_blow")
    _seek(env, assist, total, obs_box, 128, 189, max_f=80)
    sample("stand")
    shots.append(str(_shot(obs_box[0], tag, total[0], read_snapshot(env.get_ram()))))

    for _ in range(12):
        _step(env, assist, total, nes_action("B"), obs_box)
    sample("blew")
    for _ in range(240):
        snap = _step(env, assist, total, nes_idle_action(), obs_box)
        if snap.mode != PLAY_MODE or snap.level != 0:
            break
    sample("post_song")
    shots.append(str(_shot(obs_box[0], tag, total[0], read_snapshot(env.get_ram()))))

    last_key = None
    entered = False
    cand_i = 0
    grid = list(STAIR_CANDIDATES)
    for x in range(80, 161, 8):
        for y in range(104, 161, 8):
            grid.append((x, y))
    for i in range(DRAIN_MAX):
        snap = read_snapshot(env.get_ram())
        key = (snap.level, snap.mode, snap.screen)
        if key != last_key:
            shots.append(str(_shot(obs_box[0], tag, total[0], snap)))
            sample(f"change_{i}")
            last_key = key
        if _l7_play(snap):
            entered = True
            sample("l7_play")
            break
        if snap.mode in (9, 11, 16) or snap.level == LEVEL7:
            snap = _step(env, assist, total, nes_idle_action(), obs_box)
            if i % 30 == 0:
                sample(f"t{i}")
            continue
        if snap.mode == PLAY_MODE and snap.level == 0 and snap.screen == POND_SCREEN:
            tx, ty = grid[min(cand_i, len(grid) - 1)]
            dx, dy = tx - snap.link_x, ty - snap.link_y
            if abs(dx) <= 3 and abs(dy) <= 3:
                cand_i = min(cand_i + 1, len(grid) - 1)
                act = nes_action("UP")
            elif abs(dx) >= abs(dy):
                act = nes_action("RIGHT" if dx > 0 else "LEFT")
            else:
                act = nes_action("DOWN" if dy > 0 else "UP")
            snap = _step(env, assist, total, act, obs_box)
        else:
            snap = _step(env, assist, total, nes_idle_action(), obs_box)
        if i % 60 == 0:
            sample(f"t{i}")

    leftover = _glance(env)
    shots.append(str(_shot(obs_box[0], f"{tag}_final", total[0], read_snapshot(env.get_ram()))))
    return {
        "entered": entered,
        "leftover": leftover,
        "samples": samples[-48:],
        "notes": notes[-24:],
        "screenshots": shots[-16:],
        "stair_index": cand_i,
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _write_provenance(name: str, payload: dict[str, Any]) -> Path:
    path = INTEGRATION_DIR / f"{name}.provenance.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _save_pin(env, name: str, *, leftover: dict, poke: dict | None, tag: str) -> dict:
    state_path = save_state(env, GAME_DIR, GAME, name)
    if poke is None:
        warning = (
            "Development pin. Geometry leftover on Demon pond 0x42 from "
            "PostSwordStart; whistle=0. Not natural-entry. Do not STATUS-promote."
        )
        via = "PostSwordStart OverworldToLevel7PondController geometry walk (whistle=0)"
    else:
        warning = (
            "Development pin. Not natural-entry. Whistle was poked on the "
            "PostSwordStart pond leftover; do not STATUS-promote."
        )
        via = "PostSwordStart pond walk + ADDR_WHISTLE recon poke + recorder drain"
    prov = {
        "acceptance_warning": warning,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "natural_entry": False,
        "request": {
            "route_eligible": False,
            "segment": name,
            "tag": tag,
            "whistle_poke": poke is not None,
            "via": via,
        },
        "schema_version": 1,
        "selected_trial": leftover,
        "state_path": str(state_path),
        "state_sha256": _sha256(Path(state_path)),
        "poke": poke,
    }
    prov_path = _write_provenance(name, prov)
    return {"state": str(state_path), "provenance": str(prov_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_state="PostSwordStart", default_tag="drain")
    parser.add_argument(
        "--skip-walk",
        action="store_true",
        help="Assume --from-state is already on pond 0x42 (OW_L7Pond).",
    )
    args = parser.parse_args()
    configure_headless()
    assist = make_assist(args.infinite_life)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    total = [0]
    obs_box: list = [None]
    payload: dict[str, Any] = {
        "from_state": args.from_state,
        "infinite_life": args.infinite_life,
        "tag": args.tag,
        "route_eligible": False,
        "recon": True,
    }
    try:
        obs, _ = reset_obs(env)
        obs_box[0] = obs
        obs, *_ = env.step(nes_idle_action())
        obs_box[0] = obs
        total[0] += 1
        start = _glance(env)
        payload["start"] = start
        skip = args.skip_walk or (
            args.from_state == POND_STATE
            and int(start.get("screen", -1)) == POND_SCREEN
            and int(start.get("level", -1)) == 0
        )
        if skip:
            walk = {"skipped": True, "leftover": start}
        else:
            walk = _walk_to_pond(env, assist, total, obs_box, args.tag)
        payload["walk"] = walk
        leftover = walk.get("leftover") or _glance(env)
        on_pond = (
            int(leftover.get("level", -1)) == 0
            and int(leftover.get("screen", leftover.get("room", -1))) == POND_SCREEN
            and int(leftover.get("mode", -1)) == PLAY_MODE
        )
        if on_pond:
            payload["pond_pin"] = _save_pin(
                env, POND_STATE, leftover=leftover, poke=None, tag=args.tag
            )
        if not on_pond:
            payload["success"] = False
            payload["failed"] = "not_on_pond"
            payload["leftover"] = leftover
        else:
            poke = _poke_whistle(env)
            payload["poke"] = poke
            payload["post_poke"] = _glance(env)
            drain = _blow_and_enter(env, assist, total, obs_box, args.tag)
            payload["drain"] = drain
            leftover = drain["leftover"]
            payload["leftover"] = leftover
            entered = bool(drain.get("entered")) and int(leftover.get("level", -1)) == LEVEL7
            payload["success"] = entered
            if entered:
                payload["entrance_pin"] = _save_pin(
                    env,
                    ENTRANCE_STATE,
                    leftover=leftover,
                    poke=poke,
                    tag=args.tag,
                )
            else:
                payload["failed"] = "did_not_enter_l7"
        payload["frames"] = total[0]
        payload["assist"] = assist.report() if assist is not None else None
        png = RECORDINGS_DIR / f"{args.tag}_final.png"
        save_rgb_png(obs_box[0], png)
        payload["screenshot"] = str(png)
        out = write_report("l7_pond_drain", payload, tag=args.tag)
        print(out)
        print(
            f"success={payload.get('success')} frames={total[0]} "
            f"leftover={payload.get('leftover')} "
            f"failed={payload.get('failed')} "
            f"pond_pin={payload.get('pond_pin')} "
            f"entrance_pin={payload.get('entrance_pin')}"
        )
        notes = (payload.get("drain") or {}).get("notes") or []
        for line in notes[-12:]:
            print(line)
    finally:
        env.close()


if __name__ == "__main__":
    main()
