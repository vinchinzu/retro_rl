"""Idle-settle mode-9 cellar 0x2F. The K4 fixture is an unloaded first frame.

L7 0x7B needed ~400f idle before the two-ladder room painted. This probe
idles 600f with no movement (never UP on the east/source spawn) and
screenshots every 100f. Do not cellar-cross here.

    QT_QPA_PLATFORM=offscreen uv run python \\
        nes/zelda_i/scratch/probe_l8_2f_settle.py \\
        --from-state Level8Interior2FCellarReconFixture \\
        --tag 20260904_S1 --infinite-life --no-video
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.ids import object_name
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOMBS,
    ADDR_BOW,
    ADDR_CANDLE,
    ADDR_KEYS,
    ADDR_MAGIC_KEY,
    ADDR_MAP,
    ADDR_RUPEES,
    ADDR_SELECTED_ITEM,
    ADDR_SWORD,
    ADDR_TRIFORCE,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)
from zelda_i.runner import add_common_args, make_assist, write_report
from zelda_i.screen_glance import leftover_from_snapshot

LEVEL8 = 8
SOURCE_STATE = "Level8Interior2FCellarReconFixture"
LOAD_IDLE = 600
SHOT_EVERY = 100
PASSAGE_MODE = 9


def _inventory(env: Any) -> dict[str, int]:
    ram = env.get_ram()
    snap = read_snapshot(ram)
    u = lambda addr: int(read_u8(ram, addr))
    return {
        "sword": u(ADDR_SWORD), "bombs": u(ADDR_BOMBS), "bow": u(ADDR_BOW),
        "arrows": u(ADDR_ARROWS), "candle": u(ADDR_CANDLE), "keys": u(ADDR_KEYS),
        "rupees": u(ADDR_RUPEES), "magic_key": u(ADDR_MAGIC_KEY),
        "selected_item": u(ADDR_SELECTED_ITEM), "triforce": u(ADDR_TRIFORCE),
        "map": u(ADDR_MAP),
        "health": int(snap.health), "heart_containers": int(snap.heart_containers),
    }


def _typed(snap: ZeldaSnapshot, *, live: bool = False) -> list:
    return [
        obj for obj in snap.objects
        if 1 <= obj.slot <= 12 and obj.type_id not in (0, 0xFF)
        and (not live or obj.hp > 0)
    ]


def _obj_row(obj: Any) -> dict[str, Any]:
    return {
        "slot": int(obj.slot), "type": int(obj.type_id),
        "type_hex": f"0x{obj.type_id:02X}", "type_name": object_name(obj.type_id),
        "xy": [int(obj.x), int(obj.y)], "hp": int(obj.hp),
    }


def _glance(env: Any) -> dict[str, Any]:
    ram = env.get_ram()
    snap = read_snapshot(ram)
    result = leftover_from_snapshot(snap)
    result.update(
        level=int(snap.level),
        screen_hex=f"0x{snap.screen:02X}",
        next_screen_hex=f"0x{snap.next_screen:02X}",
        xy=[int(snap.link_x), int(snap.link_y)],
        tile=int(snap.colliding_tile),
        facing=int(snap.facing),
        transitioning=bool(snap.transitioning),
        room_item_hex=f"0x{snap.room_item_id:02X}",
        cur_opened_doors=int(snap.cur_opened_doors),
        open_doorway_mask=int(snap.open_doorway_mask),
        inventory=_inventory(env),
        live_objects=[_obj_row(obj) for obj in _typed(snap, live=True)],
        typed_objects=[_obj_row(obj) for obj in _typed(snap)],
    )
    return result


def _save_shot(env: Any, obs: Any, tag: str, label: str, frame: int) -> str:
    snap = read_snapshot(env.get_ram())
    path = RECORDINGS_DIR / (
        f"l8_2f_settle_{tag}_{label}_f{frame}_"
        f"L{snap.level}_s{snap.screen:02x}_m{snap.mode}.png"
    )
    save_rgb_png(obs if obs is not None else env.render(), path)
    return str(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    add_common_args(
        parser,
        default_state=SOURCE_STATE,
        default_tag="20260904_S1",
    )
    parser.add_argument("--idle", type=int, default=LOAD_IDLE)
    parser.add_argument("--save-fixture", default=None)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    if not args.infinite_life:
        raise SystemExit("this fixture probe requires --infinite-life")

    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    assist = make_assist(True)
    raw_env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    payload: dict[str, Any] = {
        "trial": "0x2f_settle",
        "from_state": args.from_state,
        "fixture_only": True, "natural_entry": False, "route_eligible": False,
        "infinite_life": True,
        "prediction": {
            "written_before_run": True,
            "claim": (
                "Idle 600f on Level8Interior2FCellarReconFixture (mode 9 "
                "$EB=0x2F (208,141) tile 0x71). Room paints as a two-ladder "
                "cellar like L7 0x7B / L8 0x0F. Census drops 0x3F play "
                "residuals. No movement, no source-ladder UP. Keys 8 bombs 6 "
                "MK 1 TF 0x7F."
            ),
            "contingency": (
                "Still black at 600f: leftover stays unloaded 0x2F, do not "
                "cross, do not poke. Mode/screen change is a miss. Return to "
                "play 0x3F is a miss."
            ),
        },
        "ticks": [],
        "screenshots": [],
    }
    try:
        obs, _ = reset_obs(raw_env)
        assist.apply_env(raw_env, frame=0)
        start = _glance(raw_env)
        payload["start"] = start
        payload["screenshots"].append(
            _save_shot(raw_env, obs, args.tag, "start", 0)
        )
        payload["ticks"].append({"frame": 0, **{
            k: start[k] for k in (
                "mode", "screen_hex", "xy", "tile", "room_item_hex",
                "live_objects",
            )
        }})

        for frame in range(1, int(args.idle) + 1):
            obs = raw_env.step(nes_idle_action())[0]
            assist.apply_env(raw_env, frame=frame)
            if frame % SHOT_EVERY == 0 or frame == int(args.idle):
                glance = _glance(raw_env)
                payload["ticks"].append({"frame": frame, **{
                    k: glance[k] for k in (
                        "mode", "screen_hex", "xy", "tile", "room_item_hex",
                        "live_objects",
                    )
                }})
                payload["screenshots"].append(
                    _save_shot(
                        raw_env, obs, args.tag, f"idle_{frame}", frame
                    )
                )

        final = _glance(raw_env)
        payload["final"] = final
        payload["screenshots"].append(
            _save_shot(raw_env, obs, args.tag, "final", int(args.idle))
        )
        tel = assist.telemetry
        payload["frames"] = int(args.idle)
        payload["assist"] = assist.report()
        payload["deaths"] = int(tel.deaths)
        payload["progression_writes"] = int(tel.progression_writes)
        payload["capacity_writes"] = int(tel.capacity_writes)
        payload["success"] = (
            final["level"] == LEVEL8
            and int(final["mode"]) == PASSAGE_MODE
            and int(final["screen"]) == 0x2F
            and int(tel.deaths) == 0
            and int(tel.progression_writes) == 0
            and int(tel.capacity_writes) == 0
        )
        if args.save_fixture and payload["success"]:
            from retro_harness.env import save_state, state_path
            from zelda_i.dungeon.trace import compact_snapshot, write_state_provenance

            after = read_snapshot(raw_env.get_ram())
            path = save_state(raw_env, GAME_DIR, GAME, args.save_fixture)
            source_path = state_path(GAME_DIR, GAME, SOURCE_STATE)
            result = {
                "ok": True, "source_state": SOURCE_STATE,
                "fixture_state": args.save_fixture,
                "state": compact_snapshot(after),
                "census": final, "dest_eb": "0x2F",
                "magic_key": int(final["inventory"]["magic_key"]),
                "keys": int(final["inventory"]["keys"]),
                "bombs": int(final["inventory"]["bombs"]),
                "fixture_writes": [],
            }
            write_state_provenance(
                path,
                source_state_path=source_path if source_path.exists() else None,
                request={
                    "bead": "rr-6o7.2",
                    "phase": "level8_interior_0x2f_cellar_settled",
                    "track": "recon_fixture",
                    "route_eligible": False,
                    "fixture_only": True,
                    "natural_entry": False,
                    "fixture_writes": [],
                    "notes": [
                        "settled mode-9 cellar 0x2F after load idle, not the "
                        "first unloaded mode-9 frame, not on L8_THROUGH",
                    ],
                },
                selected_trial=result,
                natural_entry=False,
            )
            payload["saved_fixture"] = {
                "state_path": str(path),
                "provenance": str(path.with_suffix(".provenance.json")),
            }
    finally:
        report = write_report("l8_2f_settle", payload, tag=args.tag)
        raw_env.close()

    print(f"report={report}")
    print(f"success={payload.get('success')} failed={payload.get('failed')}")
    final = payload.get("final") or {}
    print(
        f"settled mode={final.get('mode')} screen={final.get('screen_hex')} "
        f"xy={final.get('xy')} tile={final.get('tile')} "
        f"live={len(final.get('live_objects') or [])}"
    )
    for tick in payload.get("ticks") or []:
        print(
            f"f{tick['frame']}: m{tick['mode']} {tick['screen_hex']} "
            f"{tick['xy']} tile={tick['tile']} live={len(tick['live_objects'])}"
        )
    return 0 if payload.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
