"""Log every $50/$627 change on the pre-L1 hunt walk.

Scratch only. Hypothesis: Survival assist refills $066F/$0670 before
``ScreenHunter.observe``, so ``damage_taken`` stays 0 while a Link-enemy
collision still zeros the forced-drop counters. Link iframes ($04F0, object
slot 0) and knockback ($00D3) survive the refill.

    QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scratch/probe_kill_streak.py
    QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scratch/probe_kill_streak.py --no-infinite-life
"""

from __future__ import annotations

import argparse
from pathlib import Path

from retro_harness.audit import AuditCapabilities, AuditedEnv
from retro_harness.env import make_env, reset_obs
from retro_harness.segment_runner import configure_headless, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon.ids import OBJECT_NAMES
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WORLD_FLAGS, read_snapshot, read_u8
from zelda_i.spine.survival import run_survival_spine, spine_final_fields

OUT_DIR = Path(__file__).resolve().parent
ADDR_IFRAMES = 0x04F0  # object_iframes[0] — Link; redcandle
ADDR_KNOCKBACK = 0x00D3  # object_knockback_timer[0]


def _objs(snap) -> list[dict]:
    out = []
    for obj in snap.objects:
        if int(obj.slot) < 1 or int(obj.type_id) in (0, 0xFF):
            continue
        if int(obj.hp) <= 0 and int(obj.type_id) != 0x60:
            continue
        out.append(
            {
                "slot": int(obj.slot),
                "type": int(obj.type_id),
                "name": OBJECT_NAMES.get(int(obj.type_id), f"unk_{obj.type_id:#04x}"),
                "hp": int(obj.hp),
                "state": int(obj.state),
                "x": int(obj.x),
                "y": int(obj.y),
            }
        )
    return out


def _event(env, frame: int, kind: str, prev: dict, snap, ram) -> dict:
    screen = int(snap.screen)
    flags = int(read_u8(ram, ADDR_WORLD_FLAGS + screen))
    return {
        "f": frame,
        "kind": kind,
        "screen": screen,
        "mode": int(snap.mode),
        "submode": int(snap.submode),
        "xy": [int(snap.link_x), int(snap.link_y)],
        "world": int(snap.world_kill_count),
        "help": int(snap.help_drop_count),
        "help_val": int(snap.help_drop_value),
        "prev_world": prev["world"],
        "prev_help": prev["help"],
        "hp": int(snap.health),
        "partial": int(snap.heart_partial),
        "iframes": int(read_u8(ram, ADDR_IFRAMES)),
        "knockback": int(read_u8(ram, ADDR_KNOCKBACK)),
        "rupees": int(snap.rupees),
        "rupees_add": int(read_u8(ram, 0x067D)),
        "flags": flags,
        "flag_kills": (flags & 0xC0) >> 6,
        "objects": _objs(snap),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--infinite-life",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Default off: the bomb walk is a Clean farm.",
    )
    parser.add_argument("--tag", default="kill_streak")
    args = parser.parse_args(argv)
    configure_headless()

    events: list[dict] = []
    prev = {"world": -1, "help": -1, "hp": -1, "partial": -1, "iframes": -1}

    def on_frame(env, _obs, _action, frame: int) -> None:
        ram = env.get_ram()
        snap = read_snapshot(ram)
        world = int(snap.world_kill_count)
        help_ = int(snap.help_drop_count)
        hp = int(snap.health)
        partial = int(snap.heart_partial)
        iframes = int(read_u8(ram, ADDR_IFRAMES))
        kind = None
        if prev["world"] >= 0 and world == 0 and prev["world"] > 0:
            kind = "reset"
        elif world != prev["world"] or help_ != prev["help"]:
            kind = "count"
        elif iframes > 0 and prev["iframes"] == 0:
            kind = "iframe_arm"
        elif hp != prev["hp"] or partial != prev["partial"]:
            kind = "hp"
        if kind is not None:
            events.append(_event(env, frame, kind, prev, snap, ram))
        prev["world"] = world
        prev["help"] = help_
        prev["hp"] = hp
        prev["partial"] = partial
        prev["iframes"] = iframes

    assist = UnlimitedHealthAssist(enabled=True) if args.infinite_life else None
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    payload: dict | None = None
    try:
        obs, _ = reset_obs(env)
        env = AuditedEnv(
            env, capabilities=AuditCapabilities.all("zelda_i.kill_streak")
        )
        run = run_survival_spine(
            env,
            obs,
            assist=assist,
            on_frame=on_frame,
            through="pre-l1",
            allow_pokes=False,
        )
        ram = env.get_ram()
        snap = read_snapshot(ram)
        hunt = {}
        for stage in run.report().get("stages", []):
            ctl = stage.get("controller") or {}
            nested = ctl.get("hunt") if isinstance(ctl.get("hunt"), dict) else {}
            if "streak_resets" in ctl:
                hunt = ctl
            elif "streak_resets" in nested:
                hunt = nested
        payload = {
            "ok": run.report().get("ok"),
            "failed": run.report().get("failed_stage"),
            "final": spine_final_fields(snap, ram),
            "assist": None if assist is None else assist.report(),
            "hunt": {
                k: hunt.get(k)
                for k in (
                    "kills",
                    "kills_counter",
                    "rupees_banked",
                    "damage_taken",
                    "hurt_events",
                    "streak_best",
                    "streak_resets",
                    "kills_by_screen",
                    "notes",
                )
            },
            "events": events,
        }
    finally:
        env.close()

    if payload is None:
        raise RuntimeError("probe ended before a report")
    out = OUT_DIR / f"{args.tag}.json"
    rec = RECORDINGS_DIR / f"{args.tag}.json"
    write_json_report(out, payload)
    write_json_report(rec, payload)

    resets = [e for e in events if e["kind"] == "reset"]
    assist_rep = payload.get("assist") or {}
    print(
        f"ok={payload['ok']} failed={payload['failed']} "
        f"rupees={payload['final']['rupees']} "
        f"hunt_damage={payload['hunt'].get('damage_taken')} "
        f"hurt={payload['hunt'].get('hurt_events')} "
        f"assist_events={assist_rep.get('damage_events')} "
        f"assist_total={assist_rep.get('total_damage')} "
        f"streak_best={payload['hunt'].get('streak_best')} "
        f"streak_resets={payload['hunt'].get('streak_resets')} "
        f"kills={payload['hunt'].get('kills')}/{payload['hunt'].get('kills_counter')} "
        f"events={len(events)} resets={len(resets)}"
    )
    for e in resets:
        names = ",".join(f"{o['name']}@{o['x']},{o['y']}" for o in e["objects"][:6])
        print(
            f"  RESET f={e['f']} scr=0x{e['screen']:02x} mode={e['mode']} "
            f"xy={e['xy']} world {e['prev_world']}->{e['world']} "
            f"iframes={e['iframes']} knock={e['knockback']} "
            f"hp={e['hp']:#04x} part={e['partial']:#04x} "
            f"objs={names}"
        )
    print(f"wrote {out}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
