"""Does an overworld wave come back? Walk the pre-L1 corridor, then lap it.

One pass of ``0x77 -> 0x4A`` cannot fund the 20R bomb pack (``docs/PRE_L1.md``:
32 bodies, ~12R of random drops, and the forced 5-rupees need a 26-kill
streak). The only supply left is a **second** wave on the same screens, which
every prior sitting has listed as untested.

The ROM says exactly when that happens, and it is not "walk two screens away"
(`aldonunez/zelda1-disassembly`):

* ``ModifyObjCountByHistoryOW`` (``Z_05.asm``) clears a room's kill-count
  flags -- a full respawn -- only when the room is **absent from the 6-entry
  ``RoomHistory`` ($621)** *and* its flags already read the max, 7.
* ``SaveKillCountOW`` writes 7 exactly when ``RoomKillCount >= RoomObjCount``,
  i.e. when the screen was **cleared**, and otherwise accumulates the partial
  count (capped at 7).
* ``RunCrossRoomTasksAndBeginUpdateMode`` (``Z_07.asm``) only appends a room
  to the history **when it is not already in it**, and the cycling index does
  not advance otherwise.

That last rule is why the 0x4A<->0x49 restock in ``rupee_farm`` never worked
and why no out-and-back over the corridor can: every screen on the way back is
already in the history, so nothing is ever evicted. The corridor has **seven**
distinct screens against six history slots, so the smallest thing that does
work is a full lap to ``0x77`` and back -- walking onto 0x77 evicts 0x78, and
each screen after that evicts the next one in front of Link.

The lap table itself is ``gathering.PRE_L1_LAP_HOPS`` and the arithmetic is
``overworld.respawn``; this probe is the live check on both. It prints, per
screen *visit*: the live wave, the kill-count flags on arrival and departure,
and the history at the moment the ROM made its decision. A respawn is
``flags 7 -> 0`` on arrival with the wave back up.

    QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scratch/probe_respawn_lap.py --tag lap1
"""

from __future__ import annotations

import argparse

from retro_harness.audit import AuditCapabilities, AuditedEnv
from retro_harness.env import make_env, reset_obs
from retro_harness.segment_runner import configure_headless, write_json_report
from zelda_i.combat import live_enemies
from zelda_i.dungeon.ids import OBJECT_NAMES
from zelda_i.overworld import gathering as gathering_mod
from zelda_i.overworld.gathering import make_shop_p7_walk_controller
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.overworld.respawn import read_room_history
from zelda_i.ram import PLAY_MODE, read_snapshot, world_flag
from zelda_i.spine.survival import run_survival_spine, spine_final_fields

from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent


def _wave(snap) -> dict[str, int]:
    out: dict[str, int] = {}
    for obj in live_enemies(snap):
        name = OBJECT_NAMES.get(int(obj.type_id), f"unk_{int(obj.type_id):#04x}")
        out[name] = out.get(name, 0) + 1
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="lap")
    parser.add_argument("--laps", type=int, default=1)
    args = parser.parse_args(argv)
    configure_headless()

    laps = max(int(args.laps), 0)
    gathering_mod.make_shop_p7_walk_controller = (  # type: ignore[assignment]
        lambda: make_shop_p7_walk_controller(laps=laps)
    )

    visits: list[dict] = []
    state = {"screen": -1}

    def on_frame(env, _obs, _action, frame: int) -> None:
        ram = env.get_ram()
        snap = read_snapshot(ram)
        if int(snap.level) != 0 or int(snap.mode) != PLAY_MODE:
            return
        screen = int(snap.screen)
        flags = int(world_flag(ram, screen)) & 0x07
        if screen != state["screen"]:
            state["screen"] = screen
            visits.append(
                {
                    "f": frame,
                    "screen": f"{screen:#04x}",
                    # The ROM decided with the history as it stands here:
                    # ``ModifyObjCountByHistoryOW`` runs inside
                    # ``CreateRoomObjects``, *before* the room is appended.
                    "hist": [f"{r:#04x}" for r in read_room_history(ram)[0]],
                    "hist_i": read_room_history(ram)[1],
                    "flags_in": flags,
                    "flags_out": flags,
                    "wave_in": _wave(snap),
                    "peak": len(live_enemies(snap)),
                    "R": int(snap.rupees),
                    "hp": int(snap.health),
                    "streak": int(snap.world_kill_count),
                }
            )
        elif visits:
            cur = visits[-1]
            cur["flags_out"] = flags
            cur["peak"] = max(int(cur["peak"]), len(live_enemies(snap)))
            if len(cur["wave_in"]) < len(_wave(snap)) or sum(
                cur["wave_in"].values()
            ) < sum(_wave(snap).values()):
                # The list is built over a few frames after the scroll.
                cur["wave_in"] = _wave(snap)
            cur["R_out"] = int(snap.rupees)
            cur["streak_out"] = int(snap.world_kill_count)

    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    payload: dict | None = None
    try:
        obs, _ = reset_obs(env)
        env = AuditedEnv(env, capabilities=AuditCapabilities.all("zelda_i.lap"))
        run = run_survival_spine(
            env, obs, assist=None, on_frame=on_frame, through="pre-l1",
            allow_pokes=False,
        )
        ram = env.get_ram()
        snap = read_snapshot(ram)
        report = run.report()
        hunt: dict = {}
        stages = []
        for stage in report.get("stages", []):
            ctl = stage.get("controller") or {}
            nested = ctl.get("hunt") if isinstance(ctl.get("hunt"), dict) else {}
            if "streak_resets" in nested:
                hunt = nested
            stages.append(
                {
                    "name": stage.get("name"),
                    "frames": stage.get("frames"),
                    "ok": stage.get("ok"),
                    "notes": stage.get("notes"),
                }
            )
        payload = {
            "tag": args.tag,
            "laps": int(args.laps),
            "ok": report.get("ok"),
            "failed": report.get("failed_stage"),
            "final": spine_final_fields(snap, ram),
            "visits": visits,
            "hunt": hunt,
            "stages": stages,
        }
    finally:
        env.close()

    write_json_report(OUT_DIR / f"{args.tag}.json", payload)
    write_json_report(RECORDINGS_DIR / f"{args.tag}.json", payload)
    print(
        f"ok={payload['ok']} failed={payload['failed']} "
        f"R={payload['final']['rupees']} visits={len(visits)}"
    )
    print("| # | f | screen | wave | n | flags in->out | history | R | streak |")
    print("|" + "---|" * 9)
    for i, v in enumerate(visits):
        wave = " ".join(f"{k}x{n}" for k, n in sorted(v["wave_in"].items())) or "-"
        print(
            f"| {i} | {v['f']} | `{v['screen']}` | {wave} | {v['peak']} | "
            f"{v['flags_in']}->{v['flags_out']} | {' '.join(v['hist'])} | "
            f"{v['R']}->{v.get('R_out', v['R'])} | "
            f"{v['streak']}->{v.get('streak_out', v['streak'])} |"
        )
    h = payload["hunt"]
    print(
        f"kills={h.get('kills')}/{h.get('kills_counter')} "
        f"best={h.get('streak_best')} resets={h.get('streak_resets')} "
        f"drops={h.get('drops_by_state')}\n  by_screen={h.get('kills_by_screen')}"
        f"\n  stages={[(s['name'], s['frames'], s['ok']) for s in stages]}"
        f"\n  notes={payload['stages'][-1]['notes']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
