"""Find the live 0x30 CheckWarps stand after kill-clear + block push.

Controller walk + optional XY poke around the revealed top-right stairs.
No InitMode9, no NEXT_SCREEN, no 0x04 door poke.
"""
from __future__ import annotations

from retro_harness.env import make_env
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.level9_stairs import (
    ROOM30,
    ROOM30_STAIR_X,
    ROOM30_STAIR_Y,
    dest_report,
    live_combat_objects,
    stair_transition_modes,
    walk_to_step,
    chase_sword_step,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_LINK_X, ADDR_LINK_Y, PLAY_MODE, read_snapshot
from zelda_i.scripts.run_level9_ganon import FIXTURE_SOURCE, _assign, _idle, _step
from zelda_i.scripts.run_level9_stairs import materialize_stair_room, _walk_target

TAG = "l9_room30_stand_probe"


def _note(env, label: str) -> dict:
    snap = read_snapshot(env.get_ram())
    block = next((o for o in snap.objects if o.type_id == 0x68), None)
    row = dest_report(snap)
    row["label"] = label
    row["block"] = None if block is None else {"x": block.x, "y": block.y}
    print(
        f"{label}: room=0x{snap.screen:02x} mode={snap.mode} "
        f"xy=({snap.link_x},{snap.link_y}) tile=0x{snap.colliding_tile:02x} "
        f"block={row['block']}"
    )
    return row


def reveal_stairs(env, total, assist):
    cooldown = 0
    for _ in range(2500):
        snap = read_snapshot(env.get_ram())
        if stair_transition_modes(snap.mode):
            return
        combat = live_combat_objects(snap)
        if snap.screen == ROOM30 and not combat:
            break
        frame, cooldown = chase_sword_step(snap, cooldown)
        _step(env, frame.action, assist=assist, total=total)
    _idle(env, 12, assist=assist, total=total)
    for x, y in ((0x88, 0x90), (0x80, 0x90), (0x78, 0x90), (0x90, 0x90)):
        _walk_target(env, total, x, y)
        for _ in range(40):
            snap = read_snapshot(env.get_ram())
            if stair_transition_modes(snap.mode):
                return
            _step(env, nes_action("LEFT"), assist=None, total=total)
    _walk_target(env, total, ROOM30_STAIR_X, ROOM30_STAIR_Y, frames=400)


def try_stand(env, total, x, y, hold="IDLE", frames=40):
    for _ in range(300):
        snap = read_snapshot(env.get_ram())
        if stair_transition_modes(snap.mode) or (
            snap.screen != ROOM30 and snap.mode == PLAY_MODE
        ):
            return dest_report(snap), True
        frame = walk_to_step(snap, x, y, y_first=True, tol=0)
        if frame.reason == "walk_arrived":
            break
        _step(env, frame.action, assist=None, total=total)
    action = nes_idle_action() if hold == "IDLE" else nes_action(hold)
    for _ in range(frames):
        snap = read_snapshot(env.get_ram())
        if stair_transition_modes(snap.mode) or (
            snap.screen != ROOM30 and snap.mode == PLAY_MODE
        ):
            return dest_report(snap), True
        _step(env, action, assist=None, total=total)
    return dest_report(read_snapshot(env.get_ram())), False


def poke_scan(env, total, xs, ys):
    hits = []
    start = read_snapshot(env.get_ram())
    sx, sy = int(start.link_x), int(start.link_y)
    for y in ys:
        for x in xs:
            _assign(env, ADDR_LINK_X, x)
            _assign(env, ADDR_LINK_Y, y)
            _step(env, nes_idle_action(), assist=None, total=total)
            _idle(env, 8, assist=None, total=total)
            snap = read_snapshot(env.get_ram())
            rec = {
                "stand": [x, y],
                "tile": int(snap.colliding_tile),
                "mode": int(snap.mode),
                "screen": int(snap.screen),
                "xy": [snap.link_x, snap.link_y],
            }
            if stair_transition_modes(snap.mode) or snap.screen != ROOM30:
                rec["triggered"] = True
                hits.append(rec)
                print("POKE_HIT", rec)
                return hits
            if 0x70 <= int(snap.colliding_tile) <= 0x73:
                rec["stair_tile"] = True
                hits.append(rec)
    _assign(env, ADDR_LINK_X, sx)
    _assign(env, ADDR_LINK_Y, sy)
    _step(env, nes_idle_action(), assist=None, total=total)
    return hits


def main() -> int:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    total = [0]
    report = {"ok": False, "room": "0x30", "route_eligible": False}
    try:
        obs, loader, loaded = materialize_stair_room(env, ROOM30, total=total)
        report["loaded"] = loaded
        report["loader"] = loader.label
        if not loaded:
            report["error"] = "loader failed"
            return 1
        report["settle"] = _note(env, "settle")
        save_rgb_png(obs, RECORDINGS_DIR / f"{TAG}_settle.png")
        reveal_stairs(env, total, assist)
        report["after_reveal"] = _note(env, "after_reveal")
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{TAG}_after_reveal.png")

        candidates = [
            (ROOM30_STAIR_X, ROOM30_STAIR_Y),
            (208, 93),
            (208, 97),
            (208, 109),
            (200, 93),
            (212, 93),
            (206, 93),
            (208, 141),
            (208, 125),
            (216, 93),
            (208, 89),
        ]
        attempts = []
        for x, y in candidates:
            for hold in ("IDLE", "UP", "DOWN"):
                dest, hit = try_stand(env, total, x, y, hold=hold, frames=30)
                attempts.append({"target": [x, y], "hold": hold, "hit": hit, "dest": dest})
                print(
                    f"STAND ({x},{y}) {hold} hit={hit} mode={dest['mode']} "
                    f"screen=0x{dest['screen']:02x} xy={dest['link']}"
                )
                if hit:
                    report["winner"] = {"target": [x, y], "hold": hold, "dest": dest}
                    save_rgb_png(env.render(), RECORDINGS_DIR / f"{TAG}_hit.png")
                    report["ok"] = True
                    report["attempts"] = attempts
                    write_json_report(RECORDINGS_DIR / f"{TAG}.json", report)
                    print("WINNER", report["winner"])
                    return 0
        report["attempts"] = attempts
        xs = list(range(196, 221, 2))
        ys = list(range(85, 145, 2))
        report["poke_hits"] = poke_scan(env, total, xs, ys)
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{TAG}_after_poke.png")
        report["final"] = _note(env, "final")
        write_json_report(RECORDINGS_DIR / f"{TAG}.json", report)
        print("NO_CONTROLLER_HIT poke_hits", len(report["poke_hits"]))
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
