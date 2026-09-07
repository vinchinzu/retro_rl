"""After the 5 Wizzrobes in L9 room 0x10 are dead (via the proven should_swing_at
combat loop) and room_all_dead's countdown reaches 0, sweep candidate stand
points to find where ADDR_ARROWS actually flips to 2 (Silver Arrows pickup).
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.combat import should_swing_at
from zelda_i.level9.stair_run import _assign
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import ADDR_LINK_X, ADDR_LINK_Y, read_snapshot

STATE_NAME = "L9Room10EntryReal"


def combat_step(snap):
    live = [o for o in snap.objects if o.type_id in (0x23, 0x24) and o.hp > 0]
    if not live:
        return None
    nearest = min(live, key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y))
    dx, dy = nearest.x - snap.link_x, nearest.y - snap.link_y
    d = ("RIGHT" if dx > 0 else "LEFT") if abs(dx) > abs(dy) else ("DOWN" if dy > 0 else "UP")
    if should_swing_at(snap.link_x, snap.link_y, d, live):
        return nes_action(d, "A")
    return nes_action(d)


def main() -> int:
    configure_headless()
    env = make_env(GAME, STATE_NAME, GAME_DIR, render_mode="rgb_array")
    reset_obs(env)
    assist = UnlimitedHealthAssist(enabled=True)
    frame = 0
    # Phase 1: kill all 5 Wizzrobes.
    while frame < 8000:
        snap = read_snapshot(env.get_ram())
        act = combat_step(snap)
        if act is None:
            break
        env.step(act)
        frame += 1
        assist.apply_env(env, frame=frame)
    snap = read_snapshot(env.get_ram())
    print(f"combat done f{frame} xy=({snap.link_x},{snap.link_y}) rad={snap.room_all_dead} arrows={snap.arrows}")
    # Phase 2: idle until room_all_dead counts down to 0.
    while frame < 9000:
        env.step(nes_idle_action())
        frame += 1
        assist.apply_env(env, frame=frame)
        snap = read_snapshot(env.get_ram())
        if snap.room_all_dead == 0:
            break
    print(f"rad=0 reached f{frame} xy=({snap.link_x},{snap.link_y}) rad={snap.room_all_dead} arrows={snap.arrows}")
    # Phase 3: sweep candidate stand points across the open middle zone.
    candidates = [
        (120, 141), (128, 141), (112, 141), (120, 149), (120, 133),
        (120, 125), (120, 157), (96, 141), (144, 141), (120, 109),
        (120, 165), (104, 141), (136, 141),
    ]
    for cx, cy in candidates:
        _assign(env, ADDR_LINK_X, cx)
        _assign(env, ADDR_LINK_Y, cy)
        for _ in range(30):
            env.step(nes_idle_action())
            frame += 1
            assist.apply_env(env, frame=frame)
        snap = read_snapshot(env.get_ram())
        print(f"stand ({cx},{cy}) -> arrows={snap.arrows} room_item_id={snap.room_item_id}")
        if snap.arrows >= 2:
            print("FOUND IT")
            break
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
