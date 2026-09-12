from retro_harness.env import make_env
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot
from zelda_i.level4.boss_combat import Level4GleeokFightController
from zelda_i.dungeon.gleeok import gleeok_live, gleeok_heads_live, gleeok_fireballs, _south_stand_action
from zelda_i.level4.boss_combat import approach_dodge_thr, APPROACH_SOUTH_Y
from retro_harness.nes import nes_action

def dodge_with_vertical(snap, *, thr=18, allow_vertical=False):
    balls = gleeok_fireballs(snap)
    if not balls:
        return None
    nearest = min(balls, key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y))
    dist = abs(nearest.x - snap.link_x) + abs(nearest.y - snap.link_y)
    if dist > thr:
        return None
    dx = nearest.x - snap.link_x
    dy = nearest.y - snap.link_y
    # When fireball is on the same row, step DOWN to let it pass overhead!
    if abs(dy) <= 6 and abs(dx) <= 24:
        if snap.link_y < 155:
            return "DOWN"
    if allow_vertical and abs(dy) > abs(dx):
        if abs(dx) >= 2:
            if dx >= 0:
                return "LEFT" if snap.link_x > 56 else "RIGHT"
            return "RIGHT" if snap.link_x < 200 else "LEFT"
        if snap.link_x >= 120:
            return "RIGHT" if snap.link_x < 200 else "LEFT"
        return "LEFT" if snap.link_x > 56 else "RIGHT"
    if nearest.x >= snap.link_x:
        return "LEFT" if snap.link_x > 56 else "RIGHT"
    return "RIGHT" if snap.link_x < 200 else "LEFT"

import zelda_i.level4.boss_combat as bc
orig_dodge = bc._fireball_dodge_dir
bc._fireball_dodge_dir = dodge_with_vertical

for test_hp in (111, 107, 106, 104, 102, 100):
    env = make_env(GAME, "Level4GleeokEnter", GAME_DIR, render_mode="rgb_array")
    env.reset()
    env.data.set_value("health", test_hp)
    ctl = Level4GleeokFightController(tag="test", continuous_mode=True, stand_dy=22, max_frames=3000)
    total = [0]
    res = ctl.run(env, None, total)
    env.close()
    ok = res.get("ok", False)
    frames = res.get("frames", 0)
    err = res.get("error", "")
    print(f"start_hp={test_hp:3d} -> ok={ok} frames={frames:4d} err={err}")
