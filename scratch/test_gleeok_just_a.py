from retro_harness.env import make_env
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot
from zelda_i.level4.boss_combat import Level4GleeokFightController
from zelda_i.dungeon.gleeok import _south_stand_action, STAND_DY
from retro_harness.nes import nes_action

def south_stand_stationary(snap, body, *, stand_dy=STAND_DY):
    sx = int(body.x)
    sy = min(173, int(body.y) + stand_dy)
    if abs(snap.link_x - sx) > 3 or abs(snap.link_y - sy) > 3:
        if abs(snap.link_y - sy) >= abs(snap.link_x - sx):
            face = "DOWN" if snap.link_y < sy else "UP"
        else:
            face = "RIGHT" if snap.link_x < sx else "LEFT"
        return nes_action(face)
    if snap.facing != 2: # not facing UP
        return nes_action("UP", "A")
    return nes_action("A")

import zelda_i.level4.boss_combat as bc
bc._south_stand_action = south_stand_stationary

for test_hp in (111, 107, 106, 104, 102, 100, 96):
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
    print(f"just_a start_hp={test_hp:3d} -> ok={ok} frames={frames:4d} err={err}")
