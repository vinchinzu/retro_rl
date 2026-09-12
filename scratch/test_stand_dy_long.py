from retro_harness.env import make_env
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.level4.boss_combat import Level4GleeokFightController

for dy in (20, 22, 24, 26, 28, 30):
    env = make_env(GAME, "Level4GleeokEnter", GAME_DIR, render_mode="rgb_array")
    env.reset()
    ctl = Level4GleeokFightController(tag="test", continuous_mode=True, stand_dy=dy, max_frames=3500)
    total = [0]
    res = ctl.run(env, None, total)
    env.close()
    ok = res.get("ok", False)
    frames = res.get("frames", 0)
    err = res.get("error", "")
    notes = [n for n in res.get("notes", []) if "boss_dead" in n or "tf08" in n]
    print(f"dy={dy:2d} (y={111+dy:3d}) -> ok={ok} frames={frames:4d} err={err} notes={notes}")
