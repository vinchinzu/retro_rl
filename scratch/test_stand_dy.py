from retro_harness.env import make_env
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.level4.boss_combat import Level4GleeokFightController

with open("scratch/l4_entered_0x13.state", "rb") as f:
    state_bytes = f.read()

for dy in (22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50):
    for fb_thr in (14, 18, 22, 26):
        env = make_env(GAME, "Level4Entrance", GAME_DIR, render_mode="rgb_array")
        env.reset()
        env.em.set_state(state_bytes)
        ctl = Level4GleeokFightController(
            tag="test",
            continuous_mode=True,
            stand_dy=dy,
            fireball_dodge_dist=fb_thr,
            max_frames=1000,
        )
        total = [0]
        res = ctl.run(env, None, total)
        env.close()
        frames = res.get("frames", 0)
        ok = res.get("ok", False)
        err = res.get("error", "")
        print(f"dy={dy:2d} fb_thr={fb_thr:2d} -> ok={ok} frames={frames:4d} err={err} dmg_events={res.get('dmg_events', 0)}")
