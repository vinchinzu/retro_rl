from retro_harness.env import make_env
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot, write_u8, ADDR_HEALTH
from zelda_i.level4.boss_combat import Level4GleeokFightController

for test_hp in (111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 100, 96):
    env = make_env(GAME, "Level4GleeokEnter", GAME_DIR, render_mode="rgb_array")
    env.reset()
    # Poke starting health only to test the fight policy threshold
    ram = env.get_ram()
    write_u8(ram, ADDR_HEALTH, test_hp)
    ctl = Level4GleeokFightController(tag="test", continuous_mode=True, stand_dy=22, max_frames=3000)
    total = [0]
    res = ctl.run(env, None, total)
    env.close()
    ok = res.get("ok", False)
    frames = res.get("frames", 0)
    err = res.get("error", "")
    print(f"start_hp={test_hp:3d} -> ok={ok} frames={frames:4d} err={err}")
