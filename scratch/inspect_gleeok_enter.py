from retro_harness.env import make_env
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot

env = make_env(GAME, "Level4GleeokEnter", GAME_DIR, render_mode="rgb_array")
env.reset()
snap = read_snapshot(env.get_ram())
print(f"Level4GleeokEnter: level={snap.level} screen=0x{snap.screen:02x} mode={snap.mode} xy=({snap.link_x},{snap.link_y}) hp={snap.health} keys={snap.keys} bombs={snap.bombs}")
from zelda_i.level4.boss_combat import make_gleeok_fight_controller
ctl = make_gleeok_fight_controller(tag="test", continuous_mode=True)
total = [0]
res = ctl.run(env, None, total)
print(f"Level4GleeokEnter result: ok={res.get('ok')} tf08={res.get('tf08')} frames={res.get('frames')} end_hp={read_snapshot(env.get_ram()).health}")
env.close()
