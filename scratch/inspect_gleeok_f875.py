from retro_harness.env import make_env
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot
from zelda_i.level4.boss_combat import Level4GleeokFightController
from zelda_i.dungeon.gleeok import gleeok_live, gleeok_heads_live

env = make_env(GAME, "Level4GleeokEnter", GAME_DIR, render_mode="rgb_array")
env.reset()
ctl = Level4GleeokFightController(tag="test", continuous_mode=True, stand_dy=22, max_frames=880)
total = [0]
res = ctl.run(env, None, total)
snap = read_snapshot(env.get_ram())
bodies = gleeok_live(snap)
heads = gleeok_heads_live(snap)
print(f"At frame {ctl.frames}: bodies={len(bodies)} heads={len(heads)}")
for b in bodies:
    print(f"  body slot {b.slot}: hp={b.hp} state={b.state} xy=({b.x},{b.y})")
for h in heads:
    print(f"  head slot {h.slot}: hp={h.hp} state={h.state} xy=({h.x},{h.y})")
env.close()
