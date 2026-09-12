from retro_harness.env import make_env
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot
from zelda_i.level4.key01 import make_room_01_key_controller
from zelda_i.dungeon.engine import DungeonPhase

env = make_env(GAME, "Level4Room01", GAME_DIR, render_mode="rgb_array")
env.reset()
snap0 = read_snapshot(env.get_ram())
print(f"Room01 start: screen=0x{snap0.screen:02x} mode={snap0.mode} xy=({snap0.link_x},{snap0.link_y}) hp={snap0.health} keys={snap0.keys}")

# Test 1: with phase = FIGHT
ctl = make_room_01_key_controller()
ctl.phase = DungeonPhase.FIGHT
for f in range(2000):
    snap = read_snapshot(env.get_ram())
    act = ctl.step(snap)
    if ctl.success:
        print(f"Phase FIGHT success at f={f}! keys={snap.keys} hp={snap.health}")
        break
    env.step(act.action)

# Test 2: with phase = COLLECT_REWARD
env.reset()
ctl2 = make_room_01_key_controller()
ctl2.phase = DungeonPhase.COLLECT_REWARD
for f in range(2000):
    snap = read_snapshot(env.get_ram())
    act = ctl2.step(snap)
    if ctl2.success:
        print(f"Phase COLLECT_REWARD success at f={f}! keys={snap.keys} hp={snap.health}")
        break
    env.step(act.action)
env.close()
