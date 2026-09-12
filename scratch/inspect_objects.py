from retro_harness.env import make_env
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot
from retro_harness.nes import nes_action

env = make_env(GAME, "Level4Entrance", GAME_DIR, render_mode="rgb_array")
env.reset()
with open("scratch/l4_entered_0x13.state", "rb") as f:
    state = f.read()
env.em.set_state(state)

# Step 206 frames matching trace
from zelda_i.level4.boss_combat import make_gleeok_fight_controller, approach_dodge_thr, APPROACH_SOUTH_Y, _fireball_dodge_dir, _south_stand_action
ctl = make_gleeok_fight_controller(tag="test_gleeok", continuous_mode=True)
from zelda_i.dungeon.gleeok import gleeok_live

snap0 = read_snapshot(env.get_ram())
start_health = int(snap0.health)
approached = False

for f in range(210):
    snap = read_snapshot(env.get_ram())
    bodies = gleeok_live(snap)
    if f in (101, 150, 200, 204, 205, 206):
        print(f"--- FRAME {f} (link: {snap.link_x}, {snap.link_y}, hp={snap.health}, mode={snap.mode}) ---")
        for o in snap.objects:
            if o.type_id != 0:
                print(f"  slot {o.slot:2d}: type=0x{o.type_id:02x} hp={o.hp} xy=({o.x},{o.y}) state=0x{o.state:02x} facing={o.facing}")
    dodge_thr = approach_dodge_thr(start_health=start_health, approached=approached)
    if not approached:
        dodge_a = _fireball_dodge_dir(snap, thr=dodge_thr)
        if dodge_a is not None:
            env.step(nes_action(dodge_a))
            continue
        if snap.link_y < APPROACH_SOUTH_Y:
            env.step(nes_action("DOWN"))
            continue
        bx = bodies[0].x if bodies else 124
        if abs(snap.link_x - bx) > 8:
            env.step(nes_action("RIGHT" if snap.link_x < bx else "LEFT"))
            continue
        approached = True
        continue
    dodge = _fireball_dodge_dir(snap, thr=dodge_thr)
    if dodge is not None:
        env.step(nes_action(dodge))
        continue
    if bodies:
        env.step(_south_stand_action(snap, bodies[0], stand_dy=ctl.stand_dy))
    else:
        env.step(nes_action("UP", "A"))
env.close()
