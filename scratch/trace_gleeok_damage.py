from retro_harness.env import make_env
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot
from zelda_i.level4.boss_combat import make_gleeok_fight_controller

env = make_env(GAME, "Level4GleeokEnter", GAME_DIR, render_mode="rgb_array")
env.reset()
ctl = make_gleeok_fight_controller(tag="test", continuous_mode=True)

# Custom loop to record every hit Link takes
total = [0]
# We can run ctl.run and look at its log or we can run frame by frame
last_hp = 111
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.gleeok import gleeok_live, gleeok_heads_live, gleeok_fireballs, _south_stand_action, _fireball_dodge_dir
from zelda_i.level4.boss_combat import approach_dodge_thr, APPROACH_SOUTH_Y

snap0 = read_snapshot(env.get_ram())
start_health = int(snap0.health)
approached = False
invuln = 0

for f in range(2500):
    snap = read_snapshot(env.get_ram())
    if snap.health < last_hp:
        print(f"Hit at f={f}: hp {last_hp} -> {snap.health} (lost {last_hp - snap.health}) link=({snap.link_x},{snap.link_y}) app={approached} invuln={invuln}")
        last_hp = snap.health
        invuln = 48
    if invuln > 0:
        invuln -= 1
        
    if snap.screen == 0x03 and (snap.triforce & 0x08):
        print(f"TF08 collected at f={f}!")
        break
    if snap.mode == 17:
        print(f"Dead at f={f}!")
        break
        
    bodies = gleeok_live(snap)
    heads = gleeok_heads_live(snap)
    dodge_thr = approach_dodge_thr(start_health=start_health, approached=approached)
    
    if not approached:
        if invuln <= 0:
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
        
    dodge = None if invuln > 0 else _fireball_dodge_dir(snap, thr=dodge_thr)
    if dodge is not None:
        env.step(nes_action(dodge))
        continue
        
    if bodies:
        env.step(_south_stand_action(snap, bodies[0], stand_dy=ctl.stand_dy))
    elif heads:
        nearest = min(heads, key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y))
        dx, dy = nearest.x - snap.link_x, nearest.y - snap.link_y
        face = ("RIGHT" if dx > 0 else "LEFT") if abs(dx) >= abs(dy) else ("DOWN" if dy > 0 else "UP")
        env.step(nes_action(face, "A"))
    else:
        # Move up towards door to 0x03
        if snap.screen == 0x13:
            env.step(nes_action("UP"))
        else:
            env.step(nes_action("UP"))

env.close()
