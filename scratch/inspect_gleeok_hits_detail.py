from retro_harness.env import make_env
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot
from zelda_i.level4.boss_combat import Level4GleeokFightController
from zelda_i.dungeon.gleeok import gleeok_live, gleeok_heads_live, gleeok_fireballs, _south_stand_action, _fireball_dodge_dir
from zelda_i.level4.boss_combat import approach_dodge_thr, APPROACH_SOUTH_Y
from retro_harness.nes import nes_action

env = make_env(GAME, "Level4GleeokEnter", GAME_DIR, render_mode="rgb_array")
env.reset()
ctl = Level4GleeokFightController(tag="test", continuous_mode=True, stand_dy=22, max_frames=800)

last_hp = 111
approached = False
invuln = 0

for f in range(800):
    snap = read_snapshot(env.get_ram())
    bodies = gleeok_live(snap)
    heads = gleeok_heads_live(snap)
    balls = gleeok_fireballs(snap)
    
    if snap.health < last_hp or f in (165, 166, 167, 368, 369, 370, 495, 496, 497, 565, 566, 567, 738, 739, 740):
        print(f"f={f:3d} hp={snap.health} link=({snap.link_x},{snap.link_y}) facing={snap.facing} sword={snap.sword} invuln={invuln}")
        if bodies:
            print(f"   body: ({bodies[0].x},{bodies[0].y}) hp={bodies[0].hp}")
        for h in heads:
            print(f"   head({h.slot}): ({h.x},{h.y}) hp={h.hp}")
        for bl in balls:
            print(f"   ball({bl.slot}): ({bl.x},{bl.y})")
            
    if snap.health < last_hp:
        last_hp = snap.health
        invuln = 48
    if invuln > 0:
        invuln -= 1
        
    dodge_thr = approach_dodge_thr(start_health=111, approached=approached)
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
    else:
        env.step(nes_action("UP", "A"))

env.close()
