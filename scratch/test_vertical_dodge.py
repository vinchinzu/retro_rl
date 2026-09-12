from retro_harness.env import make_env
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot
from zelda_i.level4.boss_combat import Level4GleeokFightController
from zelda_i.dungeon.gleeok import gleeok_live, gleeok_heads_live, gleeok_fireballs, _south_stand_action
from zelda_i.level4.boss_combat import approach_dodge_thr, APPROACH_SOUTH_Y
from retro_harness.nes import nes_action

def smart_dodge_dir(snap, thr=18):
    balls = gleeok_fireballs(snap)
    if not balls:
        return None
    nearest = min(balls, key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y))
    dist = abs(nearest.x - snap.link_x) + abs(nearest.y - snap.link_y)
    if dist > thr:
        return None
    dx = nearest.x - snap.link_x
    dy = nearest.y - snap.link_y
    # If fireball is on nearly the same row as Link, step DOWN away from body to evade!
    if abs(dy) <= 6:
        if snap.link_y < 155:
            return "DOWN"
    # If fireball is directly above/below Link, step horizontal
    if nearest.x >= snap.link_x:
        return "LEFT" if snap.link_x > 56 else "RIGHT"
    return "RIGHT" if snap.link_x < 200 else "LEFT"

for test_hp in (111, 106, 104, 102, 100, 96):
    env = make_env(GAME, "Level4GleeokEnter", GAME_DIR, render_mode="rgb_array")
    env.reset()
    env.data.set_value("health", test_hp)
    
    ctl = Level4GleeokFightController(tag="test", continuous_mode=True, stand_dy=22, max_frames=2500)
    
    last_hp = test_hp
    approached = False
    invuln = 0
    ok = False
    
    for f in range(2500):
        snap = read_snapshot(env.get_ram())
        if snap.screen == 0x03 and (snap.triforce & 0x08):
            ok = True
            print(f"start_hp={test_hp:3d} -> SUCCESS at f={f}! final_hp={snap.health}")
            break
        if snap.mode == 17:
            print(f"start_hp={test_hp:3d} -> DEAD at f={f}! (hp was {snap.health})")
            break
            
        bodies = gleeok_live(snap)
        heads = gleeok_heads_live(snap)
        
        if snap.health < last_hp:
            last_hp = snap.health
            invuln = 48
        if invuln > 0:
            invuln -= 1
            
        if not approached:
            if invuln <= 0:
                dodge_a = smart_dodge_dir(snap, thr=22)
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
            
        dodge = None if invuln > 0 else smart_dodge_dir(snap, thr=18)
        if dodge is not None:
            env.step(nes_action(dodge))
            continue
            
        if bodies:
            env.step(_south_stand_action(snap, bodies[0], stand_dy=22))
        elif heads:
            nearest = min(heads, key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y))
            dx, dy = nearest.x - snap.link_x, nearest.y - snap.link_y
            face = ("RIGHT" if dx > 0 else "LEFT") if abs(dx) >= abs(dy) else ("DOWN" if dy > 0 else "UP")
            env.step(nes_action(face, "A"))
        else:
            # Move UP into room 0x03
            env.step(nes_action("UP"))
            
    env.close()
