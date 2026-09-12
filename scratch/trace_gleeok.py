import sys
from retro_harness.env import make_env
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.level4.boss_combat import make_gleeok_fight_controller
from zelda_i.ram import read_snapshot
from zelda_i.dungeon.gleeok import gleeok_live, gleeok_heads_live, gleeok_fireballs

def main():
    env = make_env(GAME, "Level4Entrance", GAME_DIR, render_mode="rgb_array")
    env.reset()
    with open("scratch/l4_entered_0x13.state", "rb") as f:
        state = f.read()
    env.em.set_state(state)
    
    ctl = make_gleeok_fight_controller(tag="test_gleeok", continuous_mode=True)
    # We can inspect frame by frame inside a custom loop or hook
    total = [0]
    
    # Let's run custom stepping loop mirroring Level4GleeokFightController but printing details around f=180..210
    from zelda_i.level4.boss_combat import approach_dodge_thr, APPROACH_SOUTH_Y, _fireball_dodge_dir, _south_stand_action
    from retro_harness.nes import nes_action, nes_idle_action
    
    snap0 = read_snapshot(env.get_ram())
    start_health = int(snap0.health)
    approached = False
    
    for f in range(250):
        snap = read_snapshot(env.get_ram())
        bodies = gleeok_live(snap)
        heads = gleeok_heads_live(snap)
        balls = gleeok_fireballs(snap)
        
        if f >= 180 or snap.mode == 17:
            b_info = [f"body({b.slot}):({b.x},{b.y})" for b in bodies]
            h_info = [f"head({h.slot}):({h.x},{h.y})" for h in heads]
            f_info = [f"ball({bl.slot}):({bl.x},{bl.y})" for bl in balls]
            print(f"f={f:3d} mode={snap.mode} hp={snap.health} link=({snap.link_x},{snap.link_y}) app={approached} | {' '.join(b_info)} | {' '.join(h_info)} | {' '.join(f_info)}")
            
        if snap.mode == 17:
            print(f"DEAD at f={f}")
            break
            
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
            if f >= 180:
                print(f"   -> dodge {dodge}")
            env.step(nes_action(dodge))
            continue
            
        if bodies:
            act = _south_stand_action(snap, bodies[0], stand_dy=ctl.stand_dy)
            if f >= 180:
                print(f"   -> stand_act {act}")
            env.step(act)
        else:
            env.step(nes_action("UP", "A"))

    env.close()

if __name__ == "__main__":
    main()
