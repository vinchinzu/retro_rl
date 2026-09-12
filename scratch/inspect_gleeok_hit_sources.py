from retro_harness.env import make_env
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot
from zelda_i.level4.boss_combat import make_gleeok_fight_controller
from zelda_i.dungeon.gleeok import gleeok_live, gleeok_heads_live, gleeok_fireballs, _south_stand_action, _fireball_dodge_dir
from zelda_i.level4.boss_combat import approach_dodge_thr, APPROACH_SOUTH_Y
from retro_harness.nes import nes_action

env = make_env(GAME, "Level4GleeokEnter", GAME_DIR, render_mode="rgb_array")
env.reset()
ctl = make_gleeok_fight_controller(tag="test", continuous_mode=True)

last_hp = 111
approached = False
invuln = 0

for f in range(1500):
    snap = read_snapshot(env.get_ram())
    bodies = gleeok_live(snap)
    heads = gleeok_heads_live(snap)
    balls = gleeok_fireballs(snap)
    
    if snap.health < last_hp:
        print(f"HIT at f={f}: hp {last_hp}->{snap.health} link=({snap.link_x},{snap.link_y})")
        for b in bodies:
            d = abs(b.x - snap.link_x) + abs(b.y - snap.link_y)
            print(f"   body: xy=({b.x},{b.y}) dist={d} hp={b.hp}")
        for h in heads:
            d = abs(h.x - snap.link_x) + abs(h.y - snap.link_y)
            print(f"   head(slot {h.slot}): xy=({h.x},{h.y}) dist={d} hp={h.hp}")
        for bl in balls:
            d = abs(bl.x - snap.link_x) + abs(bl.y - snap.link_y)
            print(f"   ball(slot {bl.slot}): xy=({bl.x},{bl.y}) dist={d}")
        last_hp = snap.health
        invuln = 48
    if invuln > 0:
        invuln -= 1
        
    if not bodies and f > 30:
        print(f"Boss dead at f={f}!")
        break
        
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
