from retro_harness.env import make_env
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot
from zelda_i.level4.path import make_room_31_clear_controller
from zelda_i.level4.dungeon import ROOM_31_SPEC, CombatTuning
from zelda_i.dungeon.engine import DungeonPhase
import dataclasses

with open("nes/zelda_i/custom_integrations/LegendOfZelda-Nes/Level4Room31.state", "rb") as f:
    state_bytes = f.read()

for eng_dist in (24, 36, 48, 56):
    env = make_env(GAME, "Level4Room31", GAME_DIR, render_mode="rgb_array")
    env.reset()
    env.em.set_state(state_bytes)
    
    combat = dataclasses.replace(ROOM_31_SPEC.combat, engage_distance=eng_dist)
    spec = dataclasses.replace(ROOM_31_SPEC, combat=combat)
    ctl = make_room_31_clear_controller()
    ctl.spec = spec
    ctl.phase = DungeonPhase.FIGHT
    
    snap0 = read_snapshot(env.get_ram())
    last_hp = snap0.health
    dmg = 0
    for f in range(10000):
        snap = read_snapshot(env.get_ram())
        if snap.health < last_hp:
            dmg += (last_hp - snap.health)
            last_hp = snap.health
        act = ctl.step(snap)
        if ctl.success:
            print(f"eng_dist={eng_dist} -> SUCCESS in {f}f! dmg={dmg} final_hp={snap.health}")
            break
        if snap.mode == 17 or f >= 9999:
            print(f"eng_dist={eng_dist} -> FAILED at {f}f dmg={dmg}")
            break
        env.step(act.action)
    env.close()
