from retro_harness.env import make_env
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot
from zelda_i.level4.path import make_room_32_clear_controller
from zelda_i.level4.dungeon import ROOM_32_SPEC, CombatTuning
from zelda_i.dungeon.engine import DungeonPhase
import dataclasses

with open("nes/zelda_i/custom_integrations/LegendOfZelda-Nes/Level4Room32.state", "rb") as f:
    state_bytes = f.read()

for backstep in (0, 16, 20, 24, 28):
    for avoid_w in (False, True):
        for dom_axis in (False, True):
            for eng_dist in (56, 72):
                env = make_env(GAME, "Level4Room32", GAME_DIR, render_mode="rgb_array")
                env.reset()
                env.em.set_state(state_bytes)
                
                combat = CombatTuning(
                    patrol=ROOM_32_SPEC.combat.patrol,
                    engage_distance=eng_dist,
                    engage_dominant_axis=dom_axis,
                    attack_phase=4,
                    engage_attack_period=6,
                    engage_attack_hold=3,
                    patrol_attack_period=10,
                    patrol_attack_hold=3,
                    contact_backstep=backstep,
                    avoid_walls=avoid_w,
                    avoid_wall_bounds=(56, 200, 109, 173) if avoid_w else (40, 216, 93, 189),
                )
                spec = dataclasses.replace(ROOM_32_SPEC, combat=combat)
                ctl = make_room_32_clear_controller()
                ctl.spec = spec
                ctl.phase = DungeonPhase.FIGHT
                
                snap0 = read_snapshot(env.get_ram())
                last_hp = snap0.health
                dmg = 0
                for f in range(5000):
                    snap = read_snapshot(env.get_ram())
                    if snap.health < last_hp:
                        dmg += (last_hp - snap.health)
                        last_hp = snap.health
                    act = ctl.step(snap)
                    if ctl.success:
                        print(f"back={backstep:2d} avoid={avoid_w} dom={dom_axis} eng={eng_dist} -> SUCCESS in {f:4d}f! dmg={dmg} final_hp={snap.health}")
                        break
                    if snap.mode == 17 or f >= 4999:
                        break
                    env.step(act.action)
                env.close()
