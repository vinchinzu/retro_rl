from retro_harness.env import make_env
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot
from zelda_i.level4.stepladder import make_room_30_clear_controller
from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action

class TunedClear30Controller(type(make_room_30_clear_controller())):
    def step(self, snap):
        # Override TO_BAND to swing UP intermittently
        if self.phase.name == "TO_BAND":
            self.frames += 1
            self.phase_frames += 1
            if snap.link_y <= 149 and abs(snap.link_x - 120) <= 40:
                self._set_phase(self.phase.FIGHT, "on_north_band")
            else:
                if abs(snap.link_x - 120) > 6 and snap.link_y > 160:
                    return FrameAction(
                        nes_action("RIGHT" if snap.link_x < 120 else "LEFT"),
                        "center_x_south",
                    )
                if snap.link_y > 133:
                    # Swing UP intermittently while walking north!
                    if (self.frames % 6) < 3:
                        return FrameAction(nes_action("UP", "A"), "walk_north_band_slash")
                    return FrameAction(nes_action("UP"), "walk_north_band")
                self._set_phase(self.phase.FIGHT, "on_north_band")
        return super().step(snap)

env = make_env(GAME, "Level4Room30", GAME_DIR, render_mode="rgb_array")
env.reset()
snap0 = read_snapshot(env.get_ram())
ctl = TunedClear30Controller()
last_hp = snap0.health
for f in range(ctl.max_frames):
    snap = read_snapshot(env.get_ram())
    if snap.health < last_hp:
        print(f"  Hit at f={f}: {last_hp} -> {snap.health} (-{last_hp - snap.health}) link=({snap.link_x},{snap.link_y})")
        last_hp = snap.health
    act = ctl.step(snap)
    if ctl.success:
        print(f"Room30 success at f={f}! final_hp={snap.health} damage={snap0.health - snap.health}")
        break
    if snap.mode == 17:
        print(f"Dead at f={f}!")
        break
    env.step(act.action)
env.close()
