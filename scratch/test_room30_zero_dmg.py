from retro_harness.env import make_env
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot
from zelda_i.level4.stepladder import make_room_30_clear_controller
from zelda_i.combat import should_swing_at
from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action

class TunedClear30Controller(type(make_room_30_clear_controller())):
    def step(self, snap):
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
                    if (self.frames % 6) < 3:
                        return FrameAction(nes_action("UP", "A"), "walk_north_band_slash")
                    return FrameAction(nes_action("UP"), "walk_north_band")
                self._set_phase(self.phase.FIGHT, "on_north_band")
        return super().step(snap)

    def _fight_step(self, snap):
        live = self._live(snap)
        if live:
            nearest = min(live, key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y))
            dx = nearest.x - snap.link_x
            dy = nearest.y - snap.link_y
            above = nearest.y < snap.link_y - 6
            if above and abs(dx) > 8:
                direction = "RIGHT" if dx > 0 else "LEFT"
                if should_swing_at(snap.link_x, snap.link_y, direction, live) or abs(dx) <= 24:
                    return self._swing(direction, "align_x_flyer")
        return super()._fight_step(snap)

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
