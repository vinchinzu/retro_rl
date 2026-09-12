from retro_harness.env import make_env
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot
from zelda_i.level4.map21 import make_room_20_clear_controller
from zelda_i.combat import should_swing_at
from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action

class SafeClear20Controller(type(make_room_20_clear_controller())):
    def _fight_step(self, snap):
        self.combat_frames += 1
        live = self._live(snap)
        self.last_live_enemies = len(live)
        self.max_live_enemies = max(self.max_live_enemies, len(live))
        if not live and self.max_live_enemies >= 5:
            self.success = True
            self._set_phase(self.phase.DONE, "room_cleared")
            return FrameAction(nes_idle_action(), "done")
        y = int(snap.link_y)
        if y > 200:
            return FrameAction(nes_action("UP"), "join_south_band")
        if y < 192:
            return FrameAction(nes_action("DOWN"), "join_south_band")
        if not live:
            return FrameAction(nes_idle_action(), "wait_spawn")
        nearest = min(live, key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y))
        dx = nearest.x - snap.link_x
        dy = nearest.y - snap.link_y
        above = nearest.y < 192
        if above and abs(dx) <= 28:
            # When flyer is above, slash UP, but don't walk north into water/flyer
            if (self.combat_frames % 6) < 3:
                return FrameAction(nes_action("UP", "A"), "slash_up_flyer")
            # When not swinging, face UP without walking UP into contact
            return FrameAction(nes_action("DOWN" if y < 196 else "UP"), "hold_south_band")
        direction = "RIGHT" if dx > 0 else "LEFT"
        if abs(dx) > 8:
            if should_swing_at(snap.link_x, snap.link_y, direction, live) or abs(dx) <= 24:
                return self._swing(direction, "align_x_vire")
            return FrameAction(nes_action(direction), "align_x_vire")
        direction = "UP" if above or dy <= 0 else "DOWN"
        if should_swing_at(snap.link_x, snap.link_y, direction, (nearest,)) or abs(dx) <= 16:
            return self._swing(direction, "engage")
        tx = (48, 88, 120, 160, 200)[self.patrol_index % 5]
        if abs(snap.link_x - tx) <= 6:
            self.patrol_index += 1
            tx = (48, 88, 120, 160, 200)[self.patrol_index % 5]
        return FrameAction(nes_action("RIGHT" if snap.link_x < tx else "LEFT"), "patrol_south")

env = make_env(GAME, "Level4Room20", GAME_DIR, render_mode="rgb_array")
env.reset()
snap0 = read_snapshot(env.get_ram())
ctl = SafeClear20Controller()
last_hp = snap0.health
for f in range(ctl.max_frames):
    snap = read_snapshot(env.get_ram())
    if snap.health < last_hp:
        print(f"  Hit at f={f}: {last_hp} -> {snap.health} (-{last_hp - snap.health}) link=({snap.link_x},{snap.link_y})")
        last_hp = snap.health
    act = ctl.step(snap)
    if ctl.success:
        print(f"Room20 success at f={f}! final_hp={snap.health} damage={snap0.health - snap.health}")
        break
    if snap.mode == 17:
        print(f"Dead at f={f}!")
        break
    env.step(act.action)
env.close()
