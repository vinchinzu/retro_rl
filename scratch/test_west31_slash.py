from retro_harness.env import make_env
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot
from zelda_i.level4.keyup20 import make_maze_31_west_controller
from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action

class TunedMaze31WestController(type(make_maze_31_west_controller())):
    def _thread(self, xy, snap, wps, next_phase, next_note, *, door_approach=False):
        if self._stall >= 96:
            return self._fail(f"west_solid_{xy[0]}_{xy[1]}")
        from zelda_i.level4.keyup20 import _advance, _west_door_dir, _dir_to, MAZE_31_WEST_DOOR_Y
        self.path_index = _advance(wps, self.path_index, xy)
        if self.path_index >= len(wps):
            self._set_phase(next_phase, next_note)
            return FrameAction(nes_action(), next_note)
        goal = wps[self.path_index]
        if door_approach and goal[1] == MAZE_31_WEST_DOOR_Y:
            stepped = _west_door_dir(xy, goal)
            if stepped is None:
                self.path_index += 1
                return FrameAction(nes_action(), "wp_idle")
            direction, reason = stepped
            if (self.frames % 6) < 3:
                return FrameAction(nes_action(direction, "A"), reason + "_slash")
            return FrameAction(nes_action(direction), reason)
        direction = _dir_to(xy, goal)
        if direction is None:
            self.path_index += 1
            return FrameAction(nes_action(), "wp_idle")
        if (self.frames % 6) < 3:
            return FrameAction(nes_action(direction, "A"), "join_maze_west_slash")
        return FrameAction(nes_action(direction), "join_maze_west")

env = make_env(GAME, "Level4Room31PostLadder", GAME_DIR, render_mode="rgb_array")
env.reset()
snap0 = read_snapshot(env.get_ram())
ctl = TunedMaze31WestController()
last_hp = snap0.health
for f in range(ctl.max_frames):
    snap = read_snapshot(env.get_ram())
    if snap.health < last_hp:
        print(f"  Hit at f={f}: {last_hp} -> {snap.health} (-{last_hp - snap.health}) link=({snap.link_x},{snap.link_y})")
        last_hp = snap.health
    act = ctl.step(snap)
    if ctl.success:
        print(f"Room31 West success at f={f}! final_hp={snap.health} damage={snap0.health - snap.health}")
        break
    if snap.mode == 17:
        print(f"Dead at f={f}!")
        break
    env.step(act.action)
env.close()
