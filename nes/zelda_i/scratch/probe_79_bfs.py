import sys
from collections import deque
from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot, PLAY_MODE
from zelda_i.runner import make_assist
from zelda_i.overworld.path import OverworldPathController
from zelda_i.overworld.sword_cave import SwordCaveController, SEGMENT_MAX_FRAMES as SWORD_MAX
from zelda_i.overworld.zd_map import map1_route
from zelda_i.route.chain import boot_to_ready

def main():
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    assist = make_assist(True)
    obs, _ = reset_obs(env)
    obs, _ = boot_to_ready(env, first_playthrough=True, assist=assist)
    
    sword = SwordCaveController()
    while not sword.success:
        snap = read_snapshot(env.get_ram())
        act = sword.step(snap)
        obs, *_ = env.step(act.action)
        assist.apply_env(env)
        
    hops = map1_route().hops[:2]
    walk = OverworldPathController(
        hops=hops,
        require_sword=True,
        farm_below_hearts=0,
        need_rupees=0,
        evade=True,
        max_frames=4000,
    )
    while not walk.success:
        snap = read_snapshot(env.get_ram())
        act = walk.step(snap)
        obs, *_ = env.step(act.action)
        assist.apply_env(env)
        
    snap = read_snapshot(env.get_ram())
    print(f"Entered 0x79 at: screen={snap.screen:#04x}, mode={snap.mode}, x={snap.link_x}, y={snap.link_y}")
    
    # Now Link is on 0x79 at (0, 125).
    # Save the base emulator state.
    base_state = env.em.get_state()
    
    # Let's explore reachable coordinates.
    # To explore quickly without enemy disturbance during BFS, we can either:
    # 1. Clear enemy slots in RAM, or
    # 2. Step 1 frame in each cardinal direction.
    
    visited = {(int(snap.link_x), int(snap.link_y))}
    queue = deque([(int(snap.link_x), int(snap.link_y), base_state)])
    
    max_x = int(snap.link_x)
    max_x_coord = (int(snap.link_x), int(snap.link_y))
    reached_east = False
    
    step_count = 0
    while queue and step_count < 2000:
        step_count += 1
        x, y, state = queue.popleft()
        
        for d in ("RIGHT", "DOWN", "UP", "LEFT"):
            env.em.set_state(state)
            # Step a few frames in direction d to see if Link moves
            act = nes_action(d)
            for _ in range(8):
                env.step(act)
                assist.apply_env(env)
            nxt_snap = read_snapshot(env.get_ram())
            if nxt_snap.mode != PLAY_MODE or nxt_snap.screen != 0x79:
                print(f"Transitioned out of 0x79 with dir {d} from ({x}, {y}) -> screen {nxt_snap.screen:#04x}, mode {nxt_snap.mode}, link=({nxt_snap.link_x}, {nxt_snap.link_y})")
                if nxt_snap.screen == 0x7A:
                    print("SUCCESS! Reached 0x7A!")
                    reached_east = True
                    return
                continue
            
            nx, ny = int(nxt_snap.link_x), int(nxt_snap.link_y)
            if nx > max_x:
                max_x = nx
                max_x_coord = (nx, ny)
                print(f"New max x: {max_x} at ({nx}, {ny}) via {d}")
                if max_x >= 232:
                    print("Reached x >= 232!")
                    reached_east = True
                    return
            
            # Discretize to grid of 4x4 or 2x2 to avoid state explosion
            key = (nx // 4, ny // 4)
            if key not in visited:
                visited.add(key)
                queue.append((nx, ny, env.em.get_state()))
                
    print(f"Done BFS. Max x reached: {max_x} at {max_x_coord}. Reached east? {reached_east}")

if __name__ == "__main__":
    main()
