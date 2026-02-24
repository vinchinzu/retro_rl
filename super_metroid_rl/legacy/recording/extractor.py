import os
import gzip
import shutil
import stable_retro as retro
from .world import WorldMap
from .stats import Leaderboard

class StateExtractor:
    def __init__(self, state_dir, retro_state_dir, world_map: WorldMap, leaderboard: Leaderboard):
        self.state_dir = state_dir
        self.retro_state_dir = retro_state_dir
        self.world_map = world_map
        self.leaderboard = leaderboard

    def extract_all_transitions(self, bk2_path):
        """
        Replays the entire movie and saves a state at EVERY stable room transition.
        Updates the WorldMap with any new room IDs found.
        Returns a list of (room_name, room_id, state_path) tuples.
        """
        print(f"Loading movie: {bk2_path}")
        
        # Parse Timestamp from filename if possible
        # Format: Name-Timestamp.bk2 or Name-000000-Timestamp.bk2
        from_file_timestamp = None
        base_name = os.path.basename(bk2_path)
        name_part = os.path.splitext(base_name)[0]
        parts = name_part.split('-')
        if len(parts) >= 2 and parts[-1].isdigit():
             # Check if it looks like a timestamp (e.g. > 1600000000)
             ts_candidate = int(parts[-1])
             if ts_candidate > 1600000000:
                 from_file_timestamp = ts_candidate
        
        if from_file_timestamp is None:
             # Fallback to file mtime or current time
             from_file_timestamp = int(os.path.getmtime(bk2_path))

        movie = retro.Movie(bk2_path)
        movie.step()

        env = retro.make(
            game="SuperMetroid-Snes",
            state=None,
            use_restricted_actions=retro.Actions.ALL,
            players=1
        )
        env.initial_state = movie.get_state()
        env.reset()

        print("Replaying... (Scanning for all transitions)")
        
        extracted_states = []
        splits = [] # List of dicts
        
        step = 0
        current_room_stable = None
        current_room_start_frame = 0
        
        gameplay_stable_frames = 0
        FRAME_GAMEPLAY = 8
        prev_room = 0
        split_index = 0 # To preserve order of appearance for same-file splits
        
        while movie.step():
            keys = []
            for i in range(env.num_buttons):
                keys.append(movie.get_key(i, 0))
            _, _, _, _, info = env.step(keys)
            step += 1
            
            curr_room = info.get('room_id', 0)
            curr_game_state = info.get('game_state', 0)

            # 1. Establish initial stable room
            if current_room_stable is None:
                if curr_room != 0:
                    current_room_stable = curr_room
                    current_room_start_frame = step
                    print(f"[{step}] Start Room established: {hex(curr_room)}")
                prev_room = curr_room
                continue

            # 2. Check for transition
            if curr_room != current_room_stable:
                if curr_game_state == FRAME_GAMEPLAY:
                    gameplay_stable_frames += 1
                    
                    if gameplay_stable_frames >= 30: # 0.5s stability
                         print(f"[{step}] Transition detected! {hex(current_room_stable)} -> {hex(curr_room)}")
                         
                         duration_frames = step - current_room_start_frame
                         duration_sec = duration_frames / 60.0
                         
                         # Determine name with FROM context for stats
                         base_prev_name = self.world_map.get_room_name(current_room_stable) or f"{hex(current_room_stable)}"
                         origin_name = self.world_map.get_room_name(prev_room)
                         
                         print(f"DEBUG: Transition {hex(current_room_stable)} -> {hex(curr_room)}")
                         print(f"DEBUG: prev_room={hex(prev_room)} ({origin_name})")
                         print(f"DEBUG: current={hex(current_room_stable)} ({base_prev_name})")
                         
                         display_name = base_prev_name
                         if origin_name:
                             display_name = f"{base_prev_name} [from {origin_name}]"
                         
                         # Leaderboard Update
                         is_pb, diff = self.leaderboard.update_time(display_name, duration_frames, duration_sec, timestamp=from_file_timestamp + split_index)
                         split_index += 1
                         
                         splits.append({
                             "room": display_name,
                             "frames": duration_frames,
                             "seconds": duration_sec,
                             "is_pb": is_pb,
                             "diff": diff
                         })

                         # Define New Room Name (for State Saving)
                         # We can also add context to the state file so 'Climb [from Parlor].state' exists
                         base_new_name = self.world_map.get_room_name(curr_room) 
                         if not base_new_name:
                             clean_prev = self.world_map.get_room_name(current_room_stable)
                             if clean_prev:
                                 inferred = self.world_map.infer_next_room_name(clean_prev)
                                 if inferred:
                                      base_new_name = inferred
                                      self.world_map.add_room(base_new_name, curr_room)
                         
                         if not base_new_name:
                              base_new_name = f"Unknown_Room_{hex(curr_room)}"
                         
                         # Add context to state name
                         state_save_name = base_new_name
                         if base_prev_name: # Use base_prev_name for context, not display_name
                             state_save_name = f"{base_new_name} [from {base_prev_name}]"
                         
                         state_data = env.em.get_state()
                         out_path = self._save_state(state_data, state_save_name)
                         
                         print(f"State saved: {state_save_name}.state")
                         extracted_states.append((base_new_name, curr_room, out_path))
                         
                         prev_room = current_room_stable # The room we just came from
                         current_room_stable = curr_room # The room we just entered
                         current_room_start_frame = step
                         gameplay_stable_frames = 0
                else:
                    gameplay_stable_frames = 0
            else:
                 gameplay_stable_frames = 0


        env.close()
        
        # Add final split
        if current_room_stable:
            duration_frames = step - current_room_start_frame
            duration_sec = duration_frames / 60.0
            base_last_name = self.world_map.get_room_name(current_room_stable) or f"{hex(current_room_stable)}"
            origin_name = self.world_map.get_room_name(prev_room)
            
            display_name = base_last_name
            if origin_name:
                display_name = f"{base_last_name} [from {origin_name}]"
            
            is_pb, diff = self.leaderboard.update_time(display_name, duration_frames, duration_sec, timestamp=from_file_timestamp + split_index)
            
            splits.append({
                "room": display_name,
                "frames": duration_frames,
                "seconds": duration_sec,
                "diff": diff,
                "is_pb": is_pb
            })

        self._print_splits(splits)
        return extracted_states

    def _print_splits(self, splits):
        print("\n" + "="*65)
        print(f"{'Room Name':<20} | {'Time':<10} | {'Diff':<10} | {'Status':<10}")
        print("-" * 65)
        total_frames = 0
        for s in splits:
             pb_str = "NEW PB!" if s['is_pb'] else ""
             diff_str = f"{s['diff']:+d}" if not s['is_pb'] else f"-{abs(s['diff'])}"
             
             print(f"{s['room']:<20} | {s['seconds']:.2f}s     | {diff_str:<10} | {pb_str}")
             total_frames += s['frames']
        print("-" * 65)
        print(f"{'TOTAL':<20} | {total_frames/60.0:.2f}s")
        print("="*65 + "\n")


    def _save_state(self, state_data, state_name):
        # Save to custom dir
        out_path = os.path.join(self.state_dir, f"{state_name}.state")
        with gzip.open(out_path, "wb") as f:
            f.write(state_data)
            
        # Copy to retro dir
        shutil_path = os.path.join(self.retro_state_dir, f"{state_name}.state")
        with gzip.open(shutil_path, "wb") as f:
            f.write(state_data)
        
        print(f"State saved: {os.path.basename(out_path)}")
        return out_path
