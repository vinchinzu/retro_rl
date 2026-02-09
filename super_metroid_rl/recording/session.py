import os
import shutil
import stable_retro as retro
from .recorder import DemoRecorder
from .extractor import StateExtractor
from .world import WorldMap
from .stats import Leaderboard
from .manifest import RecordingManifest

class SessionManager:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.game_dir = os.path.join(base_dir, "super_metroid_rl") 
        self.recording_dir = os.path.join(base_dir, "demos")
        
        self.state_dir = os.path.join(base_dir, "custom_integrations", "SuperMetroid-Snes")
        self.map_path = os.path.join(base_dir, "world_map.json")
        self.stats_path = os.path.join(base_dir, "best_times.json")
        self.manifest_path = os.path.join(base_dir, "demos", "recordings.json")
        
        # Ensure Dirs
        os.makedirs(self.recording_dir, exist_ok=True)
        os.makedirs(self.state_dir, exist_ok=True)
        
        # Retro Integ
        retro.data.Integrations.add_custom_path(os.path.join(base_dir, "custom_integrations"))
        self.retro_state_dir = os.path.dirname(retro.data.get_file_path("SuperMetroid-Snes", "rom.sha"))

        # Components
        self.world = WorldMap(self.map_path)
        self.stats = Leaderboard(self.stats_path)
        self.manifest = RecordingManifest(self.manifest_path)
        self.recorder = DemoRecorder(self.recording_dir, self.state_dir)
        self.extractor = StateExtractor(self.state_dir, self.retro_state_dir, self.world, self.stats)

    def recover_stranded_files(self):
        try:
            temp_dirs = [d for d in os.listdir(self.recording_dir) if d.startswith("temp_") and os.path.isdir(os.path.join(self.recording_dir, d))]
            if temp_dirs:
                print(f"Checking for stranded recordings in {len(temp_dirs)} temp folders...")
                for td in temp_dirs:
                    full_td = os.path.join(self.recording_dir, td)
                    files = [f for f in os.listdir(full_td) if f.endswith(".bk2")]
                    for f in files:
                        ts = td.replace("temp_", "")
                        new_name = f"recover-{ts}-{f}"
                        shutil.move(os.path.join(full_td, f), os.path.join(self.recording_dir, new_name))
                        print(f"[RECOVERY] Recovered {new_name}")
                        # Optionally add to manifest as 'recovered'?
                        self.manifest.add_recording(os.path.join(self.recording_dir, new_name), "Unknown(Recovered)", tags=["recovered"])
                    try:
                        shutil.rmtree(full_td)
                    except: pass
        except Exception as e:
            print(f"[RECOVERY ERROR] {e}")

    def run_loop_mode(self, start_state, loop_forever=True):
        """
        Record -> Extract Next -> Loop to Next (if loop_forever)
        """
        current_state = start_state
        
        self.recover_stranded_files()
        
        while True:
            # 1. Record
            print(f"\n>>> Loading State: {current_state} <<<")
            saved_file, action = self.recorder.record_session(current_state)
            
            if saved_file:
                 self.manifest.add_recording(saved_file, current_state, tags=["manual_demo"])
            
            if action == "RESET":
                # Just loop back to same state
                print("Looping... (Restarting Room)")
                continue
                
            elif action == "QUIT":
                if saved_file:
                    print("Session Ended. Checking for auto-extract...")
                    
                    # 2. Extract
                    try:
                        extracted = self.extractor.extract_all_transitions(saved_file)
                        
                        if extracted:
                            # Update Manifest with Analysis
                            route = [e[0] for e in extracted]
                            self.manifest.update_analysis(saved_file, route)

                            # Assuming linear progression, the LAST extracted state is the next start
                            last_state_name = extracted[-1][0]
                            print(f"Next State Identified: {last_state_name}")
                            current_state = last_state_name
                            
                            if not loop_forever:
                                print("Single-pass mode complete. Exiting.")
                                break
                        else:
                            print("No new transitions found in recording.")
                            # If we didn't go anywhere, maybe we want to try again?
                            if not loop_forever:
                                break
                            
                    except Exception as e:
                        print(f"Extraction failed: {e}")
                        break
                else:
                    print("No recording saved yet. Temp dirs are preserved for recovery.")
                    break
            
            if not loop_forever and action != "RESET":
                 break
        
        self.recorder.close()
