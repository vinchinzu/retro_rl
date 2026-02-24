import json
import os
import time

class Leaderboard:
    def __init__(self, stats_path):
        self.stats_path = stats_path
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.stats_path):
            try:
                with open(self.stats_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print("Warning: Could not parse stats file. Starting fresh.")
                return {}
        return {}

    def save(self):
        with open(self.stats_path, 'w') as f:
            json.dump(self.data, f, indent=4, sort_keys=True)

    def update_time(self, room_name, frames, seconds, timestamp=None):
        """
        Updates the best time for a room.
        Returns: (is_pb, difference_frames)
        """
        if timestamp is None:
            timestamp = int(time.time())
        
        # Initialize if not present
        if room_name not in self.data:
            self.data[room_name] = {
                "best_frames": frames,
                "best_seconds": seconds,
                "history": [] 
            }
            # First time history
            self.data[room_name]["history"].append({
                "timestamp": timestamp,
                "frames": frames,
                "seconds": seconds
            })
            self.save()
            return True, 0 # First time is always a PB!

        # Append to history if not duplicate
        if "history" not in self.data[room_name]:
            self.data[room_name]["history"] = []
            
        # Check for duplicate timestamp in history to prevent double-counting re-runs
        existing = next((item for item in self.data[room_name]["history"] if item["timestamp"] == timestamp), None)
        if not existing:
            self.data[room_name]["history"].append({
                "timestamp": timestamp,
                "frames": frames,
                "seconds": seconds
            })
        else:
             # Basic update if values changed? For now just ignore
             pass

        current_best = self.data[room_name]["best_frames"]
        
        # Check for PB
        if frames < current_best:
            diff = frames - current_best
            self.data[room_name]["best_frames"] = frames
            self.data[room_name]["best_seconds"] = seconds
            self.save()
            return True, diff
            
        self.save() # Save mainly for the history update
        return False, frames - current_best

    def get_best(self, room_name):
        entry = self.data.get(room_name)
        if entry:
            return entry["best_frames"], entry["best_seconds"]
        return None, None
