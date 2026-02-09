import json
import os
import re

class WorldMap:
    def __init__(self, map_path):
        self.map_path = map_path
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.map_path):
            with open(self.map_path, 'r') as f:
                return json.load(f)
        return {}

    def save(self):
        with open(self.map_path, 'w') as f:
            json.dump(self.data, f, indent=4, sort_keys=True)

    def get_room_name(self, room_id):
        # Allow room_id to be int or string
        # We want to find a value in self.data that matches room_id
        
        target_int = -1
        target_str = ""
        
        if isinstance(room_id, int):
            target_int = room_id
            target_str = hex(room_id)
        elif isinstance(room_id, str):
            target_str = room_id
            try:
                target_int = int(room_id, 16)
            except:
                pass
                
        for name, val in self.data.items():
            # Val could be int (37368) or string ("0x91f8")
            if val == target_int or val == target_str:
                return name
            # Handle mixed case hex strings?
            if isinstance(val, str) and isinstance(target_str, str):
                if val.lower() == target_str.lower():
                    return name
                    
        return None

    def get_room_id(self, room_name):
        val = self.data.get(room_name)
        if val is None:
            return None
        if isinstance(val, int):
            return val
        if isinstance(val, str):
            try:
                return int(val, 16)
            except:
                return None
        return None

    def add_room(self, room_name, room_id):
        # Always save as hex string for consistency/readability in JSON
        if isinstance(room_id, int):
            hex_id = hex(room_id)
        else:
            hex_id = room_id
            
        current_val = self.data.get(room_name)
        
        # Check if update usage
        # We might have mixed types in file, so robust check
        same = False
        if current_val == hex_id:
            same = True
        elif isinstance(current_val, int) and isinstance(room_id, int) and current_val == room_id:
            same = True
        
        if not same:
            self.data[room_name] = hex_id
            print(f"[WorldMap] Discovered/Updated {room_name} -> {hex_id}")
            self.save()

    def infer_next_room_name(self, current_room_name):
        # Simplistic heuristic: Room15 -> Room16
        match = re.search(r"(Room)(\d+)", current_room_name, re.IGNORECASE)
        if match:
            prefix = match.group(1)
            num = int(match.group(2))
            return f"{prefix}{num + 1}"
        return None
