import json
import os
import time
import uuid

class RecordingManifest:
    def __init__(self, manifest_path):
        self.path = manifest_path
        self.data = self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []

    def save(self):
        with open(self.path, 'w') as f:
            json.dump(self.data, f, indent=4)

    def add_recording(self, filename, start_state, tags=None):
        record = {
            "id": str(uuid.uuid4()),
            "filename": os.path.basename(filename),
            "timestamp": int(time.time()),
            "start_state": start_state,
            "tags": tags or [],
            "analyzed": False,
            "route": []
        }
        self.data.append(record)
        self.save()
        return record["id"]

    def update_analysis(self, filename, route_data):
        """
        route_data: list of room names or splits
        """
        basename = os.path.basename(filename)
        for record in self.data:
            if record["filename"] == basename:
                record["route"] = route_data
                record["analyzed"] = True
                self.save()
                return True
        return False
