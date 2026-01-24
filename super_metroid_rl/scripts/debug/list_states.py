import stable_retro as retro
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INTEGRATION_PATH = os.path.join(SCRIPT_DIR, "custom_integrations")
retro.data.Integrations.add_custom_path(INTEGRATION_PATH)

print("Custom Integration Path:", INTEGRATION_PATH)
try:
    states = retro.data.list_states(game="SuperMetroid-Snes")
    print(f"Found {len(states)} states:")
    for s in sorted(states):
        print(f" - {s}")
except Exception as e:
    print(f"Error listing states: {e}")
