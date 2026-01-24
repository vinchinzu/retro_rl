import stable_retro as retro
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INTEGRATION_PATH = os.path.join(SCRIPT_DIR, "custom_integrations")
retro.data.Integrations.add_custom_path(INTEGRATION_PATH)

env = retro.make(game="SuperMetroid-Snes", state="BossTorizo")
print(f"Buttons: {env.buttons}")
print(f"Action Space: {env.action_space}")
env.close()
