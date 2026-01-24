
import os
import stable_retro as retro

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
INTEGRATION_PATH = os.path.join(REPO_ROOT, "custom_integrations")
ROMS_PATH = os.path.join(REPO_ROOT, "roms")
retro.data.Integrations.add_custom_path(INTEGRATION_PATH)

print(f"Added custom path: {INTEGRATION_PATH}")
print(f"Retro version: {retro.__version__}")

try:
    print("Attempting to find game: SuperMetroid-Snes")
    rom_path = retro.data.get_romfile_path("SuperMetroid-Snes")
    print(f"Found ROM path: {rom_path}")
except Exception as e:
    print(f"Error finding ROM: {e}")

try:
    print("Listing games in custom path...")
    # This might not be exposed easily, but let's try specific lookup
    print(f"Is SuperMetroid-Snes known? {'SuperMetroid-Snes' in retro.data.list_games()}")
    
    # Check if files exist
    expected_rom = os.path.join(ROMS_PATH, "rom.sfc")
    print(f"Checking for file: {expected_rom} -> {os.path.exists(expected_rom)}")
    integration_rom = os.path.join(INTEGRATION_PATH, "SuperMetroid-Snes", "rom.sfc")
    print(f"Integration ROM present: {integration_rom} -> {os.path.exists(integration_rom)}")
    if os.path.exists(expected_rom) and not os.path.exists(integration_rom):
        print("Note: ROM is in roms/. Copy or symlink it into custom_integrations/SuperMetroid-Snes/rom.sfc for stable-retro.")
except Exception as e:
    print(f"Error: {e}")
