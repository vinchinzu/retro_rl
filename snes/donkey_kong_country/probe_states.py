"""Probe all DKC save states and report their level_id and RAM values.

Usage::

    uv run python donkey_kong_country/probe_states.py
"""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.resolve()

from retro_harness.env import make_env

GAME = "DonkeyKongCountry-Snes"
GAME_DIR = ROOT_DIR / "donkey_kong_country"
STATES_DIR = GAME_DIR / "custom_integrations" / GAME

# RAM addresses
RAM_LEVEL_ID = 0x003E
RAM_CAMERA_X = 0x00B2
RAM_LIVES = 0x0575
RAM_PLAYER_X = 0x00B4
RAM_PLAYER_Y = 0x00B6

# All registered DKC levels and their expected level_ids
REGISTERED_LEVELS = {
    "JungleHijinks": 0x16,
    "RopeyRampage": 0x0C,
    "WinkysWalkway": 0xD9,
    "MineCartCarnage": 0x2E,
    "BouncyBonanza": 0x07,
}

def probe_state(state_name: str) -> dict | None:
    """Load a state, step a few frames, and read RAM values."""
    try:
        env = make_env(game=GAME, state=state_name, game_dir=GAME_DIR, render_mode=None)
        env.reset()
        base_env = env.unwrapped
        for _ in range(10):
            base_env.step([0] * 12)
        ram = env.get_ram()

        result = {
            "level_id": int(ram[RAM_LEVEL_ID]),
            "camera_x": int(ram[RAM_CAMERA_X]) | (int(ram[RAM_CAMERA_X + 1]) << 8),
            "lives": int(ram[RAM_LIVES]),
            "player_x": int(ram[RAM_PLAYER_X]) | (int(ram[RAM_PLAYER_X + 1]) << 8),
            "player_y": int(ram[RAM_PLAYER_Y]) | (int(ram[RAM_PLAYER_Y + 1]) << 8),
            "nonzero_bytes": sum(1 for b in ram if int(b) != 0),
            "total_bytes": len(ram),
        }
        env.close()
        return result
    except Exception as e:
        return {"error": str(e)}

def main():
    state_files = sorted(STATES_DIR.glob("*.state"))
    if not state_files:
        print("No .state files found in", STATES_DIR)
        return

    print(f"{'State':<45} {'level_id':>10} {'camera_x':>10} {'lives':>6} {'nonzero':>8}")
    print("-" * 85)

    for sf in state_files:
        name = sf.stem
        result = probe_state(name)
        if result is None or "error" in result:
            err = result.get("error", "unknown") if result else "unknown"
            print(f"{name:<45} ERROR: {err}")
            continue

        lid = result["level_id"]
        marker = ""
        if name in REGISTERED_LEVELS:
            expected = REGISTERED_LEVELS[name]
            marker = " <-- OK" if lid == expected else f" <-- MISMATCH (expected 0x{expected:02X})"
        nz = result["nonzero_bytes"]
        total = result["total_bytes"]
        valid = "OK" if nz > 100 else "EMPTY"

        print(
            f"{name:<45} 0x{lid:02X} ({lid:3d}) {result['camera_x']:>10} "
            f"{result['lives']:>6} {nz:>5}/{total} {valid}{marker}"
        )

    print()
    print("Registered levels:")
    for name, expected_lid in sorted(REGISTERED_LEVELS.items()):
        sf = STATES_DIR / f"{name}.state"
        if sf.exists():
            print(f"  [OK] {name}.state exists (expected level_id=0x{expected_lid:02X})")
        else:
            print(f"  [!!] {name}.state MISSING")

if __name__ == "__main__":
    main()
