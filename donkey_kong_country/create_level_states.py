"""Reusable DKC state creator — play any level, press a key to name it, F5 to save.

Usage::

    uv run python donkey_kong_country/create_level_states.py
    uv run python donkey_kong_country/create_level_states.py --state W1Overworld
    uv run python donkey_kong_country/create_level_states.py --list

Controls:
    Gameplay   : arrow keys + Z(B) X(A) C(Y) S(X) Q(L) W(R)
    TAB        : toggle turbo (10x speed)
    F5         : SAVE state + screenshot with current target name
    F6         : RESUME save (pick up where you left off with --state Resume)
    F9         : CHECKPOINT save (quick mid-level, rewind with F7)
    F7         : RELOAD last checkpoint
    1-6        : pick level slot (within current world, 6 = boss)
    7          : select final boss (King K. Rool)
    9 / 0      : prev / next world
    F4         : type a custom name in terminal
    ESC        : quit

DKC Levels (SNES):
    World 1 — Kongo Jungle         World 2 — Monkey Mines
      1-1  JungleHijinxs             2-1  WinkysWalkway
      1-2  RopeyRampage              2-2  MineCartCarnage
      1-3  ReptileRumble             2-3  BouncyBonanza
      1-4  CoralCapers               2-4  StopAndGoStation
      1-5  BarrelCannonCanyon        2-5  MillstoneMayhem
    World 3 — Vine Valley           World 4 — Gorilla Glacier
      3-1  VultureCulture            4-1  SnowBarrelBlast
      3-2  TreeTopTown               4-2  SlipslideRide
      3-3  ForestFrenzy              4-3  IceAgeAlley
      3-4  TempleTemplest            4-4  CroctopusChase
      3-5  OrangUtanGang             4-5  TorchlightTrouble
    World 5 — Kremkroc Industries   World 6 — Chimp Caverns
      5-1  OilDrumAlley              6-1  TankedUpTrouble
      5-2  TrickTrackTrek            6-2  ManicMincers
      5-3  ElevatorAntics            6-3  MistyMine
      5-4  PoisonPond                6-4  NeckyNutmare
      5-5  MineCartMadness           6-5  LoopyLights
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "x11")

ROOT_DIR = Path(__file__).parent.parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from retro_harness.env import make_env, save_state

GAME = "DonkeyKongCountry-Snes"
GAME_DIR = ROOT_DIR / "donkey_kong_country"
SCREENSHOT_DIR = GAME_DIR / "state_screenshots"

# ---- All DKC levels by world ------------------------------------------------
WORLDS = {
    1: ("Kongo Jungle", [
        "JungleHijinks",
        "RopeyRampage",
        "ReptileRumble",
        "CoralCapers",
        "BarrelCannonCanyon",
        "Boss_VeryGnawty",
    ]),
    2: ("Monkey Mines", [
        "WinkysWalkway",
        "MineCartCarnage",
        "BouncyBonanza",
        "StopAndGoStation",
        "MillstoneMayhem",
        "Boss_MasterNecky",
    ]),
    3: ("Vine Valley", [
        "VultureCulture",
        "TreeTopTown",
        "ForestFrenzy",
        "TempleTemplest",
        "OrangUtanGang",
        "Boss_QueenB",
    ]),
    4: ("Gorilla Glacier", [
        "SnowBarrelBlast",
        "SlipslideRide",
        "IceAgeAlley",
        "CroctopusChase",
        "TorchlightTrouble",
        "Boss_ReallyGnawty",
    ]),
    5: ("Kremkroc Industries", [
        "OilDrumAlley",
        "TrickTrackTrek",
        "ElevatorAntics",
        "PoisonPond",
        "MineCartMadness",
        "Boss_DumbDrum",
    ]),
    6: ("Chimp Caverns", [
        "TankedUpTrouble",
        "ManicMincers",
        "MistyMine",
        "NeckyNutmare",
        "LoopyLights",
        "Boss_MasterNeckySnr",
    ]),
}

# Final boss (not in a world — use F4 or slot 7 trick)
FINAL_BOSS = "Boss_KingKRool"

# RAM addresses
RAM_LEVEL_ID = 0x003E
RAM_CAMERA_X = 0x00B2
RAM_LIVES = 0x0575
RAM_PLAYER_X = 0x00B4
RAM_PLAYER_Y = 0x00B6


def _read_ram(env):
    ram = env.get_ram()
    level_id = int(ram[RAM_LEVEL_ID])
    camera_x = int(ram[RAM_CAMERA_X]) | (int(ram[RAM_CAMERA_X + 1]) << 8)
    lives = int(ram[RAM_LIVES])
    player_x = int(ram[RAM_PLAYER_X]) | (int(ram[RAM_PLAYER_X + 1]) << 8)
    player_y = int(ram[RAM_PLAYER_Y]) | (int(ram[RAM_PLAYER_Y + 1]) << 8)
    return level_id, camera_x, lives, player_x, player_y


def _take_screenshot(env, name: str) -> Path:
    """Capture current frame as PNG."""
    import numpy as np
    try:
        from PIL import Image
    except ImportError:
        print("  (pillow not installed — skipping screenshot)")
        return None

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    obs = env.get_screen()
    img = Image.fromarray(np.asarray(obs))
    path = SCREENSHOT_DIR / f"{name}.png"
    img.save(path)
    return path


def list_levels():
    """Print all known DKC levels."""
    state_dir = GAME_DIR / "custom_integrations" / GAME
    for wnum, (wname, levels) in WORLDS.items():
        print(f"\n  World {wnum} — {wname}")
        for i, lvl in enumerate(levels, 1):
            state_path = state_dir / f"{lvl}.state"
            marker = "OK" if state_path.exists() else "--"
            tag = "BOSS" if lvl.startswith("Boss_") else f"{wnum}-{i}"
            print(f"    {tag:>4}  [{marker}]  {lvl}")

    # Final boss
    fb_path = state_dir / f"{FINAL_BOSS}.state"
    fb_marker = "OK" if fb_path.exists() else "--"
    print(f"\n  Final Boss")
    print(f"       [{fb_marker}]  {FINAL_BOSS}")

    # Resume point
    resume_path = state_dir / "Resume.state"
    if resume_path.exists():
        print(f"\n  Resume point: [OK]  (--state Resume)")
    print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="DKC Level State Creator")
    parser.add_argument("--state", default="QuickSave",
                        help="Starting state to load (default: QuickSave)")
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--list", action="store_true",
                        help="List all known levels and which states exist, then exit")
    args = parser.parse_args()

    if args.list:
        list_levels()
        return

    env = make_env(game=GAME, state=args.state, game_dir=GAME_DIR, render_mode="rgb_array")

    current_world = 1
    current_slot = 1  # 1-indexed
    custom_name = None  # overrides world/slot when set
    saved_states: dict[str, int] = {}

    def _target_name():
        if custom_name:
            return custom_name
        _, levels = WORLDS[current_world]
        return levels[current_slot - 1]

    def _world_line():
        _, levels = WORLDS[current_world]
        parts = []
        for i, lvl in enumerate(levels, 1):
            short = lvl.replace("Boss_", "BOSS:")
            tag = f">{short}<" if i == current_slot and not custom_name else short
            parts.append(f"{i}:{tag}")
        return f"W{current_world} [{' / '.join(parts)}]"

    def on_hud(info: dict) -> list[str]:
        level_id, camera_x, lives, _, player_y = _read_ram(env)
        target = _target_name()
        warn = " !! cam_x>0" if camera_x > 200 else ""
        return [
            f"lid=0x{level_id:02X}({level_id}) cam={camera_x} lives={lives} py={player_y}",
            f"TARGET: {target}{warn}",
            _world_line(),
            "9/0 world  1-6 slot  F5 SAVE  F9 chkpt  F7 reload",
        ]

    from retro_harness.play_session import PlaySession

    session = PlaySession(
        env,
        game_dir=str(GAME_DIR),
        game=GAME,
        scale=args.scale,
        title="DKC State Creator",
    )
    session.on_hud = on_hud

    original_key_handler = session.on_key_down

    def on_key(key):
        import pygame
        nonlocal current_world, current_slot, custom_name

        # 1-6: pick level slot within current world (6 = boss)
        if key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6):
            slot = key - pygame.K_0
            if slot <= len(WORLDS[current_world][1]):
                current_slot = slot
                custom_name = None
                print(f"\n[SLOT] {current_world}-{current_slot}: {_target_name()}")
            return True

        # 7: final boss (King K. Rool)
        if key == pygame.K_7:
            custom_name = FINAL_BOSS
            print(f"\n[TARGET] -> {FINAL_BOSS}")
            return True

        # F6: quick-resume save (always saves as "Resume" — reload with --state Resume)
        if key == pygame.K_F6:
            path = save_state(env, GAME_DIR, GAME, "Resume")
            level_id, camera_x, lives, _, _ = _read_ram(env)
            _take_screenshot(env, "Resume")
            print(f"\n[RESUME SAVED] lid=0x{level_id:02X} cam={camera_x} lives={lives}")
            print(f"  Restart with: --state Resume")
            return True

        # F9: checkpoint save (in-memory + disk), F7/F8 reloads it
        if key == pygame.K_F9:
            session.save_state("Checkpoint")
            level_id, camera_x, lives, _, _ = _read_ram(env)
            print(f"  [CHECKPOINT] lid=0x{level_id:02X} cam={camera_x} lives={lives} — F7 to reload")
            return True

        # 9 / 0: switch world ([ and ] conflict with speed controls)
        if key == pygame.K_9:
            current_world = max(1, current_world - 1)
            current_slot = min(current_slot, len(WORLDS[current_world][1]))
            custom_name = None
            print(f"\n[WORLD] {current_world} — {WORLDS[current_world][0]}")
            return True
        if key == pygame.K_0:
            current_world = min(6, current_world + 1)
            current_slot = min(current_slot, len(WORLDS[current_world][1]))
            custom_name = None
            print(f"\n[WORLD] {current_world} — {WORLDS[current_world][0]}")
            return True

        # F4: custom name
        if key == pygame.K_F4:
            name = input("\nEnter state name: ").strip()
            if name:
                custom_name = name
                print(f"[TARGET] -> {custom_name}")
            return True

        # F5: save state + screenshot
        if key == pygame.K_F5:
            target = _target_name()
            path = save_state(env, GAME_DIR, GAME, target)
            saved_states[target] = saved_states.get(target, 0) + 1

            level_id, camera_x, lives, _, _ = _read_ram(env)
            ss_path = _take_screenshot(env, target)

            print(f"\n{'='*60}")
            print(f"[SAVED] '{target}' -> {path}")
            if ss_path:
                print(f"  screenshot -> {ss_path}")
            print(f"  level_id = 0x{level_id:02X} ({level_id})")
            print(f"  camera_x = {camera_x}")
            print(f"  lives    = {lives}")
            if camera_x > 200:
                print(f"  WARNING: camera_x={camera_x} — are you at the level START?")
            print(f"{'='*60}\n")
            return True

        return original_key_handler(key)

    session.on_key_down = on_key

    print("=" * 60)
    print("DKC Level State Creator")
    print("=" * 60)
    print(f"Starting from: {args.state}")
    print()
    print("Controls:")
    print("  1-6      select level slot in current world (6 = boss)")
    print("  7        select final boss (King K. Rool)")
    print("  9 / 0    prev / next world")
    print("  F4       type custom state name")
    print("  F5       SAVE state + screenshot (named)")
    print("  F6       RESUME save (restart later with --state Resume)")
    print("  F9       CHECKPOINT save (quick mid-level save)")
    print("  F7       RELOAD checkpoint")
    print("  TAB      turbo speed")
    print("  ESC      quit")
    print()
    list_levels()
    print("=" * 60)

    session.run()

    if saved_states:
        print(f"\n{'='*60}")
        print("Session Summary — States saved:")
        for name, count in saved_states.items():
            print(f"  {name}: {count} save(s)")
        print()
        print("Next steps:")
        print("  1. Validate: uv run python donkey_kong_country/probe_states.py")
        print("  2. Register: add level to platformer_common/levels/dkc.py")
        print("  3. Selftest: uv run python -m platformer_common --level <alias> selftest")
        print("=" * 60)


if __name__ == "__main__":
    main()
