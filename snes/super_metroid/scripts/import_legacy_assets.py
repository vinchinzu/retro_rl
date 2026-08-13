"""Import proven room seeds and legacy models from the sibling project."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

_REPO_ROOT = Path(__file__).resolve().parents[3]
from super_metroid.paths import (  # noqa: E402
    EARLY_POLICY_DIR,
    GAME_DIR,
    MAPS_DIR,
    MODELS_DIR,
    POLICY_DIR,
)

LEGACY_DIR = GAME_DIR.parent.parent / "snes_editor" / "super_metroid_rl"
SEEDS = (
    "seg00_landing_site.json",
    "seg01_parlor.json",
    "seg02_climb.json",
    "seg03_pit_room.json",
    "seg04_bb_elev_hallway.json",
    "seg05_morph_ball_room.json",
)
REFERENCE_MAPS = (
    "bomb_torizo.png",
    "brinstar.png",
    "ceres.png",
    "crateria.png",
    "full_game_route.json",
    "landing_site.png",
    "maridia.png",
    "norfair.png",
    "tourian.png",
    "west_ocean.png",
    "wrecked_ship.png",
)

_BUTTON_COUNT = 12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_or_link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if _sha256(target) != _sha256(source):
            raise FileExistsError(f"{target} exists with different content")
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _bk2_actions(path: Path) -> list[list[int]]:
    """Extract stable-retro input-log rows in environment button order."""
    with zipfile.ZipFile(path) as archive:
        log = archive.read("Input Log.txt").decode()
    actions: list[list[int]] = []
    for line in log.splitlines():
        line = line.strip()
        if not line.startswith("|") or line.startswith("["):
            continue
        groups = [group for group in line.split("|") if group]
        if len(groups) < 2 or len(groups[1]) < _BUTTON_COUNT:
            continue
        player_one = groups[1]
        action = [0] * _BUTTON_COUNT
        for column in range(_BUTTON_COUNT):
            if player_one[column] != ".":
                # stable-retro's BK2 columns are the reverse of env order.
                action[_BUTTON_COUNT - 1 - column] = 1
        actions.append(action)
    return actions


def _write_policy(
    target: Path,
    actions: list[list[int]],
    *,
    source: Path,
    source_slice: str,
    policy_id: str,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "raw_buttons": actions,
        "num_frames": len(actions),
        "metadata": {
            "policy_id": policy_id,
            "source": str(source),
            "source_sha256": _sha256(source),
            "source_slice": source_slice,
            "provenance": "legacy manual replay; acceptance requires natural-entry replay",
        },
    }
    target.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"policy: {target}")


def _import_early_policies() -> None:
    landing_runs = LEGACY_DIR / "optimizer" / "runs" / "sm_landing_site"
    segments = landing_runs / "segments"

    # The longer manual Zebes replay naturally collects both early Missile
    # expansions. Frame 4,748 is the Construction Zone transition immediately
    # after Morph Ball; the next recorded action is index 4,749.
    missile_demo = (
        LEGACY_DIR
        / "demos"
        / "SuperMetroid-Snes-ZebesStart-000000-1768921714.bk2"
    )
    missile_actions = _bk2_actions(missile_demo)
    _write_policy(
        EARLY_POLICY_DIR / "two_missile_detour.json",
        missile_actions[4749:],
        source=missile_demo,
        source_slice="[4749:] after Construction Zone entry",
        policy_id="two_missile_detour",
    )

    # These room recordings return from the top of Construction Zone through
    # Morph Ball Room and the Blue Brinstar elevator into Pit Room.
    for source_name, target_name, policy_id in (
        (
            "seg09_morph_ball_room.json",
            "construction_to_elevator.json",
            "construction_and_morph_return",
        ),
        (
            "seg10_bb_elev_hallway.json",
            "elevator_to_pit.json",
            "elevator_return",
        ),
    ):
        source = segments / source_name
        payload = json.loads(source.read_text(encoding="utf-8"))
        _write_policy(
            EARLY_POLICY_DIR / target_name,
            payload["raw_buttons"],
            source=source,
            source_slice="raw_buttons",
            policy_id=policy_id,
        )

    # This successful manual replay begins in Pit Room, climbs continuously,
    # collects Bombs, defeats Bomb Torizo, exits through Flyway, and settles in
    # Parlor. The first 20 frames only settle the source save-state fall; the
    # natural runner recreates and verifies the grounded entry fingerprint.
    torizo_demo = (
        LEGACY_DIR
        / "demos"
        / "recover-1768944402-SuperMetroid-Snes-Room14-000000.bk2"
    )
    torizo_actions = _bk2_actions(torizo_demo)
    _write_policy(
        EARLY_POLICY_DIR / "pit_to_post_torizo.json",
        torizo_actions[20:],
        source=torizo_demo,
        source_slice="[20:] after Pit Room grounded settle",
        policy_id="pit_to_torizo_replay",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--skip-maps", action="store_true")
    args = parser.parse_args()
    if not LEGACY_DIR.is_dir():
        raise FileNotFoundError(LEGACY_DIR)
    seed_source = LEGACY_DIR / "optimizer" / "runs" / "sm_landing_site" / "segments"
    for filename in SEEDS:
        source = seed_source / filename
        target = POLICY_DIR / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        print(f"seed: {target}")
    _import_early_policies()

    if not args.skip_models:
        manifest_path = MODELS_DIR / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest["models"]:
            source = LEGACY_DIR / "models" / item["filename"]
            if _sha256(source) != item["sha256"]:
                raise ValueError(f"hash mismatch for {source}")
            target = MODELS_DIR / "imported" / item["filename"]
            _copy_or_link(source, target)
            print(f"model: {target}")

    if not args.skip_maps:
        map_source = LEGACY_DIR / "maps"
        for filename in REFERENCE_MAPS:
            source = map_source / filename
            target = MAPS_DIR / "legacy" / filename
            _copy_or_link(source, target)
            print(f"map: {target}")
        _copy_or_link(
            LEGACY_DIR / "world_map.json",
            MAPS_DIR / "legacy" / "world_map.json",
        )
        print(f"map: {MAPS_DIR / 'legacy' / 'world_map.json'}")


if __name__ == "__main__":
    main()
