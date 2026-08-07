"""Doorway-natural entry save-states for isolated room practice (dev only).

Segment start contract
----------------------
Each practice fixture is produced by **door-warping through the catalog entry
door**, then settling Samus **just inside** that doorway (not mid-room, not
stuck in the door shell). That keeps every segment:

* start-aligned with a real door hop (same boundary continuous play will use);
* re-rollable later for enemy/door RNG by re-entering the same door;
* free of full-route loadout freezes that break input after some anchors.

These states are **never** continuous-run evidence.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from retro_harness.actions import idle_action
from super_metroid.dev.common import (
    apply_dev_survivability,
    boot_from_state,
    door_warp,
    free_place_if_stuck,
    make_dev_env,
    place_samus,
    save_dev_state,
)
from super_metroid.dev.route_dev import PHANTOON_ENTRY, ROUTE_FULL
from super_metroid.paths import GAME_DIR, INTEGRATION_DIR, ROOM_PROBLEMS_PATH
from super_metroid.ram import (
    ADDR_MOMENTUM_X,
    ADDR_MOMENTUM_X_SUB,
    ADDR_SAMUS_POSE,
    ADDR_SPEED_COUNTER,
    ADDR_SPEED_FLAG,
    ADDR_VELOCITY_X,
    ADDR_VELOCITY_X_SUB,
    ADDR_VELOCITY_Y,
    ADDR_VELOCITY_Y_SUB,
    GameplayPhase,
    parse_env_state,
    write_wram_u16,
)
from super_metroid.rooms.room_graph import load_problem_catalog, problem_by_id
from super_metroid.rooms.segment_contract import (
    EntryContract,
    entry_door_orientation,
    resolve_entry_door_ptr,
)
from super_metroid.rooms.work_queue import difficulty_score

# Prefer *controllable* mid-game boots. Full late-game topology anchors
# (dev_route_full + grant_route_loadout) can leave Samus input-frozen after
# door_warp; natural post-Spore is the default for doorway fixtures.
_BOOT_CANDIDATES = (
    INTEGRATION_DIR / "natural_post_spore_spawn.state",
    INTEGRATION_DIR / "dev_b1_supers_natural.state",
    INTEGRATION_DIR / "dev_kpdr_red_tower.state",
    PHANTOON_ENTRY,
    INTEGRATION_DIR / "dev_phantoon_entry.state",
    ROUTE_FULL,
    INTEGRATION_DIR / "dev_route_full.state",
)

# Samus pose: 1 = standing facing right, 2 = standing facing left (common).
_POSE_FACE_RIGHT = 1
_POSE_FACE_LEFT = 2

# Inward offset from door lip so the segment starts *inside* the room.
_DOORWAY_INSET_PX = 56

# Extra motion clears (momentum / collision leftovers / speed charge).
# Practice fixtures intentionally zero kinematics — natural continuous hops
# must NOT use this path (door entry speed is route-critical).
_ADDR_POSE_INPUT = 0x0A28
_ADDR_POSE_TURN = 0x0A2A
_ADDR_POSE_MOVEMENT = 0x0A2C

# Off-map / wrap sentinel used by free_place helpers.
_OFF_MAP = 60_000


def _default_boot_state() -> Path:
    for path in _BOOT_CANDIDATES:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "no bootstrap boot state found; expected one of: "
        + ", ".join(str(path) for path in _BOOT_CANDIDATES)
    )


def resolve_entry_door(
    problem: Mapping[str, Any],
    door_map: Mapping[tuple[int, int], int] | None = None,
) -> int | None:
    """Resolve entry door pointer.

    Canonical path is :func:`resolve_entry_door_ptr`. Optional ``door_map`` is
    only for tests that inject a prebuilt ``(source, target) → door_ptr`` table.
    """
    if door_map is not None:
        entry = problem.get("entry")
        if not isinstance(entry, dict):
            return None
        source = entry.get("sourceRoomId")
        if source is None:
            return None
        return door_map.get((int(source), int(problem["roomId"])))
    return resolve_entry_door_ptr(problem)


def build_entry_door_map(
    *,
    reference_root: Path | None = None,
) -> dict[tuple[int, int], int]:
    """Build ``(source, target) → door_ptr`` from the reference connection graph.

    Prefer problem ``entry.doorPtr`` via :func:`resolve_entry_door_ptr` at
    call sites. This map is for bulk coverage checks / tests only.
    """
    from super_metroid.rooms.segment_contract import (
        DEFAULT_REFERENCE_ROOT,
        _load_connections_cached,
    )

    root = reference_root or DEFAULT_REFERENCE_ROOT
    door_map: dict[tuple[int, int], int] = {}
    for connection in _load_connections_cached(root):
        pairs = [(connection.first, connection.second)]
        if connection.direction == "Bidirectional":
            pairs.append((connection.second, connection.first))
        for source, target in pairs:
            if source.door_ptr is None:
                continue
            door_map[(source.room_id, target.room_id)] = source.door_ptr
    return door_map


def doorway_spawn(
    problem: Mapping[str, Any],
    *,
    warp_x: int,
    warp_y: int,
    inset_px: int = _DOORWAY_INSET_PX,
) -> dict[str, Any]:
    """Compute just-inside-doorway spawn from door orientation + warp sample.

    Horizontal doors: keep floor-ish Y from the warp settle; offset X inward.
    Vertical doors: keep X; offset Y inward.

    Requires a known entry door orientation (no silent default).
    """
    orient = entry_door_orientation(problem)
    if orient is None:
        raise ValueError(
            f"problem {problem.get('problemId')!r} entry has no door orientation"
        )
    geo = problem.get("geometry") or {}
    width_px = max(64, int(geo.get("widthBlocks") or 16) * 16)
    height_px = max(64, int(geo.get("heightBlocks") or 16) * 16)

    entry = problem.get("entry") or {}
    endpoint = entry.get("endpoint") if isinstance(entry, dict) else None
    block = None
    if isinstance(endpoint, dict) and endpoint.get("block"):
        block = [int(endpoint["block"][0]), int(endpoint["block"][1])]

    # Discard off-map warp samples (door shell / wrap); prefer geometry.
    if warp_x > _OFF_MAP or warp_y > _OFF_MAP:
        warp_x, warp_y = width_px // 2, max(120, height_px // 2)

    if block is not None:
        door_x = block[0] * 16 + 8
        door_y = block[1] * 16 + 8
    else:
        door_x, door_y = warp_x, warp_y

    if orient == "right":
        # Door on right wall → stand left of lip, face left into room.
        x = min(door_x, width_px - 8) - inset_px
        y = door_y if 32 <= door_y <= height_px - 16 else max(warp_y, 120)
        pose = _POSE_FACE_LEFT
        face = "left"
    elif orient == "left":
        x = max(door_x, 8) + inset_px
        y = door_y if 32 <= door_y <= height_px - 16 else max(warp_y, 120)
        pose = _POSE_FACE_RIGHT
        face = "right"
    elif orient == "down":
        x = door_x if 16 <= door_x <= width_px - 16 else max(warp_x, 64)
        y = min(door_y, height_px - 8) - inset_px
        pose = _POSE_FACE_RIGHT
        face = "right"
    elif orient == "up":
        x = door_x if 16 <= door_x <= width_px - 16 else max(warp_x, 64)
        y = max(door_y, 8) + inset_px
        pose = _POSE_FACE_RIGHT
        face = "right"
    else:
        raise ValueError(f"unsupported door orientation: {orient!r}")

    # Clamp into room interior.
    x = max(24, min(int(x), width_px - 24))
    y = max(40, min(int(y), height_px - 24))
    return {
        "x": x,
        "y": y,
        "pose": pose,
        "face": face,
        "doorOrientation": orient,
        "insetPx": inset_px,
        "doorBlock": block,
        "warpSample": {"x": warp_x, "y": warp_y},
    }


def _clear_motion_and_pose(env: Any, pose: int) -> None:
    """Standing pose + zero speeds so the fixture is controller-ready.

    Clears speed-booster charge and momentum as well — practice segments start
    neutral. Continuous natural-entry hops must retain leave kinematics; do not
    call this from product controllers.
    """
    write_wram_u16(env, ADDR_SAMUS_POSE, pose)
    write_wram_u16(env, ADDR_VELOCITY_Y, 0)
    write_wram_u16(env, ADDR_VELOCITY_Y_SUB, 0)
    write_wram_u16(env, ADDR_VELOCITY_X, 0)
    write_wram_u16(env, ADDR_VELOCITY_X_SUB, 0)
    write_wram_u16(env, ADDR_MOMENTUM_X, 0)
    write_wram_u16(env, ADDR_MOMENTUM_X_SUB, 0)
    write_wram_u16(env, ADDR_SPEED_COUNTER, 0)
    write_wram_u16(env, ADDR_SPEED_FLAG, 0)
    write_wram_u16(env, _ADDR_POSE_INPUT, 0)
    write_wram_u16(env, _ADDR_POSE_TURN, 0)
    write_wram_u16(env, _ADDR_POSE_MOVEMENT, 0)


def _settle_controller_ready(env: Any, pose: int, *, x: int, y: int) -> Any:
    """Idle a few frames for gravity, then re-assert pose/motion before save."""
    for _ in range(12):
        apply_dev_survivability(env)
        env.step(idle_action())
    # Physics may nudge Y and overwrite pose during idle — re-assert contract.
    place_samus(env, x, y)
    _clear_motion_and_pose(env, pose)
    for _ in range(2):
        apply_dev_survivability(env)
        env.step(idle_action())
    _clear_motion_and_pose(env, pose)
    return parse_env_state(env)


def bootstrap_entry_state(
    problem_id: str,
    *,
    catalog_path: Path = ROOM_PROBLEMS_PATH,
    boot_state: Path | None = None,
    overwrite: bool = False,
    settle_frames: int = 900,
    post_warp_idle: int = 20,
    boot_idle_frames: int = 0,
    doorway_inset_px: int = _DOORWAY_INSET_PX,
) -> dict[str, Any]:
    """Door-warp through the entry door and save a just-inside-doorway state.

    ``boot_idle_frames`` is recorded on the entry contract so RNG re-rolls can
    reproduce the same pre-warp wait on a later bootstrap.
    """
    catalog = load_problem_catalog(catalog_path)
    problem = problem_by_id(catalog, problem_id)
    practice = problem["practice"]
    state_path = GAME_DIR / str(practice["stateFile"])
    if state_path.is_file() and not overwrite:
        return {
            "problemId": problem_id,
            "status": "existing_skipped",
            "statePath": str(state_path.resolve()),
            "message": "entry state already exists; pass overwrite=True to replace",
        }

    door_ptr = resolve_entry_door_ptr(problem)
    if door_ptr is None:
        return {
            "problemId": problem_id,
            "status": "no_entry_door",
            "message": "catalog entry missing or no door pointer for source→room",
        }

    boot = (boot_state or _default_boot_state()).expanduser().resolve()
    room_id = int(problem["roomId"])

    env = make_dev_env()
    try:
        boot_from_state(env, boot)
        for _ in range(max(0, boot_idle_frames)):
            apply_dev_survivability(env)
            env.step(idle_action())
        # Survivability only — avoid full late loadout freezes on some anchors.
        apply_dev_survivability(env)
        state = door_warp(
            env,
            door_ptr,
            settle_frames=settle_frames,
            expected_room=room_id,
        )
        if state.room_id != room_id:
            return {
                "problemId": problem_id,
                "status": "warp_failed",
                "doorPtrHex": f"0x{door_ptr:04X}",
                "gotRoomIdHex": f"0x{state.room_id:04X}",
                "expectedRoomIdHex": problem["roomIdHex"],
                "gameState": state.game_state,
            }

        for _ in range(max(1, post_warp_idle)):
            apply_dev_survivability(env)
            env.step(idle_action())
            state = parse_env_state(env)

        warp_x, warp_y = int(state.samus_x), int(state.samus_y)
        try:
            spawn = doorway_spawn(
                problem,
                warp_x=warp_x,
                warp_y=warp_y,
                inset_px=doorway_inset_px,
            )
        except ValueError as exc:
            return {
                "problemId": problem_id,
                "status": "no_orientation",
                "message": str(exc),
            }
        place_samus(env, spawn["x"], spawn["y"])
        _clear_motion_and_pose(env, int(spawn["pose"]))
        state = _settle_controller_ready(
            env,
            int(spawn["pose"]),
            x=int(spawn["x"]),
            y=int(spawn["y"]),
        )

        # Last-resort unstick if still off-map.
        if state.samus_x > _OFF_MAP or state.samus_y > _OFF_MAP:
            free_place_if_stuck(env, int(spawn["x"]), int(spawn["y"]))
            state = _settle_controller_ready(
                env,
                int(spawn["pose"]),
                x=int(spawn["x"]),
                y=int(spawn["y"]),
            )

        if state.room_id != room_id:
            return {
                "problemId": problem_id,
                "status": "place_left_room",
                "gotRoomIdHex": f"0x{state.room_id:04X}",
                "expectedRoomIdHex": problem["roomIdHex"],
            }
        if state.phase is not GameplayPhase.ORDINARY_GAMEPLAY:
            return {
                "problemId": problem_id,
                "status": "not_ordinary",
                "phase": state.phase.value,
                "gameState": state.game_state,
            }

        save_dev_state(env, state_path)
        entry_contract = EntryContract.from_problem(
            problem,
            door_ptr=door_ptr,
            spawn=spawn,
            boot_idle_frames=boot_idle_frames,
        )
        contract_dict = entry_contract.to_dict()
        provenance = {
            "schemaVersion": 2,
            "problemId": problem_id,
            "method": "doorway_natural_bootstrap",
            "bootState": str(boot),
            "doorPtrHex": f"0x{door_ptr:04X}",
            "entrySourceRoomIdHex": entry_contract.entry_source_room_id_hex,
            "entryContract": contract_dict,
            "statePath": str(state_path.resolve()),
            "roomIdHex": problem["roomIdHex"],
            "samusX": state.samus_x,
            "samusY": state.samus_y,
            "pose": state.pose,
            "developmentOnly": True,
            "acceptanceWarning": (
                "Doorway-natural entry fixture (door-warp + just-inside-door "
                "settle). Teleport practice only — never continuous-run evidence."
            ),
            "capturedAt": datetime.now(timezone.utc).isoformat(),
        }
        provenance_path = state_path.with_suffix(".provenance.json")
        provenance_path.write_text(
            json.dumps(provenance, indent=2) + "\n",
            encoding="utf-8",
        )
        return {
            "problemId": problem_id,
            "status": "bootstrapped",
            "statePath": str(state_path.resolve()),
            "provenancePath": str(provenance_path.resolve()),
            "doorPtrHex": f"0x{door_ptr:04X}",
            "roomIdHex": problem["roomIdHex"],
            "samusX": state.samus_x,
            "samusY": state.samus_y,
            "pose": state.pose,
            "entryContract": contract_dict,
            "developmentOnly": True,
        }
    finally:
        env.close()


def bootstrap_entry_states(
    *,
    catalog_path: Path = ROOM_PROBLEMS_PATH,
    queue: int | None = 1,
    max_problems: int | None = None,
    overwrite: bool = False,
    boot_state: Path | None = None,
    boot_idle_frames: int = 0,
    problem_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Bootstrap entry states for a queue slice or explicit problem list."""
    catalog = load_problem_catalog(catalog_path)
    if problem_ids is not None:
        selected = list(problem_ids)
    else:
        selected = [
            str(problem["problemId"])
            for problem in catalog["problems"]
            if queue is None or int(problem.get("queue", 3)) == queue
        ]
        by_id = {str(p["problemId"]): p for p in catalog["problems"]}
        selected.sort(
            key=lambda pid: (
                difficulty_score(by_id[pid]),
                by_id[pid]["roomId"],
            )
        )
    if max_problems is not None:
        selected = selected[: max(0, max_problems)]

    results: list[dict[str, Any]] = []
    for problem_id in selected:
        results.append(
            bootstrap_entry_state(
                problem_id,
                catalog_path=catalog_path,
                boot_state=boot_state,
                overwrite=overwrite,
                boot_idle_frames=boot_idle_frames,
            )
        )

    status_counts: dict[str, int] = {}
    for row in results:
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "schemaVersion": 1,
        "requested": len(selected),
        "statusCounts": status_counts,
        "results": results,
        "developmentOnly": True,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
    }
