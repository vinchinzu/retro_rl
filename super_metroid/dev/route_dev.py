"""Full-game route skeleton via door-warp hops (boss fights skipped).

Development-only. Walks the research completion-sequence room paths using
bank-``$83`` door definitions, granting loadout/boss bits so gray doors and
post-boss unlocks do not block the topology probe.

Boss *fights* are intentionally skipped: when a leg starts at a boss room we
set the corresponding boss bit and continue. The goal is a continuous
room-to-room warp chain covering the full research completion sequence
(Ceres → Morph → … → Mother Brain → Escape → Landing Site), including the
late-game spine:

```text
Phantoon → Gravity → Botwoon → Draygon → Ridley
→ Statues → Tourian elev → Mother Brain
→ Escape 4 → Landing Site
```

Hop data:

- ``maps/full_route_hops.json`` — all 22 legs
- ``maps/late_game_route_hops.json`` — late 9-leg subset

The single null door hop (Ceres ship ``0xDF45 → 0x91F8``) is substituted with
a known bank-``$83`` door into Landing Site for topology probes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from collections.abc import Callable

from retro_harness.actions import idle_action
from super_metroid.dev.common import (
    apply_dev_survivability,
    boot_from_state,
    door_warp,
    free_place_if_stuck,
    make_dev_env,
    place_samus,
    save_dev_state,
    state_summary,
)
from super_metroid.paths import INTEGRATION_DIR, MAPS_DIR, RECORDINGS_DIR
from super_metroid.ram import (
    parse_env_state,
    read_bank7e_wram,
    set_event_flag,
    write_wram_u8,
    write_wram_u16,
)
from super_metroid.video import FrameVideoWriter, concat_videos

FrameSink = Callable[[Any], None]

DEFAULT_TOUR_VIDEO = RECORDINGS_DIR / "full_route_tour.mp4"
DEFAULT_TOUR_REPORT = RECORDINGS_DIR / "full_route_tour.json"
DEFAULT_HYBRID_VIDEO = RECORDINGS_DIR / "full_route_hybrid.mp4"
DEFAULT_HYBRID_REPORT = RECORDINGS_DIR / "full_route_hybrid.json"
DEFAULT_CONTINUOUS_PREFIX_VIDEO = RECORDINGS_DIR / "start_to_supers.mp4"
# After continuous Super collect, warp the rest of the research path.
DEFAULT_HYBRID_START_LEG = "spore_spawn_supers"
DEFAULT_FRAMES_PER_ROOM = 36

LATE_HOPS_PATH = MAPS_DIR / "late_game_route_hops.json"
FULL_HOPS_PATH = MAPS_DIR / "full_route_hops.json"
FULL_GRAPH_PATH = MAPS_DIR / "full_room_graph.json"

# Near-complete mid/late loadout for route probing (development only).
# Morph/bombs/varia/gravity/hijump/space/speed/screw/spring/grapple-ish.
ROUTE_ITEMS = 0xF32F
# Charge + ice + wave + plasma (Tourian metroids).
ROUTE_BEAMS = 0x100B

# Ceres ship → Landing Site has no bank-$83 door; substitute a known LS entry.
# 0x896A is Parlor → Landing Site (also used by tourian_escape_4 finish hop).
NULL_DOOR_SUBSTITUTES: dict[tuple[int, int], int] = {
    (0xDF45, 0x91F8): 0x896A,
}

# boss_bits_for_area offsets from $7E:D828
BOSS_BIT_AREA = {
    "crateria": 0,
    "brinstar": 1,
    "norfair": 2,
    "wrecked_ship": 3,
    "maridia": 4,
    "tourian": 5,
}

# Anchor id → (area_index, bit_mask) for skip-boss writes.
BOSS_SKIP_FLAGS: dict[str, tuple[int, int]] = {
    "bomb_torizo": (0, 0x04),  # Crateria bomb-torizo bit (dev door unlocks)
    "spore_spawn": (1, 0x02),
    "kraid": (1, 0x01),
    "phantoon": (3, 0x01),
    "botwoon": (4, 0x02),
    "draygon": (4, 0x01),
    "ridley": (2, 0x01),
    "mother_brain": (5, 0x02),
}

# When leaving a boss anchor, also mark event flags that unlock follow-on doors.
# Event 0x0E = Mother Brain defeated / escape start.
EVENT_ON_BOSS_SKIP: dict[str, tuple[int, ...]] = {
    "mother_brain": (0x0E, 3, 4, 5),
}

# Item bits (collected/equipped at 0x09A2/0x09A4).
ITEM_MORPH = 0x0004
ITEM_BOMBS = 0x1000
ITEM_VARIA = 0x0001
ITEM_SPEED = 0x2000
# Beam bits at 0x09A6/0x09A8.
BEAM_ICE = 0x0002

# Default free-air placement after each hop (x, y). Overridden per-room as needed.
DEFAULT_PLACE = (120, 180)
ROOM_PLACE: dict[int, tuple[int, int]] = {
    0x91F8: (500, 300),  # Landing Site
    0x92FD: (200, 180),  # Parlor / Climb
    0x9E9F: (80, 180),  # Morph Ball Room
    0xA107: (120, 180),  # First Missile / Blue Brinstar
    0x9804: (120, 180),  # Bomb Torizo
    0x9DC7: (200, 180),  # Spore Spawn
    0x9B5B: (120, 180),  # Spore Super room
    0x9E11: (120, 180),  # Early Power Bombs
    0xA59F: (200, 300),  # Kraid
    0xA6E2: (120, 180),  # Varia Suit Room
    0xAD1B: (120, 180),  # Speed Booster
    0xA890: (120, 180),  # Ice Beam
    0xCD13: (140, 180),  # Phantoon
    0xCE40: (80, 180),  # Gravity
    0xD95E: (100, 180),  # Botwoon
    0xDA60: (120, 180),  # Draygon
    0xB32E: (140, 200),  # Ridley
    0xA66A: (128, 180),  # Statues
    0xDAAE: (128, 100),  # Tourian elev
    0xDD58: (880, 180),  # Mother Brain
    0xDE4D: (400, 100),  # Escape 1
    0xDEDE: (200, 180),  # Escape 4
    0xDF45: (200, 180),  # Ceres Ship
}

# Ordered late-game skeleton (skip early continuous-verified prefix).
LATE_LEG_ORDER: tuple[tuple[str, str], ...] = (
    ("phantoon", "gravity_suit"),
    ("gravity_suit", "botwoon"),
    ("botwoon", "draygon"),
    ("draygon", "ridley"),
    ("ridley", "statues"),
    ("statues", "tourian_elevator"),
    ("tourian_elevator", "mother_brain"),
    ("mother_brain", "tourian_escape_4"),
    ("tourian_escape_4", "landing_site_finish"),
)

# Full research completion sequence (22 legs).
FULL_LEG_ORDER: tuple[tuple[str, str], ...] = (
    ("ceres_elevator", "ceres_ridley"),
    ("ceres_ridley", "landing_site"),
    ("landing_site", "morph_ball"),
    ("morph_ball", "first_missile"),
    ("first_missile", "bomb_torizo"),
    ("bomb_torizo", "spore_spawn"),
    ("spore_spawn", "spore_spawn_supers"),
    ("spore_spawn_supers", "early_power_bombs"),
    ("early_power_bombs", "kraid"),
    ("kraid", "varia"),
    ("varia", "speed_booster"),
    ("speed_booster", "ice_beam"),
    ("ice_beam", "phantoon"),
    ("phantoon", "gravity_suit"),
    ("gravity_suit", "botwoon"),
    ("botwoon", "draygon"),
    ("draygon", "ridley"),
    ("ridley", "statues"),
    ("statues", "tourian_elevator"),
    ("tourian_elevator", "mother_brain"),
    ("mother_brain", "tourian_escape_4"),
    ("tourian_escape_4", "landing_site_finish"),
)

# Anchors at/after which full late loadout is safe/required for topology.
_FULL_LOADOUT_AFTER = {
    "morph_ball",
    "first_missile",
    "bomb_torizo",
    "spore_spawn",
    "spore_spawn_supers",
    "early_power_bombs",
    "kraid",
    "varia",
    "speed_booster",
    "ice_beam",
    "phantoon",
    "gravity_suit",
    "botwoon",
    "draygon",
    "ridley",
    "statues",
    "tourian_elevator",
    "mother_brain",
    "tourian_escape_4",
    "landing_site_finish",
}

PHANTOON_ENTRY = INTEGRATION_DIR / "dev_phantoon_entry.state"
NATURAL_POST_SPORE = INTEGRATION_DIR / "natural_post_spore_spawn.state"
ROUTE_RIDLEY_ENTRY = INTEGRATION_DIR / "dev_route_ridley_entry.state"
ROUTE_MB_ENTRY = INTEGRATION_DIR / "dev_route_mother_brain_entry.state"
ROUTE_FULL_LATE = INTEGRATION_DIR / "dev_route_late_full.state"
ROUTE_FULL = INTEGRATION_DIR / "dev_route_full.state"


def load_late_hops(path: Path = LATE_HOPS_PATH) -> dict[str, list[dict[str, Any]]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_full_hops(path: Path = FULL_HOPS_PATH) -> dict[str, list[dict[str, Any]]]:
    """Load the full 22-leg hop table (same shape as ``load_late_hops``)."""
    return json.loads(path.read_text(encoding="utf-8"))


def leg_key(source: str, target: str) -> str:
    return f"{source}__{target}"


def hops_for_leg(
    source: str,
    target: str,
    *,
    hops_data: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Resolve hop chain for a leg (prefer explicit table, else late then full)."""
    key = leg_key(source, target)
    if hops_data is not None:
        return hops_data[key]
    late = load_late_hops()
    if key in late:
        return late[key]
    return load_full_hops()[key]


def grant_route_loadout(env: Any) -> None:
    """Development inventory/ammo for full late-game topology probes."""
    write_wram_u16(env, 0x09A2, ROUTE_ITEMS)
    write_wram_u16(env, 0x09A4, ROUTE_ITEMS)
    write_wram_u16(env, 0x09A6, ROUTE_BEAMS)
    write_wram_u16(env, 0x09A8, ROUTE_BEAMS)
    write_wram_u16(env, 0x09C4, 1499)
    write_wram_u16(env, 0x09C2, 1499)
    write_wram_u16(env, 0x09C8, 230)
    write_wram_u16(env, 0x09C6, 230)
    write_wram_u16(env, 0x09CC, 50)
    write_wram_u16(env, 0x09CA, 50)
    write_wram_u16(env, 0x09D0, 50)
    write_wram_u16(env, 0x09CE, 50)
    write_wram_u16(env, 0x09D4, 400)
    write_wram_u16(env, 0x09D6, 400)
    write_wram_u16(env, 0x09D2, 1)


def apply_tour_loadout(env: Any, *, stage: str = "late") -> None:
    """Stage-based loadout for full-route topology (development only).

    Stages:
    - ``early``: survivability only (no full item dump).
    - ``mid`` / ``late``: full ``grant_route_loadout``.
    """
    if stage == "early":
        write_wram_u16(env, 0x09C4, 99)
        write_wram_u16(env, 0x09C2, 99)
        return
    grant_route_loadout(env)


def _or_items(env: Any, mask: int) -> None:
    items = int.from_bytes(read_bank7e_wram(env)[0x09A2 : 0x09A2 + 2], "little")
    collected = int.from_bytes(read_bank7e_wram(env)[0x09A4 : 0x09A4 + 2], "little")
    write_wram_u16(env, 0x09A2, items | mask)
    write_wram_u16(env, 0x09A4, collected | mask)


def _or_beams(env: Any, mask: int) -> None:
    beams = int.from_bytes(read_bank7e_wram(env)[0x09A6 : 0x09A6 + 2], "little")
    collected = int.from_bytes(read_bank7e_wram(env)[0x09A8 : 0x09A8 + 2], "little")
    write_wram_u16(env, 0x09A6, beams | mask)
    write_wram_u16(env, 0x09A8, collected | mask)


def _ensure_ammo_max(env: Any, max_addr: int, cur_addr: int, minimum: int) -> None:
    wram = read_bank7e_wram(env)
    current_max = int.from_bytes(wram[max_addr : max_addr + 2], "little")
    if current_max < minimum:
        write_wram_u16(env, max_addr, minimum)
        write_wram_u16(env, cur_addr, minimum)


def apply_anchor_progress(env: Any, anchor_id: str) -> None:
    """Apply item/capacity/boss grants when leaving (or arriving at) an anchor.

    Progressive grants keep early video narrative cleaner; once Morph is left
    we switch to full late loadout for topology reliability.
    """
    if anchor_id in BOSS_SKIP_FLAGS:
        skip_boss(env, anchor_id)

    if anchor_id == "morph_ball":
        _or_items(env, ITEM_MORPH)
    elif anchor_id == "first_missile":
        _or_items(env, ITEM_MORPH)
        _ensure_ammo_max(env, 0x09C8, 0x09C6, 5)
    elif anchor_id == "bomb_torizo":
        _or_items(env, ITEM_MORPH | ITEM_BOMBS)
        _ensure_ammo_max(env, 0x09C8, 0x09C6, 5)
    elif anchor_id in ("spore_spawn", "spore_spawn_supers"):
        _or_items(env, ITEM_MORPH | ITEM_BOMBS)
        _ensure_ammo_max(env, 0x09CC, 0x09CA, 5)
    elif anchor_id == "early_power_bombs":
        _or_items(env, ITEM_MORPH | ITEM_BOMBS)
        _ensure_ammo_max(env, 0x09D0, 0x09CE, 5)
        _ensure_ammo_max(env, 0x09CC, 0x09CA, 5)
    elif anchor_id in ("kraid", "varia"):
        _or_items(env, ITEM_MORPH | ITEM_BOMBS | ITEM_VARIA)
        _ensure_ammo_max(env, 0x09D0, 0x09CE, 5)
    elif anchor_id == "speed_booster":
        _or_items(env, ITEM_MORPH | ITEM_BOMBS | ITEM_VARIA | ITEM_SPEED)
    elif anchor_id == "ice_beam":
        _or_items(env, ITEM_MORPH | ITEM_BOMBS | ITEM_VARIA | ITEM_SPEED)
        _or_beams(env, BEAM_ICE)

    # After Morph, prefer full loadout so gray doors / heat / water never block.
    if anchor_id in _FULL_LOADOUT_AFTER:
        grant_route_loadout(env)
        if anchor_id in (
            "ice_beam",
            "phantoon",
            "gravity_suit",
            "botwoon",
            "draygon",
            "ridley",
            "statues",
            "tourian_elevator",
            "mother_brain",
            "tourian_escape_4",
        ):
            mark_all_major_bosses(env)


def set_boss_bit(env: Any, area_index: int, mask: int) -> None:
    """OR one boss bit into ``boss_bits_for_area[area]`` (development)."""
    address = 0xD828 + area_index
    current = int(read_bank7e_wram(env)[address])
    write_wram_u8(env, address, current | mask)


def skip_boss(env: Any, anchor_id: str) -> None:
    """Mark a boss defeated without fighting (door unlocks / route probes)."""
    if anchor_id in BOSS_SKIP_FLAGS:
        area, mask = BOSS_SKIP_FLAGS[anchor_id]
        set_boss_bit(env, area, mask)
    for event_id in EVENT_ON_BOSS_SKIP.get(anchor_id, ()):
        set_event_flag(env, event_id)


def mark_all_major_bosses(env: Any) -> None:
    """Set Kraid/Phantoon/Draygon/Ridley (+Spore/Botwoon) for statue unlocks."""
    for anchor in ("spore_spawn", "kraid", "phantoon", "botwoon", "draygon", "ridley"):
        skip_boss(env, anchor)


def resolve_hop_door(hop: dict[str, Any]) -> tuple[int, bool, str | None]:
    """Return ``(door_ptr, null_substituted, substituted_hex)``.

    When ``hop['door']`` is null/empty, look up ``NULL_DOOR_SUBSTITUTES`` by
    ``(from, to)`` room ids.
    """
    from_room = int(hop["from"], 0)
    to_room = int(hop["to"], 0)
    raw = hop.get("door")
    if raw is None or raw == "" or raw == "null":
        sub = NULL_DOOR_SUBSTITUTES.get((from_room, to_room))
        if sub is None:
            raise ValueError(
                f"null door hop with no substitute: {hop['from']} → {hop['to']}"
            )
        return sub, True, f"0x{sub:04X}"
    return int(raw, 0), False, None


def _place_for_room(env: Any, room_id: int) -> None:
    x, y = ROOM_PLACE.get(room_id, DEFAULT_PLACE)
    free_place_if_stuck(env, x, y)
    state = parse_env_state(env)
    if state.samus_x > 60000 or state.samus_y > 60000:
        place_samus(env, x, y)
        for _ in range(15):
            apply_dev_survivability(env)
            env.step(idle_action())


def _idle_and_capture(
    env: Any,
    frames: int,
    *,
    frame_sink: FrameSink | None = None,
) -> None:
    """Idle with survivability; optionally push RGB frames to ``frame_sink``."""
    for _ in range(max(0, frames)):
        apply_dev_survivability(env)
        obs, _, _, _, _ = env.step(idle_action())
        if frame_sink is not None:
            frame_sink(obs)


def run_hop_chain(
    env: Any,
    hops: list[dict[str, Any]],
    *,
    place: bool = True,
    reapply_loadout: bool = True,
    frame_sink: FrameSink | None = None,
    frames_per_room: int = 0,
) -> list[dict[str, object]]:
    """Door-warp each hop; return per-hop results.

    Null ``door`` entries are substituted via ``NULL_DOOR_SUBSTITUTES`` and
    reported with ``nullDoor: true`` / ``substitutedDoor``.

    When ``frame_sink`` is set, capture ``frames_per_room`` idle frames after
    each successful hop settle (development tour video).
    """
    results: list[dict[str, object]] = []
    for hop in hops:
        door, is_null, sub_hex = resolve_hop_door(hop)
        expect = int(hop["to"], 0)
        if reapply_loadout:
            grant_route_loadout(env)
        state = door_warp(env, door, expected_room=expect)
        ok = state.room_id == expect and state.game_state == 8
        if place and state.room_id == expect:
            _place_for_room(env, expect)
            state = parse_env_state(env)
            # Placement can leave multi-screen loads mid-settle; re-check room only.
            ok = state.room_id == expect
        report_state = state
        if ok and frame_sink is not None and frames_per_room > 0:
            _idle_and_capture(env, frames_per_room, frame_sink=frame_sink)
        entry: dict[str, object] = {
            "from": hop["from"],
            "to": hop["to"],
            "door": None if is_null else hop["door"],
            "success": ok,
            "gotRoomHex": f"0x{report_state.room_id:04X}",
            "gameState": report_state.game_state,
            "samusX": report_state.samus_x,
            "samusY": report_state.samus_y,
        }
        if is_null:
            entry["nullDoor"] = True
            entry["substitutedDoor"] = sub_hex
        results.append(entry)
        if not ok:
            break
    return results


def run_leg(
    env: Any,
    source: str,
    target: str,
    *,
    hops_data: dict[str, list[dict[str, Any]]] | None = None,
    skip_source_boss: bool = True,
    frame_sink: FrameSink | None = None,
    frames_per_room: int = 0,
) -> dict[str, object]:
    """Run one completion-sequence leg as door-warp hops."""
    hops = hops_for_leg(source, target, hops_data=hops_data)
    if skip_source_boss:
        apply_anchor_progress(env, source)
        for _ in range(5):
            apply_dev_survivability(env)
            env.step(idle_action())
    hop_results = run_hop_chain(
        env,
        hops,
        frame_sink=frame_sink,
        frames_per_room=frames_per_room,
    )
    success = bool(hop_results) and all(h["success"] for h in hop_results)
    state = parse_env_state(env)
    return {
        "source": source,
        "target": target,
        "success": success,
        "hopCount": len(hops),
        "hopsCompleted": sum(1 for h in hop_results if h["success"]),
        "hops": hop_results,
        "finalRoomIdHex": f"0x{state.room_id:04X}",
        "developmentOnly": True,
    }


def run_late_route(
    *,
    source_state: Path = PHANTOON_ENTRY,
    legs: tuple[tuple[str, str], ...] = LATE_LEG_ORDER,
    stop_after: str | None = None,
    save_checkpoints: bool = True,
) -> dict[str, object]:
    """Door-warp the late-game skeleton from Phantoon entry through ``stop_after``.

    Boss fights are skipped via boss-bit writes when leaving boss anchors.
    """
    env = make_dev_env()
    hops_data = load_late_hops()
    leg_reports: list[dict[str, object]] = []
    try:
        if not source_state.exists():
            # Fall back to red tower + warp into Phantoon via existing helper.
            from super_metroid.dev.phantoon_dev import capture_phantoon_entry

            capture_phantoon_entry(output=source_state)

        boot_from_state(env, source_state)
        grant_route_loadout(env)
        # Phantoon entry may not have PBs/Gravity yet; loadout covers topology.
        mark_all_major_bosses(env)  # safe default; statue doors need all four
        # Re-clear Phantoon-area bit then re-set when leaving — already set ok.
        for _ in range(5):
            apply_dev_survivability(env)
            env.step(idle_action())

        for source, target in legs:
            report = run_leg(
                env,
                source,
                target,
                hops_data=hops_data,
                skip_source_boss=True,
            )
            leg_reports.append(report)
            if save_checkpoints and report["success"]:
                save_dev_state(
                    env,
                    INTEGRATION_DIR / f"dev_route_anchor_{target}.state",
                )
                if target == "ridley":
                    save_dev_state(env, ROUTE_RIDLEY_ENTRY)
                if target == "mother_brain":
                    save_dev_state(env, ROUTE_MB_ENTRY)
            if not report["success"]:
                break
            if stop_after is not None and target == stop_after:
                break

        if all(r["success"] for r in leg_reports) and (
            stop_after is None or leg_reports[-1]["target"] == stop_after
        ):
            save_dev_state(env, ROUTE_FULL_LATE)

        final = state_summary(env)
        return {
            "success": bool(leg_reports) and all(r["success"] for r in leg_reports),
            "legs": leg_reports,
            "final": final,
            "ridleyState": str(ROUTE_RIDLEY_ENTRY) if ROUTE_RIDLEY_ENTRY.exists() else None,
            "motherBrainState": str(ROUTE_MB_ENTRY) if ROUTE_MB_ENTRY.exists() else None,
            "developmentOnly": True,
        }
    finally:
        env.close()


def default_full_source_state() -> Path:
    """Pick any ordinary-gameplay state that boots (door-warp ignores room)."""
    for candidate in (
        NATURAL_POST_SPORE,
        PHANTOON_ENTRY,
        ROUTE_FULL_LATE,
        INTEGRATION_DIR / "dev_route_anchor_landing_site_finish.state",
    ):
        if candidate.exists():
            return candidate
    # Last resort: first *.state under integration that is not empty.
    states = sorted(INTEGRATION_DIR.glob("*.state"))
    if states:
        return states[0]
    return NATURAL_POST_SPORE


def run_full_route(
    *,
    source_state: Path | None = None,
    legs: tuple[tuple[str, str], ...] = FULL_LEG_ORDER,
    stop_after: str | None = None,
    save_checkpoints: bool = True,
    start_from_leg: str | None = None,
    video_path: Path | None = None,
    frames_per_room: int = DEFAULT_FRAMES_PER_ROOM,
    report_path: Path | None = None,
) -> dict[str, object]:
    """Door-warp the full 22-leg completion sequence (boss fights skipped).

    Development-only topology probe. Null Ceres-ship door is substituted.
    ``start_from_leg`` skips legs until the source anchor matches.

    When ``video_path`` is set, record ``frames_per_room`` RGB frames after each
    successful hop (door transitions themselves are not encoded — room tour only).
    """
    env = make_dev_env()
    hops_data = load_full_hops()
    leg_reports: list[dict[str, object]] = []
    src = source_state if source_state is not None else default_full_source_state()
    writer: FrameVideoWriter | None = None
    encoded_frames = 0
    rooms_seen: list[str] = []
    result: dict[str, object] = {"success": False, "developmentOnly": True}
    try:
        if not src.exists():
            from super_metroid.dev.phantoon_dev import capture_phantoon_entry

            capture_phantoon_entry(output=PHANTOON_ENTRY)
            src = PHANTOON_ENTRY

        boot_from_state(env, src)
        # Full loadout from the start is OK for topology reliability; early
        # progressive grants still run when leaving anchors.
        grant_route_loadout(env)
        mark_all_major_bosses(env)
        for _ in range(5):
            apply_dev_survivability(env)
            env.step(idle_action())

        frame_sink: FrameSink | None = None
        if video_path is not None:
            probe_obs, _, _, _, _ = env.step(idle_action())
            writer = FrameVideoWriter(
                video_path,
                width=int(probe_obs.shape[1]),
                height=int(probe_obs.shape[0]),
            )
            writer.write(probe_obs)
            encoded_frames = 1

            def frame_sink(obs: Any, _w: FrameVideoWriter = writer) -> None:
                nonlocal encoded_frames
                _w.write(obs)
                encoded_frames += 1

        started = start_from_leg is None
        for source, target in legs:
            if not started:
                if source == start_from_leg:
                    started = True
                else:
                    continue
            report = run_leg(
                env,
                source,
                target,
                hops_data=hops_data,
                skip_source_boss=True,
                frame_sink=frame_sink,
                frames_per_room=frames_per_room if frame_sink else 0,
            )
            for hop in report.get("hops", []):
                if hop.get("success"):
                    room_hex = str(hop.get("to"))
                    if not rooms_seen or rooms_seen[-1] != room_hex:
                        rooms_seen.append(room_hex)
            leg_reports.append(report)
            if save_checkpoints and report["success"]:
                save_dev_state(
                    env,
                    INTEGRATION_DIR / f"dev_route_anchor_{target}.state",
                )
                if target == "ridley":
                    save_dev_state(env, ROUTE_RIDLEY_ENTRY)
                if target == "mother_brain":
                    save_dev_state(env, ROUTE_MB_ENTRY)
            if not report["success"]:
                break
            if stop_after is not None and target == stop_after:
                break

        full_success = bool(leg_reports) and all(r["success"] for r in leg_reports)
        complete = full_success and (
            stop_after is None or leg_reports[-1]["target"] == stop_after
        )
        if complete and stop_after is None:
            save_dev_state(env, ROUTE_FULL)

        final = state_summary(env)
        result = {
            "success": full_success,
            "legs": leg_reports,
            "final": final,
            "sourceState": str(src),
            "startFromLeg": start_from_leg,
            "legsAttempted": len(leg_reports),
            "legsSucceeded": sum(1 for r in leg_reports if r["success"]),
            "roomsVisited": rooms_seen,
            "roomVisitCount": len(rooms_seen),
            "uniqueRoomCount": len(set(rooms_seen)),
            "bossFights": "skipped",
            "mode": "warp_tour",
            "fullState": str(ROUTE_FULL) if ROUTE_FULL.exists() else None,
            "developmentOnly": True,
        }
        if video_path is not None:
            result["video"] = {
                "path": str(video_path),
                "encodedFrames": encoded_frames,
                "framesPerRoom": frames_per_room,
                "fps": 60,
            }
    finally:
        if writer is not None:
            writer.close()
        env.close()

    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )
        result["reportPath"] = str(report_path)
    return result


def run_hybrid_full_route(
    *,
    source_state: Path | None = None,
    stop_after: str | None = None,
    save_checkpoints: bool = True,
    start_from_leg: str | None = DEFAULT_HYBRID_START_LEG,
    video_path: Path | None = DEFAULT_HYBRID_VIDEO,
    frames_per_room: int = DEFAULT_FRAMES_PER_ROOM,
    report_path: Path | None = DEFAULT_HYBRID_REPORT,
    splice_prefix: Path | None = DEFAULT_CONTINUOUS_PREFIX_VIDEO,
    tour_video_path: Path | None = None,
) -> dict[str, object]:
    """Continuous Super prefix video + door-warp tour to Landing Site.

    Development-only full-route recording:
    1. Uses existing continuous ``start_to_supers`` (or another prefix clip).
    2. Door-warps from ``spore_spawn_supers`` through finish.
    3. Skips all boss fights via boss-bit writes.
    4. ffmpeg-concatenates prefix + tour into one hybrid video.

    Not continuous acceptance evidence (warps + progression writes on suffix).
    """
    if video_path is None:
        video_path = DEFAULT_HYBRID_VIDEO
    if report_path is None:
        report_path = DEFAULT_HYBRID_REPORT
    if start_from_leg is None:
        start_from_leg = DEFAULT_HYBRID_START_LEG

    prefix_path = Path(splice_prefix) if splice_prefix is not None else None
    if prefix_path is not None and not prefix_path.is_file():
        raise FileNotFoundError(
            f"hybrid prefix video missing: {prefix_path}. "
            "Record it with: uv run python super_metroid/scripts/record/continuous.py --to supers"
        )

    tour_path = (
        Path(tour_video_path)
        if tour_video_path is not None
        else Path(video_path).with_name(
            f"{Path(video_path).stem}_tour_only{Path(video_path).suffix}"
        )
    )

    # Record warp suffix only (prefix is a separate continuous clip).
    result = run_full_route(
        source_state=source_state,
        stop_after=stop_after,
        save_checkpoints=save_checkpoints,
        start_from_leg=start_from_leg,
        video_path=tour_path,
        frames_per_room=frames_per_room,
        report_path=None,
    )
    result["tourVideo"] = result.get("video")
    result["bossFights"] = "skipped"

    if prefix_path is not None:
        tour_file = Path(str((result.get("video") or {}).get("path") or tour_path))
        if not tour_file.is_file():
            raise RuntimeError(f"hybrid tour video was not written: {tour_file}")
        splice = concat_videos([prefix_path, tour_file], video_path)
        result["mode"] = "hybrid_continuous_prefix_plus_warp_tour"
        result["continuousPrefix"] = {
            "path": str(prefix_path.resolve()),
            "through": "spore_spawn_supers",
            "note": "continuous power-on → Super collect (separate recording)",
        }
        result["video"] = {
            "path": str(Path(video_path).resolve()),
            "splice": splice,
            "fps": 60,
        }
    else:
        result["mode"] = "warp_tour_suffix_only"
        if result.get("video"):
            # No prefix: promote tour file to the requested hybrid path name.
            result["video"] = {
                **result["video"],
                "path": str(Path(video_path).resolve()),
            }

    result["developmentOnly"] = True
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )
        result["reportPath"] = str(report_path)
    return result


def run_phantoon_to_ridley() -> dict[str, object]:
    """Convenience: Phantoon → … → Ridley (fights skipped)."""
    return run_late_route(
        legs=(
            ("phantoon", "gravity_suit"),
            ("gravity_suit", "botwoon"),
            ("botwoon", "draygon"),
            ("draygon", "ridley"),
        ),
        stop_after="ridley",
    )


def run_ridley_to_mother_brain(
    *,
    source_state: Path = ROUTE_RIDLEY_ENTRY,
) -> dict[str, object]:
    """Convenience: Ridley → statues → Tourian → Mother Brain (fights skipped)."""
    if not source_state.exists():
        # Produce ridley entry first.
        pre = run_phantoon_to_ridley()
        if not pre["success"]:
            return {"success": False, "phase": "phantoon_to_ridley", **pre}
    return run_late_route(
        source_state=source_state,
        legs=(
            ("ridley", "statues"),
            ("statues", "tourian_elevator"),
            ("tourian_elevator", "mother_brain"),
        ),
        stop_after="mother_brain",
    )


def summarize_full_graph_legs() -> list[dict[str, object]]:
    """List research completion legs with hop counts from hop JSON."""
    graph = json.loads(FULL_GRAPH_PATH.read_text(encoding="utf-8"))
    late_hops = load_late_hops()
    full_hops = load_full_hops()
    rows = []
    for leg in graph["completionSequence"]["legs"]:
        key = leg_key(leg["sourceAnchor"], leg["targetAnchor"])
        hop_list = full_hops.get(key)
        late_list = late_hops.get(key)
        null_doors = None
        if hop_list is not None:
            null_doors = sum(
                1
                for h in hop_list
                if h.get("door") is None or h.get("door") == "" or h.get("door") == "null"
            )
        rows.append(
            {
                "source": leg["sourceAnchor"],
                "target": leg["targetAnchor"],
                "rooms": len(leg["roomPath"]),
                "doorHops": len(hop_list) if hop_list else None,
                "inLateHopTable": late_list is not None,
                "inFullHopTable": hop_list is not None,
                "nullDoors": null_doors,
            }
        )
    return rows
