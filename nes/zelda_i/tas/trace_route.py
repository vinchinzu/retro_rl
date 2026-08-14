"""Replay an FM2 reference and build a RAM-labelled room screenshot atlas.

This is a planning tool, not product evidence.  It turns a deterministic TAS
button stream into settled room transitions, inventory events, optional
development states, and a contact sheet.  Use the resulting room order and
screenshots to write reactive controllers; never promote the open-loop movie
itself as a Clean or assisted route.

Example::

    uv run python -m zelda_i.tas.trace_route --level 5 --save-states \
        --tag chatterbox_l5
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from retro_harness.env import make_env, reset_obs, write_state_bytes
from retro_harness.segment_runner import configure_headless, write_json_report
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOW,
    ADDR_BRACELET,
    ADDR_CANDLE,
    ADDR_FOOD,
    ADDR_LADDER,
    ADDR_MAP,
    ADDR_RAFT,
    ADDR_ROD,
    ADDR_WHISTLE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)
from zelda_i.tas.fm2 import parse_fm2

DEFAULT_MOVIE = GAME_DIR / "tas" / "ref" / "chatterbox_allitems_4767M.fm2"
DEFAULT_OUT_ROOT = RECORDINGS_DIR / "tas_import"

_INVENTORY_ADDRS: tuple[tuple[str, int], ...] = (
    ("arrows", ADDR_ARROWS),
    ("bow", ADDR_BOW),
    ("candle", ADDR_CANDLE),
    ("whistle", ADDR_WHISTLE),
    ("food", ADDR_FOOD),
    ("rod", ADDR_ROD),
    ("raft", ADDR_RAFT),
    ("ladder", ADDR_LADDER),
    ("bracelet", ADDR_BRACELET),
    ("map", ADDR_MAP),
)


def action_array(frame: Sequence[int]) -> np.ndarray:
    """Return a raw NES-9 action; preserve simultaneous opposites from TAS."""
    values = [1 if int(value) else 0 for value in frame[:9]]
    values.extend([0] * (9 - len(values)))
    return np.asarray(values, dtype=np.int8)


def is_settled_play(snap: ZeldaSnapshot) -> bool:
    """True when room/screen and Link pose are suitable for an atlas pin."""
    return snap.mode == PLAY_MODE and not snap.transitioning and snap.health > 0


def inventory_pin(ram: np.ndarray, snap: ZeldaSnapshot) -> dict[str, int]:
    """Read compact progression values used for gain/change events."""
    values = {name: read_u8(ram, addr) for name, addr in _INVENTORY_ADDRS}
    values.update(
        {
            "sword": snap.sword,
            "bombs": snap.bombs,
            "keys": snap.keys,
            "triforce": snap.triforce,
            "heart_containers": snap.heart_containers,
        }
    )
    return values


@dataclass(frozen=True)
class RouteEvent:
    """One settled room entry or progression-value change."""

    frame: int
    kind: str
    detail: str
    level: int
    room: int
    x: int
    y: int
    health: int
    keys: int
    bombs: int
    triforce: int
    doors: int
    room_item: int
    image: str | None = None
    state: str | None = None

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["room_hex"] = f"0x{self.room:02x}"
        row["health_hex"] = f"0x{self.health:02x}"
        row["triforce_hex"] = f"0x{self.triforce:02x}"
        return row


@dataclass(frozen=True)
class AtlasFrame:
    """Image plus label retained until the contact sheet is written."""

    event_index: int
    label: str
    image: np.ndarray


def event_label(event: RouteEvent) -> str:
    """Make a compact human-readable label for one atlas cell."""
    return (
        f"f{event.frame} {event.kind} L{event.level}:0x{event.room:02X} "
        f"xy={event.x},{event.y} hp={event.health:#04x} "
        f"k={event.keys} b={event.bombs} tf={event.triforce:#04x}"
    )


def build_contact_sheet(
    frames: Sequence[AtlasFrame],
    path: Path,
    *,
    columns: int = 3,
    scale: int = 2,
) -> Path | None:
    """Write labelled nearest-neighbour screenshots in chronological order."""
    if not frames:
        return None
    columns = max(1, int(columns))
    scale = max(1, int(scale))
    first = np.asarray(frames[0].image, dtype=np.uint8)
    if first.ndim != 3 or first.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB atlas frame, got {first.shape}")

    src_h, src_w = first.shape[:2]
    image_w, image_h = src_w * scale, src_h * scale
    label_h = 28
    cell_w, cell_h = image_w, image_h + label_h
    rows = (len(frames) + columns - 1) // columns
    sheet = Image.new("RGB", (cell_w * columns, cell_h * rows), (8, 10, 16))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=10)

    for index, atlas_frame in enumerate(frames):
        row, col = divmod(index, columns)
        x0, y0 = col * cell_w, row * cell_h
        image = Image.fromarray(np.asarray(atlas_frame.image, dtype=np.uint8))
        if scale != 1:
            image = image.resize((image_w, image_h), Image.Resampling.NEAREST)
        sheet.paste(image, (x0, y0))
        draw.text(
            (x0 + 4, y0 + image_h + 3),
            atlas_frame.label,
            fill=(225, 232, 240),
            font=font,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path


def trace_movie(
    movie_path: Path,
    *,
    target_level: int,
    max_frames: int | None,
    output_dir: Path,
    save_states: bool,
    progress_every: int = 10_000,
) -> dict[str, Any]:
    """Replay a power-on FM2 and capture target-level room/inventory events."""
    configure_headless()
    movie = parse_fm2(movie_path)
    stop = movie.num_frames if max_frames is None else min(movie.num_frames, max_frames)
    images_dir = output_dir / "frames"
    images_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    events: list[RouteEvent] = []
    atlas_frames: list[AtlasFrame] = []
    all_room_entries = 0
    last_settled_location: tuple[int, int] | None = None
    previous_inventory: dict[str, int] | None = None

    def add_event(
        *,
        frame: int,
        kind: str,
        detail: str,
        snap: ZeldaSnapshot,
        obs: np.ndarray,
        capture: bool,
    ) -> None:
        image_path: Path | None = None
        state_path: Path | None = None
        event_index = len(events)
        if capture:
            stem = f"e{event_index:03d}_f{frame:06d}_{kind}_L{snap.level}_r{snap.screen:02x}"
            image_path = images_dir / f"{stem}.png"
            Image.fromarray(np.asarray(obs, dtype=np.uint8)).save(image_path)
            if save_states:
                state_name = f"TAS_L{snap.level}_f{frame:06d}_r{snap.screen:02X}"
                state_path = (
                    GAME_DIR / "custom_integrations" / GAME / f"{state_name}.state"
                )
                write_state_bytes(state_path, env.em.get_state())

        event = RouteEvent(
            frame=frame,
            kind=kind,
            detail=detail,
            level=snap.level,
            room=snap.screen,
            x=snap.link_x,
            y=snap.link_y,
            health=snap.health,
            keys=snap.keys,
            bombs=snap.bombs,
            triforce=snap.triforce,
            doors=snap.cur_opened_doors,
            room_item=snap.room_item_id,
            image=str(image_path) if image_path else None,
            state=str(state_path) if state_path else None,
        )
        events.append(event)
        if capture:
            atlas_frames.append(
                AtlasFrame(
                    event_index=event_index,
                    label=event_label(event),
                    image=np.asarray(obs, dtype=np.uint8).copy(),
                )
            )

    try:
        obs, _ = reset_obs(env)
        for frame in range(stop):
            obs, *_ = env.step(action_array(movie.frames[frame]))
            ram = env.get_ram()
            snap = read_snapshot(ram)
            current_inventory = inventory_pin(ram, snap)

            if is_settled_play(snap):
                location = (snap.level, snap.screen)
                if location != last_settled_location:
                    all_room_entries += 1
                    previous = last_settled_location
                    last_settled_location = location
                    if snap.level == target_level:
                        detail = (
                            f"enter L{snap.level}:0x{snap.screen:02x}"
                            if previous is None
                            else (
                                f"L{previous[0]}:0x{previous[1]:02x} -> "
                                f"L{snap.level}:0x{snap.screen:02x}"
                            )
                        )
                        add_event(
                            frame=frame,
                            kind="room_enter",
                            detail=detail,
                            snap=snap,
                            obs=obs,
                            capture=True,
                        )

            if previous_inventory is not None and is_settled_play(snap):
                changes = {
                    name: (previous_inventory[name], value)
                    for name, value in current_inventory.items()
                    if value != previous_inventory[name]
                }
                if changes and snap.level == target_level:
                    detail = ", ".join(
                        f"{name}:{before:#x}->{after:#x}"
                        for name, (before, after) in sorted(changes.items())
                    )
                    add_event(
                        frame=frame,
                        kind="inventory_change",
                        detail=detail,
                        snap=snap,
                        obs=obs,
                        capture=True,
                    )
            previous_inventory = current_inventory

            if progress_every > 0 and (frame + 1) % progress_every == 0:
                print(
                    f"replay {frame + 1}/{stop} "
                    f"L{snap.level}:0x{snap.screen:02x} target_events={len(events)}",
                    flush=True,
                )

        final_snap = read_snapshot(env.get_ram())
    finally:
        env.close()

    contact_sheet = build_contact_sheet(
        atlas_frames,
        output_dir / "contact_sheet.png",
    )
    report = {
        "schema": "zelda_i_tas_route_trace_v1",
        "source": str(movie_path),
        "rom": movie.rom_filename,
        "target_level": target_level,
        "movie_frames": movie.num_frames,
        "frames_played": stop,
        "commands_ignored": True,
        "reference_only": True,
        "target_event_count": len(events),
        "all_room_entries": all_room_entries,
        "visited_target": any(event.level == target_level for event in events),
        "events": [event.to_dict() for event in events],
        "contact_sheet": str(contact_sheet) if contact_sheet else None,
        "final": {
            "mode": final_snap.mode,
            "level": final_snap.level,
            "room": final_snap.screen,
            "room_hex": f"0x{final_snap.screen:02x}",
            "x": final_snap.link_x,
            "y": final_snap.link_y,
            "health": final_snap.health,
            "triforce": final_snap.triforce,
        },
    }
    write_json_report(output_dir / "trace.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movie", type=Path, default=DEFAULT_MOVIE)
    parser.add_argument("--level", type=int, default=5, dest="target_level")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--tag", default="chatterbox_l5")
    parser.add_argument("--save-states", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10_000)
    args = parser.parse_args(argv)

    output_dir = DEFAULT_OUT_ROOT / args.tag
    report = trace_movie(
        args.movie,
        target_level=args.target_level,
        max_frames=args.max_frames,
        output_dir=output_dir,
        save_states=args.save_states,
        progress_every=args.progress_every,
    )
    print(
        f"wrote {output_dir} target_events={report['target_event_count']} "
        f"visited={report['visited_target']}"
    )
    return 0 if report["visited_target"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
