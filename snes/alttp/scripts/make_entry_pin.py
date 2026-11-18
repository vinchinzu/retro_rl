"""Rebuild one dungeon ``*Entry`` pin from Hyrule Castle grounds.

``make_entry_pin.py <slug>`` loads ``HyruleCastleGrounds``, pokes that
dungeon's bring-in kit, triggers ``$010E``, waits for indoor control, and
saves the existing dev pin with provenance ``poke``. The kit is the AP-040
requirement tokens on the entrance and exit gates, plus Lamp when the
dungeon requires ``hasLightCone`` and does not award it. Items this dungeon
awards (Eastern bow, big keys) are not poked. ``$F3CC`` is never written.

A save-state pin is not continuous. The tool refuses to overwrite the pin
when the loaded room is not the room that pin has today. When the pin file
is absent and ``EntrySpec.room`` quotes the indoor room, it creates that
pin only if the load is module ``0x07`` in that room's palace, not the castle.

    cd snes && SDL_VIDEODRIVER=dummy uv run python alttp/scripts/make_entry_pin.py eastern
    cd snes && SDL_VIDEODRIVER=dummy uv run python alttp/scripts/make_entry_pin.py ice
    cd snes && SDL_VIDEODRIVER=dummy uv run python alttp/scripts/make_entry_pin.py skull
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from alttp.paths import GAME, GAME_DIR, HYRULE_CASTLE_GROUNDS_STATE
from alttp.ram import FOLLOWER, INVENTORY_FIELDS, assign_u8, read_snapshot
from alttp.scripts.z3_dungeon_extract import load_route
from alttp.spine.tape import read_pin_provenance, write_pin_provenance

_PROGRESS = 0xF3C5
_ABILITY = 0xF379
# Measured Ice Palace byte: lift-2, swim, and dash (probe apply_pokes).
_ABILITY_LIFT2_SWIM_DASH = 0x1E
_FIELDS = {field.name: field for field in INVENTORY_FIELDS}

# Helper or item token → the vanilla item it stands for. A token whose item
# is awarded by this dungeon's own locations is not an entry poke.
_ITEM = {
    "Lamp": "Lamp",
    "hasLightCone": "Lamp",
    "Bow": "Bow",
    "canShootArrows": "Bow",
    "FireRod": "FireRod",
    "canMeltThings": "FireRod",
    "Flippers": "Flippers",
    "canSwim": "Flippers",
    "TitansMitt": "TitansMitt",
    "canLiftDarkRocks": "TitansMitt",
    "PowerGlove": "PowerGlove",
    "canLiftRocks": "PowerGlove",
    "PegasusBoots": "PegasusBoots",
    "canDash": "PegasusBoots",
    "MoonPearl": "MoonPearl",
    "isDarkLink": "MoonPearl",
    "Hammer": "Hammer",
    "Hookshot": "Hookshot",
    "canGrapple": "Hookshot",
    "CaneOfSomaria": "CaneOfSomaria",
    "Cape": "Cape",
    "CaneOfByrna": "CaneOfByrna",
    "canInvul": "Cape",
    "RescueZelda": "RescueZelda",
}

# Inventory field writes for a token we keep. Progress is not an item field.
_TOKEN_FIELDS: dict[str, tuple[tuple[str, int], ...]] = {
    "Lamp": (("lamp", 1),),
    "hasLightCone": (("lamp", 1),),
    "FireRod": (("fire_rod", 1), ("magic", 0x80)),
    "canMeltThings": (("fire_rod", 1), ("magic", 0x80)),
    "Flippers": (("flippers", 1),),
    "canSwim": (("flippers", 1),),
    "TitansMitt": (("gloves", 2),),
    "canLiftDarkRocks": (("gloves", 2),),
    "PowerGlove": (("gloves", 1),),
    "canLiftRocks": (("gloves", 1),),
    "PegasusBoots": (("boots", 1),),
    "canDash": (("boots", 1),),
    "MoonPearl": (("pearl", 1),),
    "isDarkLink": (("pearl", 1),),
    "RescueZelda": (("progress", 3),),
}


@dataclass(frozen=True)
class EntrySpec:
    slug: str
    pin: str
    entrance: int
    room: int | None = None


# Saved names are the existing dev pins, not ``<Slug>Entry`` titles.
# ``room`` is set only for a pin that does not exist yet.
ENTRIES: dict[str, EntrySpec] = {
    "eastern": EntrySpec("eastern", "EasternPalaceEntry", 0x08),
    "ice": EntrySpec("ice", "IcePalaceEntry", 0x2D),
    # $010E 0x29 loaded room 0x57 (second section). First-section door is the next index.
    "skull": EntrySpec("skull", "SkullWoodsEntry", 0x2A, 0x58),
}

# Module_PreDungeon, then the spotlight-close path Overworld_UseEntrance uses.
_MODULES = (0x06, 0x0F)
# cur_palace_index_x2. Castle is 0x02 (ram.py). Not a Skull Woods load.
_CASTLE_DUNGEON = 0x02


@dataclass(frozen=True)
class Poke:
    offset: int
    value: int
    name: str


def _awarded(doc: dict) -> set[str]:
    found: set[str] = set()
    for loc in doc.get("locations") or []:
        item = loc.get("item")
        if isinstance(item, str):
            found.add(item)
        elif isinstance(item, list):
            found.update(str(part) for part in item)
    return found


def _gate_tokens(doc: dict) -> list[str]:
    """Items on entrance rooms and overworld exits. Not interior doors."""
    tokens: list[str] = []
    for gate in doc.get("gates") or []:
        name = str(gate.get("name", "")).lower()
        if "entrance" not in name and not name.endswith(" exit"):
            continue
        for item in gate.get("items") or []:
            token = str(item)
            if token not in tokens:
                tokens.append(token)
    return tokens


def _dark_exit(doc: dict) -> bool:
    for row in doc.get("connections") or []:
        if row.get("internal"):
            continue
        dest = str(row.get("destination", ""))
        if dest.startswith("Dark"):
            return True
    return False


def bring_in_tokens(doc: dict) -> list[str]:
    """Entrance-gate tokens, plus Lamp when this dungeon does not award it.

    Chest rewards (Eastern bow, big keys) stay out. Interior alternatives
    such as hammer or hookshot are not on the entrance gate, so they stay out.
    """
    awarded = _awarded(doc)
    gates = _gate_tokens(doc)
    required = [
        str(row.get("name", ""))
        for row in (doc.get("requirements") or {}).get("items") or []
    ]
    chosen: list[str] = []
    for token in gates + required:
        if not token or token.startswith("mods."):
            continue
        item = _ITEM.get(token)
        if item is None:
            if token in gates:
                raise SystemExit(f"no kit write for entrance token {token}")
            continue
        if item in awarded:
            continue
        if token not in gates and item not in ("Lamp", "RescueZelda"):
            continue
        if token not in _TOKEN_FIELDS:
            raise SystemExit(f"no kit write for {token}")
        if token not in chosen:
            chosen.append(token)
    return chosen


def kit_pokes(slug: str) -> list[Poke]:
    """Requirement kit plus the grounds support those tokens need to function.

    HyruleCastleGrounds has no sword. Eastern's probe sets fighter sword,
    shield, and progress 1. Ice's probe sets the master sword, eight hearts,
    moon pearl, boots, and the lift/swim/dash ability byte with the entrance
    trio (titans, fire rod, flippers).
    """
    if slug not in ENTRIES:
        known = ", ".join(sorted(ENTRIES))
        raise SystemExit(f"unknown slug {slug!r}; known: {known}")
    doc = load_route(slug)
    tokens = bring_in_tokens(doc)
    by_offset: dict[int, Poke] = {}

    def put(offset: int, value: int, name: str) -> None:
        off = int(offset)
        if off == int(FOLLOWER) or off < 0:
            raise SystemExit(f"refusing to write $F3CC via {name}")
        prev = by_offset.get(off)
        if prev is not None and prev.value > int(value) and prev.name == name:
            return
        by_offset[off] = Poke(off, int(value) & 0xFF, name)

    def put_field(field_name: str, value: int) -> None:
        if field_name == "progress":
            put(_PROGRESS, value, "progress")
            return
        field = _FIELDS[field_name]
        if field.size != 1:
            raise SystemExit(f"{field_name} is not a one-byte kit field")
        put(field.offset, value, field_name)

    for token in tokens:
        for field_name, value in _TOKEN_FIELDS[token]:
            put_field(field_name, value)
    ice_trio = {"canLiftDarkRocks", "canMeltThings", "canSwim"} <= set(tokens)
    if ice_trio:
        put_field("sword", 2)
        put_field("armor", 0)
        put_field("boots", 1)
        put_field("max_hp", 0x40)
        put_field("hp", 0x40)
        put(_ABILITY, _ABILITY_LIFT2_SWIM_DASH, "ability")
        if _dark_exit(doc):
            put_field("pearl", 1)
    else:
        put_field("sword", 1)
        put_field("shield", 1)
        if _PROGRESS not in by_offset:
            put(_PROGRESS, 1, "progress")
    if any(poke.offset == int(FOLLOWER) for poke in by_offset.values()):
        raise SystemExit("refusing to write $F3CC")
    return [by_offset[off] for off in sorted(by_offset)]


def _refresh(env: object) -> None:
    data = getattr(env, "data", None)
    refresh = getattr(data, "update_ram", None)
    if refresh is not None:
        refresh()


def _read_u8(env: object, offset: int) -> int:
    from alttp.ram import wram_index

    ram = env.get_ram()  # type: ignore[attr-defined]
    return int(ram[wram_index(offset)])


def apply_pokes(env: object, pokes: list[Poke]) -> None:
    """Write the kit, then two idle frames, and require the bytes to stick.

    Does not write ``$F3CC``. The idle pair is the probe's settle, not a
    save-state resume.
    """
    from alttp.startup import no_action, step_frames

    follower_before = _read_u8(env, FOLLOWER)
    for poke in pokes:
        assign_u8(env, poke.offset, poke.value)
        if _read_u8(env, poke.offset) != poke.value:
            raise SystemExit(
                f"{poke.name} ${poke.offset:04X} read "
                f"{_read_u8(env, poke.offset)}, wanted {poke.value}"
            )
    step_frames(env, no_action(), 2)
    for poke in pokes:
        got = _read_u8(env, poke.offset)
        if got != poke.value:
            raise SystemExit(
                f"{poke.name} ${poke.offset:04X} settled {got}, wanted {poke.value}"
            )
    if _read_u8(env, FOLLOWER) != follower_before:
        raise SystemExit(
            f"$F3CC changed {follower_before} -> {_read_u8(env, FOLLOWER)} during kit poke"
        )


def trigger_entrance(env: object, entrance: int, *, module: int) -> None:
    """``$010E`` plus Module_PreDungeon (or spotlight). Real room-load."""
    assign_u8(env, 0x010E, entrance & 0xFF)
    assign_u8(env, 0x010F, 0)
    assign_u8(env, 0x010C, 0x06)
    assign_u8(env, 0x0011, 0)
    assign_u8(env, 0x00B0, 0)
    assign_u8(env, 0x0010, module)


def wait_indoor_control(env: object, *, max_frames: int) -> tuple[bool, int]:
    """Idle until indoors, module ``0x07``, and controllable. Dismiss text."""
    from alttp.startup import action_for, no_action, step_frames

    frames = 0
    idle = no_action()
    while frames < max_frames:
        step_frames(env, idle, 4)
        frames += 4
        _refresh(env)
        snap = read_snapshot(env.get_ram())  # type: ignore[attr-defined]
        if snap.is_text_mode:
            step_frames(env, action_for("A"), 2)
            step_frames(env, idle, 2)
            frames += 4
            continue
        if (
            snap.indoors
            and snap.has_control
            and not snap.is_hold_up_item
            and snap.game_mode == 0x07
        ):
            return True, frames
    return False, frames


def _snap_room(env: object) -> tuple[int, int, bool, int]:
    _refresh(env)
    snap = read_snapshot(env.get_ram())  # type: ignore[attr-defined]
    return (
        int(snap.room_base_id) & 0xFF,
        int(snap.dungeon_id) & 0xFF,
        bool(snap.indoors),
        int(snap.game_mode) & 0xFF,
    )


def _load_named(env: object, name: str) -> None:
    from retro_harness.env import read_state_bytes, resync_custom_state

    env.reset()  # type: ignore[attr-defined]
    if not resync_custom_state(env, GAME_DIR, GAME, name):
        path_hint = name
        raise SystemExit(f"failed to load {path_hint}")
    # resync can no-op on a bad gzip; set_state again from the bytes we trust.
    from retro_harness.env import state_path

    path = state_path(GAME_DIR, GAME, name)
    if not path.is_file():
        raise SystemExit(f"missing pin {name}")
    env.em.set_state(read_state_bytes(path))  # type: ignore[attr-defined]
    _refresh(env)


def _today_room(env: object, pin: str) -> int:
    """Room id on the pin file as it exists now. No step, no write."""
    _load_named(env, pin)
    room, _dungeon, indoors, _module = _snap_room(env)
    if not indoors:
        raise SystemExit(f"{pin} is not indoors; refusing to rebuild")
    return room


def _load_grounds(env: object) -> None:
    from alttp.primitives import settle_control

    _load_named(env, HYRULE_CASTLE_GROUNDS_STATE)
    settle_control(env)


def rebuild(slug: str) -> int:
    from retro_harness.env import read_state_bytes, save_state, state_path
    from retro_harness.segment_runner import configure_headless

    from alttp.scripts.audit_pins import ROOM_TO_DUNGEON
    from alttp.spine.tape import provenance_path
    from alttp.startup import build_boot_env

    spec = ENTRIES[slug]
    pokes = kit_pokes(slug)
    pin_path = state_path(GAME_DIR, GAME, spec.pin)
    creating = not pin_path.is_file()
    if creating and spec.room is None:
        raise SystemExit(f"{spec.pin} does not exist; no quoted room to create")
    configure_headless()
    env = build_boot_env(HYRULE_CASTLE_GROUNDS_STATE, render_mode=None)
    try:
        want_dungeon: int | None = None
        if creating:
            today = int(spec.room) & 0xFF
            want_dungeon = ROOM_TO_DUNGEON.get(today)
            if want_dungeon is None or want_dungeon == _CASTLE_DUNGEON:
                raise SystemExit(
                    f"{spec.pin} room 0x{today:02X} is not a non-castle palace room"
                )
        else:
            today = _today_room(env, spec.pin)
        landed: tuple[int, int, bool, int] | None = None
        used_module = 0
        frames = 0
        for module in _MODULES:
            _load_grounds(env)
            apply_pokes(env, pokes)
            trigger_entrance(env, spec.entrance, module=module)
            limit = 900 if module == _MODULES[0] else 1800
            ok, frames = wait_indoor_control(env, max_frames=limit)
            room, dungeon, indoors, mode = _snap_room(env)
            print(
                f"load module=0x{module:02X} ok={int(ok)} frames={frames} "
                f"room=0x{room:02X} dungeon=0x{dungeon:02X} "
                f"indoors={int(indoors)} mode=0x{mode:02X}"
            )
            palace_ok = want_dungeon is None or dungeon == want_dungeon
            if ok and indoors and room == today and palace_ok:
                landed = (room, dungeon, indoors, mode)
                used_module = module
                break
        if landed is None:
            raise SystemExit(
                f"{spec.pin} did not reload room 0x{today:02X}; pin not written"
            )
        path = save_state(env, GAME_DIR, GAME, spec.pin)
        write_pin_provenance(
            path,
            source_run=f"poke:{slug}",
            frame=frames,
            kind="poke",
        )
        env.em.set_state(read_state_bytes(path))  # type: ignore[attr-defined]
        room, dungeon, indoors, mode = _snap_room(env)
        bad_create = creating and (
            dungeon != want_dungeon or mode != 0x07 or dungeon == _CASTLE_DUNGEON
        )
        if room != today or not indoors or bad_create:
            if creating:
                path.unlink(missing_ok=True)
                provenance_path(path).unlink(missing_ok=True)
            raise SystemExit(
                f"{spec.pin} read back room 0x{room:02X} dungeon=0x{dungeon:02X} "
                f"mode=0x{mode:02X}; pin not kept"
            )
        kind = (read_pin_provenance(path) or {}).get("kind")
        if kind != "poke":
            raise SystemExit(f"{spec.pin} provenance kind {kind!r}, wanted poke")
        names = " ".join(f"{poke.name}={poke.value}" for poke in pokes)
        print(
            f"{spec.pin} room=0x{room:02X} dungeon=0x{dungeon:02X} "
            f"kind={kind} entrance=0x{spec.entrance:02X} "
            f"module=0x{used_module:02X} {names}"
        )
    finally:
        env.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slug", help="dungeon slug (eastern, ice, skull)")
    parser.add_argument(
        "--print-kit",
        action="store_true",
        help="print the requirement kit and exit; no emulator",
    )
    args = parser.parse_args(argv)
    if args.slug not in ENTRIES:
        known = ", ".join(sorted(ENTRIES))
        raise SystemExit(f"unknown slug {args.slug!r}; known: {known}")
    pokes = kit_pokes(args.slug)
    if args.print_kit:
        spec = ENTRIES[args.slug]
        print(
            f"{args.slug} pin={spec.pin} entrance=0x{spec.entrance:02X} "
            + " ".join(f"{poke.name}=${poke.offset:04X}:{poke.value}" for poke in pokes)
        )
        return 0
    return rebuild(args.slug)


if __name__ == "__main__":
    sys.exit(main())
