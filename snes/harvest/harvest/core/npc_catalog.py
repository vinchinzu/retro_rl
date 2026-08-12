#!/usr/bin/env python3
"""NPC, dialogue, and status-flag facts backed by RAM and HM-Decomp.

This module deliberately keeps uncertain reverse-engineering data explicit:
dynamic positions come from the live game-object table, while names/dialogue
come from decoded ROM text labels.  Unknown game objects are still exported
with sprite IDs so recordings can promote them to named NPCs later.
"""

from __future__ import annotations

import argparse
import shutil
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np

from harvest.paths import DECOMP_DIR
from harvest.core.ram_catalog import read_ram_u16, read_ram_u8, read_ram_value, write_mutable_field
from harvest.runtime.rom_tools import MutableSaveState, parse_save_state, resolve_state_path
from harvest.core.tile_catalog import TILE_SIZE


TEXT_POINTER_TABLE_PATH = DECOMP_DIR / "src" / "code_banks" / "bank_83.asm"
UNLINKED_TEXT_PATH = DECOMP_DIR / "UnlinkedText.txt"

GOBJ_STRUCT_BASE = 0x019C
GOBJ_STRUCT_STRIDE = 0x24
GOBJ_STRUCT_COUNT = 40
GOBJ_INITIALIZED = 0x7777
ADDR_GOBJ_LOADED_OBJS = 0x00DC
ADDR_PLAYER_GOBJ_INDEX = 0x0905

PLAYER_SPRITE_TABLE_IDS = {0x0005, 0x0006, 0x0007, 0x000E}
KNOWN_ENTITY_SPRITES = {
    0x01D9: ("dog", "animal", "recording"),
    0x01E1: ("chicken", "animal", "recording"),
    **{sprite: ("cow", "animal", "recording") for sprite in range(0x018B, 0x01D9)},
    **{sprite: ("chicken", "animal", "recording") for sprite in range(0x0212, 0x021A)},
}

ROMANCE_NPCS = ("maria", "ann", "nina", "ellen", "eve")
ROMANCE_HEART_FIELDS = {
    "maria": "maria_hearts",
    "ann": "ann_hearts",
    "nina": "nina_hearts",
    "ellen": "ellen_hearts",
    "eve": "eve_hearts",
}
ROMANCE_MARRIAGE_BITS = {
    "maria": 0x0001,
    "ann": 0x0002,
    "nina": 0x0004,
    "ellen": 0x0008,
    "eve": 0x0010,
}
ROMANCE_200_HEART_EVENT_BITS = {
    "maria": ("romance_event_flags", 0x1000),
    "ann": ("romance_event_flags", 0x4000),
    "nina": ("family_event_flags", 0x0001),
    "ellen": ("family_event_flags", 0x0004),
    "eve": ("family_event_flags", 0x0010),
}

# CODE_81D1C5 compares hearts against the second value in each 4-byte pair.
ROMANCE_HEART_THRESHOLDS = (49, 119, 199, 249, 299, 399, 499, 599, 799, 999)
ROMANCE_HEART_TEXT_BASE = 0x03A6


@dataclass(frozen=True)
class GameObjectSnapshot:
    slot: int
    struct_offset: int
    sprite_table_idx: int
    flip: int
    pixel: Tuple[int, int]
    tile: Tuple[int, int]
    unk1: int
    metadata_pointer: int
    sprite_table_address: int
    component_total: int
    components: Tuple[int, ...]
    kind: str
    label: str
    source: str
    is_player: bool
    is_npc_candidate: bool

    def to_dict(self) -> dict:
        return {
            "slot": self.slot,
            "struct_offset": f"0x{self.struct_offset:04X}",
            "sprite_table_idx": self.sprite_table_idx,
            "sprite_table_hex": f"0x{self.sprite_table_idx:04X}",
            "flip": self.flip,
            "pixel": list(self.pixel),
            "tile": list(self.tile),
            "unk1": self.unk1,
            "metadata_pointer": f"0x{self.metadata_pointer:04X}",
            "sprite_table_address": f"0x{self.sprite_table_address:04X}",
            "component_total": self.component_total,
            "components": [f"0x{component:02X}" for component in self.components],
            "kind": self.kind,
            "label": self.label,
            "source": self.source,
            "is_player": self.is_player,
            "is_npc_candidate": self.is_npc_candidate,
        }


@dataclass(frozen=True)
class TextRecord:
    text_id: int
    pointer_label: str
    address: str
    text_label: str
    text: str
    npc: str | None
    category: str
    choices: Tuple[str, ...]

    def to_dict(self, *, compact: bool = False) -> dict:
        data = {
            "id": self.text_id,
            "id_hex": f"0x{self.text_id:04X}",
            "pointer_label": self.pointer_label,
            "address": self.address,
            "text_label": self.text_label,
            "npc": self.npc,
            "category": self.category,
            "choices": list(self.choices),
        }
        if not compact:
            data["text"] = self.text
        return data


@dataclass(frozen=True)
class StatusFlagBank:
    key: str
    address: int
    label: str
    bit_labels: dict[int, str]


STATUS_FLAG_BANKS: Tuple[StatusFlagBank, ...] = (
    StatusFlagBank("event_flags_1f5a", 0x11F5A, "Event Flags 1F5A", {0x0002: "event_gate_1f5a_0002"}),
    StatusFlagBank("event_flags_1f5c", 0x11F5C, "Event Flags 1F5C", {0x0020: "player_input_event_lock"}),
    StatusFlagBank("event_flags_1f5e", 0x11F5E, "Event Flags 1F5E", {}),
    StatusFlagBank("event_flags_1f60", 0x11F60, "Event Flags 1F60", {0x0800: "save_slot_failure_seen"}),
    StatusFlagBank("event_flags_1f62", 0x11F62, "Event Flags 1F62", {}),
    StatusFlagBank(
        "upgrade_flags",
        0x11F64,
        "Upgrade Flags",
        {
            0x0040: "house_upgraded_once",
            0x0080: "house_level_2",
        },
    ),
    StatusFlagBank(
        "marriage_flags",
        0x11F66,
        "Marriage Flags",
        {
            0x0001: "married_maria",
            0x0002: "married_ann",
            0x0004: "married_nina",
            0x0008: "married_ellen",
            0x0010: "married_eve",
            0x0080: "maria_event_blocker",
        },
    ),
    StatusFlagBank("event_flags_1f68", 0x11F68, "Event Flags 1F68", {}),
    StatusFlagBank(
        "romance_event_flags",
        0x11F6A,
        "Romance Event Flags",
        {
            0x1000: "maria_200_heart_event_seen",
            0x2000: "maria_event_blocker",
            0x4000: "ann_200_heart_event_seen",
            0x8000: "ann_event_blocker",
        },
    ),
    StatusFlagBank(
        "family_event_flags",
        0x11F6C,
        "Family/Event Flags",
        {
            0x0001: "nina_200_heart_event_seen",
            0x0002: "nina_event_blocker",
            0x0004: "ellen_200_heart_event_seen",
            0x0008: "ellen_event_blocker",
            0x0010: "eve_200_heart_event_seen",
            0x0020: "eve_event_blocker",
            0x0080: "wife_about_to_give_birth",
        },
    ),
    StatusFlagBank(
        "incubator_flags",
        0x11F6E,
        "Incubator/Family/Animal Flags",
        {
            0x0004: "first_child_born_or_expected",
            0x0008: "second_child_born_or_expected",
            0x0040: "cow_funeral",
            0x1000: "cow_born",
            0x2000: "egg_incubating",
        },
    ),
)


def _entity_name(sprite_table_idx: int, *, is_player: bool) -> tuple[str, str, str]:
    if is_player or sprite_table_idx in PLAYER_SPRITE_TABLE_IDS:
        return "player", "player", "decomp"
    if sprite_table_idx in KNOWN_ENTITY_SPRITES:
        return KNOWN_ENTITY_SPRITES[sprite_table_idx]
    if 0x0200 <= sprite_table_idx <= 0x02FF:
        return f"candidate_npc_{sprite_table_idx:04X}", "npc_candidate", "heuristic"
    return f"game_object_{sprite_table_idx:04X}", "game_object", "raw"


def _low_u8(ram: np.ndarray, address: int) -> int:
    return read_ram_u8(ram, address, live_offset=False)


def _low_u16(ram: np.ndarray, address: int) -> int:
    return read_ram_u16(ram, address, live_offset=False)


def iter_game_objects(ram: np.ndarray) -> Iterable[GameObjectSnapshot]:
    player_slot = _low_u16(ram, ADDR_PLAYER_GOBJ_INDEX)
    for slot in range(GOBJ_STRUCT_COUNT):
        offset = GOBJ_STRUCT_BASE + slot * GOBJ_STRUCT_STRIDE
        if _low_u16(ram, offset) != GOBJ_INITIALIZED:
            continue
        sprite_table_idx = _low_u16(ram, offset + 0x02)
        is_player = slot == player_slot or sprite_table_idx in PLAYER_SPRITE_TABLE_IDS
        label, kind, source = _entity_name(sprite_table_idx, is_player=is_player)
        x = _low_u16(ram, offset + 0x08)
        y = _low_u16(ram, offset + 0x0A)
        component_total = _low_u8(ram, offset + 0x13)
        components = tuple(
            _low_u8(ram, offset + 0x14 + i)
            for i in range(min(component_total, GOBJ_STRUCT_STRIDE - 0x14))
        )
        yield GameObjectSnapshot(
            slot=slot,
            struct_offset=offset,
            sprite_table_idx=sprite_table_idx,
            flip=_low_u16(ram, offset + 0x04),
            pixel=(x, y),
            tile=(x // TILE_SIZE, y // TILE_SIZE),
            unk1=_low_u16(ram, offset + 0x06),
            metadata_pointer=_low_u16(ram, offset + 0x0C),
            sprite_table_address=_low_u16(ram, offset + 0x10),
            component_total=component_total,
            components=components,
            kind=kind,
            label=label,
            source=source,
            is_player=is_player,
            is_npc_candidate=kind == "npc_candidate",
        )


def game_objects(ram: np.ndarray) -> list[GameObjectSnapshot]:
    return list(iter_game_objects(ram))


def nearest_game_objects(ram: np.ndarray, *, limit: int = 8) -> list[dict]:
    player = next((obj for obj in iter_game_objects(ram) if obj.is_player), None)
    if player is None:
        return []
    px, py = player.tile
    rows = []
    for obj in iter_game_objects(ram):
        if obj.is_player:
            continue
        data = obj.to_dict()
        data["distance_tiles"] = abs(obj.tile[0] - px) + abs(obj.tile[1] - py)
        rows.append(data)
    rows.sort(key=lambda row: (row["distance_tiles"], row["slot"]))
    return rows[:limit]


def heart_tier(hearts: int) -> int:
    for tier, threshold in enumerate(ROMANCE_HEART_THRESHOLDS):
        if hearts <= threshold:
            return tier
    return len(ROMANCE_HEART_THRESHOLDS) - 1


def romance_points_for_hearts(hearts: int) -> int:
    tier = max(1, min(10, int(hearts))) - 1
    return ROMANCE_HEART_THRESHOLDS[tier]


def heart_tier_range(tier: int) -> tuple[int, int]:
    low = 0 if tier == 0 else ROMANCE_HEART_THRESHOLDS[tier - 1] + 1
    high = ROMANCE_HEART_THRESHOLDS[tier]
    return low, high


def romance_field_for_npc(npc: str) -> str:
    key = npc.strip().lower()
    if key not in ROMANCE_HEART_FIELDS:
        known = ", ".join(ROMANCE_NPCS)
        raise ValueError(f"Unknown romance NPC {npc!r}; expected one of: {known}")
    return ROMANCE_HEART_FIELDS[key]


def set_romance_points(target: MutableSaveState, npc: str, points: int) -> int:
    value = max(0, min(999, int(points)))
    write_mutable_field(target, romance_field_for_npc(npc), value, raw=True)
    return value


def set_romance_hearts(target: MutableSaveState, npc: str, hearts: int) -> int:
    return set_romance_points(target, npc, romance_points_for_hearts(hearts))


def parse_romance_assignment(raw: str) -> tuple[str, int]:
    if "=" not in raw:
        raise ValueError(f"Expected NPC=HEARTS assignment, got {raw!r}")
    npc, hearts_text = raw.split("=", 1)
    npc = npc.strip().lower()
    if not npc:
        raise ValueError(f"Missing NPC name in assignment {raw!r}")
    return npc, int(hearts_text.strip())


def status_flags(ram: np.ndarray) -> dict[str, dict]:
    decoded: dict[str, dict] = {}
    for bank in STATUS_FLAG_BANKS:
        value = read_ram_value(ram, bank.key, raw=True)
        named = {
            name: bool(value & mask)
            for mask, name in sorted(bank.bit_labels.items())
        }
        unknown_mask = value & ~sum(bank.bit_labels.keys())
        decoded[bank.key] = {
            "address": f"0x{bank.address:05X}",
            "value": value,
            "hex": f"0x{value:04X}",
            "label": bank.label,
            "named_bits": named,
            "unknown_mask": f"0x{unknown_mask:04X}",
        }
    return decoded


def relationship_status(ram: np.ndarray) -> dict[str, dict]:
    flags = {key: row["value"] for key, row in status_flags(ram).items()}
    rows: dict[str, dict] = {}
    for npc in ROMANCE_NPCS:
        hearts = read_ram_value(ram, ROMANCE_HEART_FIELDS[npc], raw=True)
        tier = heart_tier(hearts)
        low, high = heart_tier_range(tier)
        event_field, event_mask = ROMANCE_200_HEART_EVENT_BITS[npc]
        rows[npc] = {
            "hearts": hearts,
            "heart_tier": tier,
            "heart_tier_range": [low, high],
            "heart_meter_text_id": ROMANCE_HEART_TEXT_BASE + tier,
            "married": bool(flags.get("marriage_flags", 0) & ROMANCE_MARRIAGE_BITS[npc]),
            "heart_200_event_seen": bool(flags.get(event_field, 0) & event_mask),
        }
    return rows


def current_dialogue_registers(ram: np.ndarray) -> dict:
    text_id = read_ram_value(ram, "dialog_text_id", raw=True)
    return {
        "input_lock": read_ram_value(ram, "input_lock", raw=True),
        "text_id": text_id,
        "text_id_hex": f"0x{text_id:04X}",
        "menu_cursor": read_ram_value(ram, "dialog_menu_cursor", raw=True),
        "choice_result": read_ram_value(ram, "dialog_choice_result", raw=True),
        "text_mode": read_ram_value(ram, "dialog_text_mode", raw=True),
    }


def romance_dialogue_tree(ram: np.ndarray | None = None) -> dict:
    tiers = []
    for tier, threshold in enumerate(ROMANCE_HEART_THRESHOLDS):
        low, high = heart_tier_range(tier)
        tiers.append(
            {
                "tier": tier,
                "heart_range": [low, high],
                "heart_meter_text_id": ROMANCE_HEART_TEXT_BASE + tier,
            }
        )
    data: dict[str, object] = {
        "source": "HM-Decomp/src/code_banks/bank_81.asm CODE_81D1C5 and bachelorette talk handlers",
        "heart_tiers": tiers,
        "notes": [
            "The bachelorette handlers read the active girl's heart field, map it through CODE_81D1C5, then start text 0x03A6+tier.",
            "Full branch ownership still needs more event-script labeling; extracted text groups are ROM labels, not final schedule-aware talk routes.",
        ],
    }
    if ram is not None:
        data["relationships"] = relationship_status(ram)
    return data


_POINTER_RE = re.compile(r"^\s*dl\s+([A-Za-z0-9_]+)\s*;([0-9A-Fa-f]{6});([0-9A-Fa-f]+)")
_TEXT_HEADER_RE = re.compile(r"^(DATA16_([0-9A-Fa-f]{6})):(?:([A-Za-z0-9_]+))?\s*$")
_LABEL_HEADER_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s*$")


@lru_cache(maxsize=1)
def text_pointer_table() -> dict[int, tuple[str, str]]:
    rows: dict[int, tuple[str, str]] = {}
    in_table = False
    with TEXT_POINTER_TABLE_PATH.open(encoding="utf-8") as f:
        for line in f:
            if "Text_Pointer_Table:" in line:
                in_table = True
                continue
            if not in_table:
                continue
            match = _POINTER_RE.match(line)
            if not match:
                if rows and line.strip() and not line.lstrip().startswith(";"):
                    break
                continue
            label, address, text_id = match.groups()
            rows[int(text_id, 16)] = (label, address.upper())
    return rows


@lru_cache(maxsize=1)
def decoded_text_by_address() -> dict[str, tuple[str, str]]:
    rows: dict[str, tuple[str, str]] = {}
    current_address: str | None = None
    current_label = ""
    current_lines: list[str] = []

    def flush() -> None:
        if current_address is None:
            return
        text = "\n".join(current_lines).strip()
        rows[current_address] = (current_label, text)

    with UNLINKED_TEXT_PATH.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            match = _TEXT_HEADER_RE.match(line)
            if match:
                flush()
                _, address, label = match.groups()
                current_address = address.upper()
                current_label = label or ""
                current_lines = []
                continue
            label_match = _LABEL_HEADER_RE.match(line)
            if label_match:
                flush()
                current_address = label_match.group(1)
                current_label = label_match.group(1)
                current_lines = []
                continue
            if current_address is not None:
                current_lines.append(line)
    flush()
    return rows


def _clean_text(text: str) -> str:
    return text.replace("¬", "").strip()


def _choices_from_text(text: str) -> tuple[str, ...]:
    if "▼" not in text and "»" not in text:
        return ()
    choices = []
    after_prompt = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "▼" in stripped:
            after_prompt = True
            continue
        if not after_prompt and "»" not in stripped:
            continue
        if "»" in stripped:
            stripped = stripped.split("»", 1)[0].strip()
        if stripped in {"Yes", "No"} or stripped.startswith(("Yes ", "No ", "I ", "Not ")):
            choices.append(stripped)
    return tuple(choices)


def _infer_npc(pointer_label: str, text_label: str, text: str) -> str | None:
    haystack = f"{pointer_label} {text_label}".lower()
    for npc in ROMANCE_NPCS:
        if npc in haystack:
            return npc
    if "bartender" in haystack:
        return "bartender"
    if "peddler" in haystack or "hawker" in haystack:
        return "peddler"
    if "shipper" in haystack:
        return "shipper"
    if "fortune" in haystack:
        return "fortune_teller"
    if "visitor" in haystack:
        return "visitor"
    if "grandparent" in haystack:
        return "eves_grandparent"
    return None


def _infer_category(pointer_label: str, text_label: str, text: str) -> str:
    haystack = f"{pointer_label} {text_label}".lower()
    if "question" in haystack or "▼" in text or "»" in text:
        return "choice"
    if "festival" in haystack:
        return "festival"
    if "intro" in haystack:
        return "intro"
    if "spring" in haystack or "summer" in haystack or "fall" in haystack or "winter" in haystack:
        return "seasonal"
    if "married" in haystack or "wife" in haystack:
        return "married"
    if "bluefeather" in haystack:
        return "blue_feather"
    return "dialogue"


@lru_cache(maxsize=1)
def text_records() -> tuple[TextRecord, ...]:
    decoded = decoded_text_by_address()
    rows = []
    for text_id, (pointer_label, address) in sorted(text_pointer_table().items()):
        text_label, text = decoded.get(address, ("", ""))
        clean = _clean_text(text)
        npc = _infer_npc(pointer_label, text_label, clean)
        rows.append(
            TextRecord(
                text_id=text_id,
                pointer_label=pointer_label,
                address=address,
                text_label=text_label,
                text=clean,
                npc=npc,
                category=_infer_category(pointer_label, text_label, clean),
                choices=_choices_from_text(clean),
            )
        )
    return tuple(rows)


def text_record_for_id(text_id: int) -> TextRecord | None:
    tid = int(text_id)
    for record in text_records():
        if record.text_id == tid:
            return record
    return None


def search_text_records(query: str, *, limit: int = 12) -> list[TextRecord]:
    needle = query.strip().lower()
    if not needle:
        return []
    hits: list[TextRecord] = []
    for record in text_records():
        hay = f"{record.text} {record.text_label} {record.pointer_label}".lower()
        if needle in hay:
            hits.append(record)
            if len(hits) >= limit:
                break
    return hits


def dialogue_catalog(*, npc: str | None = None, compact: bool = False) -> dict:
    npc_filter = npc.lower() if npc else None
    records = [
        record
        for record in text_records()
        if npc_filter is None or record.npc == npc_filter
    ]
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record.npc or "unknown", []).append(record.to_dict(compact=compact))
    return {
        "source": {
            "text_pointer_table": str(TEXT_POINTER_TABLE_PATH),
            "decoded_text": str(UNLINKED_TEXT_PATH),
        },
        "record_count": len(records),
        "groups": grouped,
        "romance_tree": romance_dialogue_tree(),
    }


def npc_snapshot_dict(
    ram: np.ndarray,
    *,
    include_dialogue_text: bool = False,
    npc: str | None = None,
    compact: bool = False,
) -> dict:
    objects = game_objects(ram)
    data = {
        "game_objects": [obj.to_dict() for obj in objects],
        "candidate_npcs": [obj.to_dict() for obj in objects if obj.is_npc_candidate],
        "nearest_game_objects": nearest_game_objects(ram),
        "relationships": relationship_status(ram),
        "status_flags": status_flags(ram),
        "dialogue_registers": current_dialogue_registers(ram),
        "romance_tree": romance_dialogue_tree(ram),
    }
    if include_dialogue_text:
        data["dialogue_catalog"] = dialogue_catalog(npc=npc, compact=compact)
    return data


def load_state_ram(state_name: str) -> np.ndarray:
    state = parse_save_state(resolve_state_path(state_name))
    return np.frombuffer(state.ram, dtype=np.uint8).copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export NPC/entity/dialogue facts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    entities = subparsers.add_parser("entities", help="Export live game-object/NPC positions from a save state")
    entities.add_argument("--state", default="latest", help="Save state name")
    entities.add_argument("--dialogue-text", action="store_true", help="Include decoded ROM dialogue text")
    entities.add_argument("--npc", help="Filter dialogue text to one NPC/group")
    entities.add_argument("--compact", action="store_true", help="Omit dialogue body text when listing dialogue")
    entities.add_argument("--out", help="Write JSON to this path")

    dialogue = subparsers.add_parser("dialogue", help="Export decoded ROM dialogue groups")
    dialogue.add_argument("--npc", help="Filter to one NPC/group, e.g. maria or eve")
    dialogue.add_argument("--compact", action="store_true", help="Omit dialogue body text")
    dialogue.add_argument("--out", help="Write JSON to this path")

    flags = subparsers.add_parser("flags", help="Export decoded status flags from a save state")
    flags.add_argument("--state", default="latest", help="Save state name")
    flags.add_argument("--out", help="Write JSON to this path")

    set_hearts = subparsers.add_parser("set-hearts", help="Set romance heart tiers in a save state")
    set_hearts.add_argument("assignments", nargs="+", help="One or more NPC=HEARTS pairs, e.g. ann=8 eve=6")
    set_hearts.add_argument("--state", default="latest", help="Save state name")
    set_hearts.add_argument("--no-backup", action="store_true", help="Edit without first copying the state file")

    args = parser.parse_args()
    if args.command == "entities":
        data = npc_snapshot_dict(
            load_state_ram(args.state),
            include_dialogue_text=bool(args.dialogue_text),
            npc=args.npc,
            compact=bool(args.compact),
        )
    elif args.command == "dialogue":
        data = dialogue_catalog(npc=args.npc, compact=bool(args.compact))
    elif args.command == "flags":
        ram = load_state_ram(args.state)
        data = {
            "relationships": relationship_status(ram),
            "status_flags": status_flags(ram),
            "dialogue_registers": current_dialogue_registers(ram),
        }
    elif args.command == "set-hearts":
        state_path = resolve_state_path(args.state)
        backup_path = None
        if not args.no_backup:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = state_path.with_name(f"{state_path.stem}_backup_romance_{timestamp}{state_path.suffix}")
            shutil.copy2(state_path, backup_path)
        state = MutableSaveState.load(state_path)
        updates = {}
        for raw_assignment in args.assignments:
            npc, hearts = parse_romance_assignment(raw_assignment)
            updates[npc] = {
                "hearts": hearts,
                "points": set_romance_hearts(state, npc, hearts),
                "field": romance_field_for_npc(npc),
            }
        state.save(state_path)
        data = {
            "state": str(state_path),
            "backup": str(backup_path) if backup_path is not None else None,
            "updates": updates,
            "relationships": relationship_status(np.frombuffer(bytes(state.ram), dtype=np.uint8).copy()),
        }
    else:
        parser.error(f"unknown command {args.command!r}")

    text = json.dumps(data, indent=2)
    output_path = getattr(args, "out", None)
    if output_path:
        with open(output_path, "w") as f:
            f.write(text + "\n")
        print(f"Wrote NPC catalog to {output_path}")
    else:
        print(text)


if __name__ == "__main__":
    main()
