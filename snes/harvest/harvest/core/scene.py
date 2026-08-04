"""Scene classification for runtime safety decisions.

The scene classifier is intentionally read-only and small.  It turns raw RAM or
an existing ``WorldSnapshot`` into a stable summary that planner/recovery code
can use before running normal navigation or interaction tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from harvest.core.ram_catalog import field_spec, read_ram_value
from harvest.maps.map_config import FARM_TILEMAP_IDS, MAP_REGISTRY, get_map_name


class SceneMode(str, Enum):
    NORMAL = "normal_map"
    DIALOGUE = "dialogue"
    MENU = "menu"
    INPUT_LOCKED = "input_locked"
    MAP_TRANSITION = "map_transition"
    SLEEP_WAKE_TRANSITION = "sleep_wake_transition"
    CUTSCENE_EVENT = "cutscene_event"
    ENDING_CREDITS = "ending_or_credits"
    UNKNOWN_TILEMAP = "unknown_tilemap"
    INVALID_COORDINATES = "invalid_coordinates"


class SceneLocation(str, Enum):
    FARM = "farm"
    HOUSE = "house"
    BARN = "barn"
    COOP = "coop"
    SHED = "shed"
    SHOP = "shop"
    TOWN = "town"
    PATH = "path"
    MOUNTAIN = "mountain"
    CHURCH = "church"
    FESTIVAL = "festival"
    SLEEP_ROOM = "sleep_room"
    UNKNOWN = "unknown"


HOUSE_VARIANTS = {
    0x15: "base",
    0x16: "level1",
    0x17: "level2",
}
SHOP_VARIANTS = {
    0x1C: "seed_shop",
    0x1D: "flower_back",
    0x24: "animal_shop",
}
LOCATION_BY_TILEMAP = {
    0x0C: SceneLocation.PATH,
    0x04: SceneLocation.TOWN,
    0x05: SceneLocation.TOWN,
    0x10: SceneLocation.MOUNTAIN,
    0x1B: SceneLocation.CHURCH,
    0x1C: SceneLocation.SHOP,
    0x1D: SceneLocation.SHOP,
    0x24: SceneLocation.SHOP,
    0x26: SceneLocation.SHED,
    0x27: SceneLocation.BARN,
    0x28: SceneLocation.COOP,
    0x29: SceneLocation.MOUNTAIN,  # MapMountainCave (west hole) — not outdoor spa
}
FESTIVAL_EVENT_CODES = frozenset({6, 7, 8, 9, 10, 11})
SLEEP_TRANSITION_TILEMAP = 0x0F
PLAYER_STATE_TRANSITION_BIT = 0x80
MAX_TILE_COORD = 63


@dataclass(frozen=True)
class Scene:
    mode: SceneMode
    location: SceneLocation
    tilemap: int
    tilemap_name: str
    player_x: int
    player_y: int
    input_lock: int
    reason: str = ""
    variant: str = ""
    event_code: int = 0
    dialog_text_id: int = 0
    dialog_text_mode: int = 0
    dialog_menu_cursor: int = 0
    ending_scene_index: int = 0
    ending_aux_scene_index: int = 0

    @property
    def tile(self) -> tuple[int, int]:
        return self.player_x // 16, self.player_y // 16

    @property
    def is_normal_map(self) -> bool:
        return self.mode == SceneMode.NORMAL

    @property
    def is_recoverable(self) -> bool:
        return self.mode in {
            SceneMode.DIALOGUE,
            SceneMode.MENU,
            SceneMode.INPUT_LOCKED,
            SceneMode.MAP_TRANSITION,
            SceneMode.SLEEP_WAKE_TRANSITION,
            SceneMode.CUTSCENE_EVENT,
        }

    @property
    def is_terminal(self) -> bool:
        return self.mode == SceneMode.ENDING_CREDITS

    @property
    def needs_input_dismiss(self) -> bool:
        """True when mashing A/B is the right recovery action."""
        return self.mode in {
            SceneMode.DIALOGUE,
            SceneMode.MENU,
            SceneMode.INPUT_LOCKED,
            SceneMode.CUTSCENE_EVENT,
        }

    @property
    def is_transient(self) -> bool:
        """True while the game is mid-transition and should not run chores."""
        return self.mode in {
            SceneMode.MAP_TRANSITION,
            SceneMode.SLEEP_WAKE_TRANSITION,
            SceneMode.CUTSCENE_EVENT,
        }

    def summary(self) -> str:
        variant = f"/{self.variant}" if self.variant else ""
        reason = f" {self.reason}" if self.reason else ""
        return (
            f"{self.mode.value}@{self.location.value}{variant} "
            f"tilemap=0x{self.tilemap:02X} pos=({self.player_x},{self.player_y}){reason}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "location": self.location.value,
            "variant": self.variant,
            "tilemap": self.tilemap,
            "tilemap_hex": f"0x{self.tilemap:02X}",
            "tilemap_name": self.tilemap_name,
            "player": {
                "pixel": [self.player_x, self.player_y],
                "tile": list(self.tile),
            },
            "input_lock": self.input_lock,
            "reason": self.reason,
            "event_code": self.event_code,
            "dialogue": {
                "text_id": self.dialog_text_id,
                "text_id_hex": f"0x{self.dialog_text_id:04X}",
                "text_mode": self.dialog_text_mode,
                "menu_cursor": self.dialog_menu_cursor,
            },
            "ending": {
                "scene_index": self.ending_scene_index,
                "aux_scene_index": self.ending_aux_scene_index,
            },
        }


@dataclass(frozen=True)
class _SceneFacts:
    tilemap: int
    player_x: int
    player_y: int
    input_lock: int
    player_state: int
    time_running: int
    hour: int
    event_code: int
    house_size: int
    dialog_text_id: int
    dialog_text_mode: int
    dialog_menu_cursor: int
    ending_scene_index: int
    ending_aux_scene_index: int


def classify_scene(source: Any) -> Scene:
    """Classify a ``WorldSnapshot`` or RAM array into a scene summary."""
    if isinstance(source, np.ndarray):
        return classify_scene_from_ram(source)
    return _classify(_facts_from_snapshot(source))


def classify_scene_from_ram(ram: np.ndarray) -> Scene:
    """Classify a live or save-state RAM buffer without building a full snapshot."""
    return _classify(_facts_from_ram(ram))


def _facts_from_ram(ram: np.ndarray) -> _SceneFacts:
    player_x, player_y = _player_position_from_ram(ram)
    return _SceneFacts(
        tilemap=read_ram_value(ram, "tilemap", raw=True),
        player_x=player_x,
        player_y=player_y,
        input_lock=read_ram_value(ram, "input_lock", raw=True),
        player_state=read_ram_value(ram, "player_state", raw=True),
        time_running=read_ram_value(ram, "time_running", raw=True),
        hour=read_ram_value(ram, "hour", raw=True),
        event_code=read_ram_value(ram, "weather_tomorrow", raw=True),
        house_size=read_ram_value(ram, "house_size", raw=True),
        dialog_text_id=read_ram_value(ram, "dialog_text_id", raw=True),
        dialog_text_mode=read_ram_value(ram, "dialog_text_mode", raw=True),
        dialog_menu_cursor=read_ram_value(ram, "dialog_menu_cursor", raw=True),
        ending_scene_index=read_ram_value(ram, "ending_scene_index", raw=True),
        ending_aux_scene_index=read_ram_value(ram, "ending_aux_scene_index", raw=True),
    )


def _player_position_from_ram(ram: np.ndarray) -> tuple[int, int]:
    player_x = read_ram_value(ram, "player_x", raw=True)
    player_y = read_ram_value(ram, "player_y", raw=True)
    if player_x != 0 or player_y != 0:
        return player_x, player_y

    x_addr = field_spec("player_x").address
    y_addr = field_spec("player_y").address
    if x_addr + 1 >= len(ram) or y_addr + 1 >= len(ram):
        return player_x, player_y
    direct_x = int(ram[x_addr]) | (int(ram[x_addr + 1]) << 8)
    direct_y = int(ram[y_addr]) | (int(ram[y_addr + 1]) << 8)
    if direct_x != 0 or direct_y != 0:
        return direct_x, direct_y
    return player_x, player_y


def _facts_from_snapshot(snapshot: Any) -> _SceneFacts:
    scalars = getattr(snapshot, "scalars", {})
    player = getattr(snapshot, "player")
    dialogue = getattr(snapshot, "dialogue_registers", {}) or {}
    return _SceneFacts(
        tilemap=int(getattr(player, "tilemap", scalars.get("tilemap", 0))),
        player_x=int(getattr(player, "pixel", (0, 0))[0]),
        player_y=int(getattr(player, "pixel", (0, 0))[1]),
        input_lock=int(getattr(player, "input_lock", scalars.get("input_lock", 0))),
        player_state=int(scalars.get("player_state", 0)),
        time_running=int(scalars.get("time_running", 0)),
        hour=int(scalars.get("hour", 0)),
        event_code=int(scalars.get("weather_tomorrow", 0)),
        house_size=int(scalars.get("house_size", 0)),
        dialog_text_id=int(dialogue.get("text_id", scalars.get("dialog_text_id", 0))),
        dialog_text_mode=int(dialogue.get("text_mode", scalars.get("dialog_text_mode", 0))),
        dialog_menu_cursor=int(dialogue.get("menu_cursor", scalars.get("dialog_menu_cursor", 0))),
        ending_scene_index=int(scalars.get("ending_scene_index", 0)),
        ending_aux_scene_index=int(scalars.get("ending_aux_scene_index", 0)),
    )


def _classify(facts: _SceneFacts) -> Scene:
    location, variant = _location_for(facts)
    mode = SceneMode.NORMAL
    reason = ""

    if facts.ending_scene_index or facts.ending_aux_scene_index:
        mode = SceneMode.ENDING_CREDITS
        reason = (
            f"ending_index=0x{facts.ending_scene_index:02X} "
            f"aux=0x{facts.ending_aux_scene_index:02X}"
        )
    elif facts.tilemap == SLEEP_TRANSITION_TILEMAP or facts.time_running == 2:
        mode = SceneMode.SLEEP_WAKE_TRANSITION
        location = SceneLocation.SLEEP_ROOM
        reason = "sleep/wake transition"
    elif (
        location == SceneLocation.HOUSE
        and facts.hour < 12
        and facts.player_y < 100
        and facts.input_lock == 1
    ):
        mode = SceneMode.SLEEP_WAKE_TRANSITION
        reason = "morning wake coordinates not settled"
    elif facts.player_state & PLAYER_STATE_TRANSITION_BIT:
        mode = SceneMode.MAP_TRANSITION
        reason = f"player_state=0x{facts.player_state:02X}"
    elif _invalid_coordinates(facts):
        mode = SceneMode.INVALID_COORDINATES
        reason = "player coordinates outside loaded map"
    elif _looks_like_menu(facts) and (facts.input_lock != 1 or not _known_tilemap(facts.tilemap)):
        mode = SceneMode.MENU
        reason = _dialogue_reason(facts)
    elif _looks_like_dialogue(facts) and (facts.input_lock != 1 or not _known_tilemap(facts.tilemap)):
        mode = SceneMode.DIALOGUE
        reason = _dialogue_reason(facts)
    elif facts.input_lock != 1:
        mode = SceneMode.INPUT_LOCKED
        reason = f"input_lock={facts.input_lock}"
    elif location == SceneLocation.FESTIVAL and (
        _looks_like_dialogue(facts) or facts.player_state not in {0, 1}
    ):
        mode = SceneMode.CUTSCENE_EVENT
        reason = (
            f"festival event_code={facts.event_code} "
            f"player_state=0x{facts.player_state:02X}"
        )
    elif not _known_tilemap(facts.tilemap):
        # Unregistered maps during story/event sequences are usually
        # mash-through cutscenes; treat hard unknowns as cutscenes first so
        # recovery can attempt dismiss before aborting.
        mode = SceneMode.CUTSCENE_EVENT
        reason = f"unregistered tilemap 0x{facts.tilemap:02X}"

    return Scene(
        mode=mode,
        location=location,
        tilemap=facts.tilemap,
        tilemap_name=get_map_name(facts.tilemap),
        player_x=facts.player_x,
        player_y=facts.player_y,
        input_lock=facts.input_lock,
        reason=reason,
        variant=variant,
        event_code=facts.event_code,
        dialog_text_id=facts.dialog_text_id,
        dialog_text_mode=facts.dialog_text_mode,
        dialog_menu_cursor=facts.dialog_menu_cursor,
        ending_scene_index=facts.ending_scene_index,
        ending_aux_scene_index=facts.ending_aux_scene_index,
    )


def _known_tilemap(tilemap: int) -> bool:
    return tilemap in FARM_TILEMAP_IDS or tilemap in MAP_REGISTRY or tilemap == SLEEP_TRANSITION_TILEMAP


def _location_for(facts: _SceneFacts) -> tuple[SceneLocation, str]:
    tilemap = facts.tilemap
    if tilemap in FARM_TILEMAP_IDS:
        return SceneLocation.FARM, get_map_name(tilemap)
    if tilemap in HOUSE_VARIANTS:
        variant = HOUSE_VARIANTS[tilemap]
        if facts.house_size:
            variant = f"{variant}/size{facts.house_size}"
        return SceneLocation.HOUSE, variant
    if (
        facts.event_code in FESTIVAL_EVENT_CODES
        and tilemap in {0x04, 0x0C, 0x10, 0x1B}
    ):
        return SceneLocation.FESTIVAL, f"event_code_{facts.event_code}"
    if tilemap in SHOP_VARIANTS:
        return SceneLocation.SHOP, SHOP_VARIANTS[tilemap]
    return LOCATION_BY_TILEMAP.get(tilemap, SceneLocation.UNKNOWN), ""


def _invalid_coordinates(facts: _SceneFacts) -> bool:
    if facts.player_x == 0 and facts.player_y == 0:
        return True
    if facts.player_x < 0 or facts.player_y < 0:
        return True
    return facts.player_x // 16 > MAX_TILE_COORD or facts.player_y // 16 > MAX_TILE_COORD


def _looks_like_menu(facts: _SceneFacts) -> bool:
    return facts.dialog_menu_cursor not in {0, 0xFF}


def _looks_like_dialogue(facts: _SceneFacts) -> bool:
    return facts.dialog_text_id != 0 or facts.dialog_text_mode != 0


def _dialogue_reason(facts: _SceneFacts) -> str:
    return (
        f"input_lock={facts.input_lock} text=0x{facts.dialog_text_id:04X} "
        f"mode=0x{facts.dialog_text_mode:02X} cursor=0x{facts.dialog_menu_cursor:02X}"
    )


def scene_indicates_ending(scene: Scene) -> bool:
    """True when ending/credits iterators have started."""
    return scene.is_terminal or bool(scene.ending_scene_index or scene.ending_aux_scene_index)


def morning_scene_ready(scene: Scene, hour: int) -> bool:
    """True when morning planning may safely rebuild after sleep/wake."""
    if hour >= 12:
        return False
    if scene_indicates_ending(scene):
        return False
    if scene.mode == SceneMode.NORMAL:
        return True
    # Wake-time house dialogue can linger with free input; allow rebuild.
    return (
        scene.mode in {SceneMode.DIALOGUE, SceneMode.MENU}
        and scene.input_lock == 1
        and scene.location == SceneLocation.HOUSE
    )


__all__ = [
    "Scene",
    "SceneLocation",
    "SceneMode",
    "classify_scene",
    "classify_scene_from_ram",
    "morning_scene_ready",
    "scene_indicates_ending",
]
