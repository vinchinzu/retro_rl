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
    elif location == SceneLocation.FESTIVAL and facts.player_state not in {0, 1}:
        # The festival "location" comes from the tomorrow-weather byte, so it
        # is true the whole eve of a festival. Stale dialogue registers
        # (text_id left over from an earlier chat) must not turn a free
        # farmer (input_lock == 1) into a cutscene: run24 D22 mashed A for
        # 241f x 12 in town and lost the day and the run.
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


class SceneGuard:
    """Advance a dialogue/menu/event that no active task is handling.

    A scene is "unowned" when it is dismissible (dialogue, menu, input lock,
    cutscene) and its whole signature (text id/mode, cursor, tilemap, pose,
    lock) has not changed for ``quiet_frames``: whatever task is running is
    either not looking at it or pressing buttons that do nothing. The guard
    then answers with pulsed A (B first for a menu), for at most ``bound``
    frames, and records what appeared so the run teaches us the scene id.
    """

    def __init__(self, quiet_frames: int = 300, bound: int = 480) -> None:
        self.quiet_frames = quiet_frames
        self.bound = bound
        self.log: list[dict[str, object]] = []
        self._sig: tuple | None = None
        self._quiet = 0
        self._active = 0
        self._gave_up = False
        self._name_press_step = -(10**9)

    @property
    def active(self) -> bool:
        return self._active > 0

    def observe(
        self, ram: np.ndarray, step: int = 0, child_pressed: bool = False
    ) -> tuple[str, ...] | None:
        """Buttons to pulse this frame, or None when the scene is not ours.

        ``child_pressed``: the active task pressed a button last frame. A task
        that is pressing owns the scene (berry pick, shop, tape), so the quiet
        window only counts frames in which it pressed nothing.
        """
        scene = classify_scene_from_ram(ram)
        if not scene.needs_input_dismiss or scene_indicates_ending(scene):
            self._sig, self._quiet, self._active, self._gave_up = None, 0, 0, False
            return None
        sig = (
            scene.mode,
            scene.dialog_text_id,
            scene.dialog_text_mode,
            scene.dialog_menu_cursor,
            scene.tilemap,
            scene.player_x,
            scene.player_y,
            scene.input_lock,
        )
        if sig != self._sig and not self._active:
            self._sig, self._quiet, self._gave_up = sig, 0, False
            return None
        if self._gave_up:
            return None
        if child_pressed and not self._active:
            self._quiet = 0
            return None
        self._quiet += 1
        if not self._active and self._quiet < self.quiet_frames:
            return None
        self._active += 1
        if self._active == 1:
            row = {
                "step": int(step),
                "mode": scene.mode.value,
                "text_id": scene.dialog_text_id,
                "text_id_hex": f"0x{scene.dialog_text_id:04X}",
                "text_mode": scene.dialog_text_mode,
                "cursor": scene.dialog_menu_cursor,
                "tilemap": scene.tilemap,
                "location": scene.location.value,
                "input_lock": scene.input_lock,
            }
            if len(self.log) < 200:
                self.log.append(row)
            print(f"[SCENE_GUARD] unowned {scene.summary()} {scene.reason}".rstrip())
        if self._active > self.bound:
            self._gave_up = True
            self._active = 0
            print("[SCENE_GUARD] gave up; scene did not clear")
            return None
        if scene.mode == SceneMode.MENU and self._active <= self.bound // 3:
            return ("b",)
        # A only: B on a plain dialogue can answer "No" to a prompt.
        return ("a",)


def _dog_a_blocked(ram: np.ndarray) -> bool:
    """A beside the dog starts a hug, unless this A is closing a text box.

    ``input_lock != 1`` or a scene that needs dismiss keeps A: stripping the
    press that closes the 5pm box deadlocks ShippingScene.
    """
    from harvest.core.ram_catalog import read_ram_value
    from harvest.planner.tasks.home_recover import dog_adjacent

    if int(read_ram_value(ram, "input_lock")) != 1:
        return False
    if classify_scene_from_ram(ram).needs_input_dismiss:
        return False
    return dog_adjacent(ram)


def _buttons_respecting_dog(
    ram: np.ndarray, buttons: tuple[str, ...]
) -> tuple[str, ...] | None:
    if "a" not in buttons or not _dog_a_blocked(ram):
        return buttons
    kept = tuple(button for button in buttons if button != "a")
    if "b" not in kept:
        kept = (*kept, "b")
    return kept or None


# Name grid (player, dog, horse, calf, kids). Title is 0x5C and the diary
# is 0x5E; those stay on PowerOnStartTask. B deletes a character (NameMenuKeyB).
_NAME_ENTRY_TILEMAP = 0x5F
_NAME_CURSOR_ADDR = 0x0991
_NAME_LENGTH_ADDR = 0x0994
_NAME_ENTRY_CHARS = 4
# Same gap the outdoor dog intro uses. Callers pulse every 2 frames, and an
# odd step is dropped, so a press is committed only on an even step.
_NAME_ENTRY_GAP = 40


def _name_u8(ram: np.ndarray, address: int) -> int:
    if address < 0 or address >= len(ram):
        return 0
    return int(ram[address])


def name_entry_press(
    ram: np.ndarray,
    *,
    step: int,
    last_press_step: int,
) -> tuple[tuple[str, ...], str, int] | None:
    """Next name-grid edge, or None when the live tilemap is not 0x5F.

    Empty buttons are an idle frame. The caller must not start MultiMapNav
    or a rescue walk while this returns: the farmer is at the name screen,
    often ``(0, 0)``, and is not on a map. Never includes B.
    """
    if int(read_ram_value(ram, "tilemap")) != _NAME_ENTRY_TILEMAP:
        return None
    if step % 2 != 0 or step - last_press_step < _NAME_ENTRY_GAP:
        return (), "name entry wait", last_press_step
    if int(read_ram_value(ram, "input_lock")) != 5:
        return (), "name entry wait", last_press_step
    length = _name_u8(ram, _NAME_LENGTH_ADDR)
    cursor = _name_u8(ram, _NAME_CURSOR_ADDR)
    # Same reversed grid as PowerOnStartTask: physical LEFT, then UP, then A.
    if length < _NAME_ENTRY_CHARS:
        reason = f"name entry char {length + 1}/{_NAME_ENTRY_CHARS}"
        buttons: tuple[str, ...] = ("a",)
    elif cursor == 0:
        reason = "name entry move to OK"
        buttons = ("left",)
    elif cursor == 40:
        reason = "name entry move to OK"
        buttons = ("up",)
    elif cursor == 70:
        reason = "name entry confirm"
        buttons = ("a",)
    else:
        reason = f"name entry confirm cursor={cursor}"
        buttons = ("a",)
    return buttons, reason, step


def unowned_scene_press(
    ram: np.ndarray,
    guard: SceneGuard,
    step: int = 0,
    child_pressed: bool = False,
) -> tuple[tuple[str, ...], str] | None:
    """Buttons to pulse because nobody owns this scene, and a log reason.

    ShippingScene is an A edge, not a held button, and does not wait out the
    quiet window. Tilemap 0x5F is the name grid and is taken before the guard
    so a child nav or rescue cannot walk from (0, 0). Otherwise ``guard``
    counts unowned frames. ``child_pressed`` means the active task owns the
    scene (the Spring berry pick).
    """
    from harvest.core.shipping_credit import shipping_scene_needs_dismiss

    named = name_entry_press(
        ram,
        step=step,
        last_press_step=getattr(guard, "_name_press_step", -(10**9)),
    )
    if named is not None:
        buttons, reason, last = named
        guard._name_press_step = last
        return buttons, reason

    if shipping_scene_needs_dismiss(ram):
        buttons: tuple[str, ...] | None = ("a",)
        reason = "shipping scene"
    else:
        buttons = guard.observe(ram, step, child_pressed=child_pressed)
        reason = "scene guard"
    if not buttons:
        return None
    safe = _buttons_respecting_dog(ram, buttons)
    if not safe:
        return None
    return safe, reason


__all__ = [
    "Scene",
    "SceneGuard",
    "SceneLocation",
    "SceneMode",
    "classify_scene",
    "classify_scene_from_ram",
    "morning_scene_ready",
    "name_entry_press",
    "scene_indicates_ending",
    "unowned_scene_press",
]
