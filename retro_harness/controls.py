"""
Shared input handling for SNES games via stable-retro.

Provides consistent keyboard and controller mappings across all games.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import pygame

# SNES Button Map: [B, Y, Select, Start, Up, Down, Left, Right, A, X, L, R]
SNES_B = 0
SNES_Y = 1
SNES_SELECT = 2
SNES_START = 3
SNES_UP = 4
SNES_DOWN = 5
SNES_LEFT = 6
SNES_RIGHT = 7
SNES_A = 8
SNES_X = 9
SNES_L = 10
SNES_R = 11

SNES_BUTTON_NAMES = (
    "B",
    "Y",
    "SELECT",
    "START",
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    "A",
    "X",
    "L",
    "R",
)

# Wire names: D-pad is LEFT/RIGHT; shoulders are L/R. Never use L for walk.
SNES_DPAD_LEFT = "LEFT"
SNES_DPAD_RIGHT = "RIGHT"
SNES_SHOULDER_L = "L"
SNES_SHOULDER_R = "R"

# Canonical button-name parser for task JSON, probes, and debug tooling.
# Semantic convention for project scripts:
# - A = select / confirm
# - B = cancel / back
SNES_BUTTON_NAME_TO_INDEX = {
    name: idx
    for idx, name in enumerate(SNES_BUTTON_NAMES)
}
SNES_BUTTON_NAME_TO_INDEX["IDLE"] = None

# NES Button Map (fceumm): [B, null, Select, Start, Up, Down, Left, Right, A]
# Index 1 is unused in stable-retro's NES core layout.
NES_B = 0
NES_SELECT = 2
NES_START = 3
NES_UP = 4
NES_DOWN = 5
NES_LEFT = 6
NES_RIGHT = 7
NES_A = 8

NES_BUTTON_NAMES = (
    "B",
    "SELECT",
    "START",
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    "A",
)

# Full 9-slot layout including the unused hole at index 1.
NES_ACTION_SIZE = 9
NES_BUTTON_NAME_TO_INDEX = {
    "B": NES_B,
    "SELECT": NES_SELECT,
    "START": NES_START,
    "UP": NES_UP,
    "DOWN": NES_DOWN,
    "LEFT": NES_LEFT,
    "RIGHT": NES_RIGHT,
    "A": NES_A,
    "IDLE": None,
}


def action_from_nes_button_names(
    names: list[str] | tuple[str, ...],
    *,
    action_size: int = NES_ACTION_SIZE,
) -> list[int]:
    """Build a NES action vector from button names."""
    action = [0] * action_size
    for raw in names:
        name = str(raw).strip().upper()
        if name in ("", "IDLE", "NONE", "NOOP", "WAIT"):
            continue
        if name not in NES_BUTTON_NAME_TO_INDEX:
            raise ValueError(f"unknown NES button: {raw!r}")
        index = NES_BUTTON_NAME_TO_INDEX[name]
        if index is None:
            continue
        if index >= action_size:
            raise ValueError(f"button {name} index {index} >= action_size {action_size}")
        action[index] = 1
    return action


# Physical SNES-style controller mapping -> SNES buttons.
# Many USB SNES pads are exposed to SDL/pygame as Xbox-like controllers, but
# their face-button indices still follow physical SNES positions. Default to
# that layout so the bottom button is B, right is A, left is Y, and top is X.
# Use RETRO_PAD_SWAP_AB or RETRO_PAD_SWAP_XY for label-aligned modern pads.
CONTROLLER_MAP = {
    0: SNES_B,      # bottom -> B
    1: SNES_A,      # right -> A
    2: SNES_Y,      # left -> Y
    3: SNES_X,      # top -> X
    4: SNES_L,      # LB -> L
    5: SNES_R,      # RB -> R
    6: SNES_SELECT, # Back -> Select
    7: SNES_START,  # Start -> Start
}

# Conservative fallback mapping for controllers with shifted menu buttons.
# Older code treated 10-15 as possible start/select buttons, which collides
# with common D-pad-as-buttons layouts.
CONTROLLER_MAP_FALLBACK = {
    8: SNES_SELECT,  # Share/Back alternative
}

DPAD_BUTTON_MAP_AUTO = {
    11: SNES_UP,
    12: SNES_DOWN,
    13: SNES_LEFT,
    14: SNES_RIGHT,
}


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def _env_int(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw, 0)
    except ValueError:
        return None


def _env_button_csv(name: str) -> Optional[list[int]]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 4:
        return None
    values: list[int] = []
    for part in parts:
        try:
            values.append(int(part, 0))
        except ValueError:
            return None
    return values


def _resolved_controller_map(controller_map: Optional[dict[int, int]]) -> dict[int, int]:
    active_map = dict(controller_map or CONTROLLER_MAP)
    if controller_map is None:
        if _env_flag("RETRO_PAD_SWAP_AB") and 0 in active_map and 1 in active_map:
            active_map[0], active_map[1] = active_map[1], active_map[0]
        if _env_flag("RETRO_PAD_SWAP_XY") and 2 in active_map and 3 in active_map:
            active_map[2], active_map[3] = active_map[3], active_map[2]
    return active_map


def _resolved_button_override(explicit: Optional[int], env_name: str) -> Optional[int]:
    if explicit is not None:
        return explicit
    return _env_int(env_name)


def _resolved_dpad_button_map(
    joystick: Optional["pygame.joystick.Joystick"],
    explicit: Optional[dict[int, int]],
) -> Optional[dict[int, int]]:
    if explicit is not None:
        return explicit

    parsed = _env_button_csv("RETRO_PAD_DPAD_BUTTONS")
    if parsed is not None:
        up, down, left, right = parsed
        return {
            up: SNES_UP,
            down: SNES_DOWN,
            left: SNES_LEFT,
            right: SNES_RIGHT,
        }

    if joystick is None:
        return None
    if joystick.get_numhats() > 0:
        return None
    if joystick.get_numbuttons() >= 15:
        return DPAD_BUTTON_MAP_AUTO
    return None


def pressed_snes_buttons(action: list[int]) -> list[str]:
    return [
        SNES_BUTTON_NAMES[idx]
        for idx, value in enumerate(action[: len(SNES_BUTTON_NAMES)])
        if value
    ]


def pressed_nes_buttons(action: list[int]) -> list[str]:
    """Return pressed NES button names (skips the unused index-1 hole)."""
    names: list[str] = []
    for name, idx in NES_BUTTON_NAME_TO_INDEX.items():
        if name == "IDLE" or idx is None:
            continue
        if idx < len(action) and action[idx]:
            names.append(name)
    return names


def describe_input_mapping(
    *,
    controller_map: Optional[dict[int, int]] = None,
    joystick: Optional["pygame.joystick.Joystick"] = None,
    dpad_button_map: Optional[dict[int, int]] = None,
) -> dict[str, object]:
    """Describe the effective button semantics used by the current session."""
    active_map = _resolved_controller_map(controller_map)
    active_dpad = _resolved_dpad_button_map(joystick, dpad_button_map)
    return {
        "swap_ab": controller_map is None and _env_flag("RETRO_PAD_SWAP_AB"),
        "swap_xy": controller_map is None and _env_flag("RETRO_PAD_SWAP_XY"),
        "controller_buttons": {
            str(joy_btn): SNES_BUTTON_NAMES[snes_btn]
            for joy_btn, snes_btn in sorted(active_map.items())
        },
        "fallback_buttons": {
            str(joy_btn): SNES_BUTTON_NAMES[snes_btn]
            for joy_btn, snes_btn in sorted(CONTROLLER_MAP_FALLBACK.items())
        },
        "dpad_buttons": (
            {
                str(joy_btn): SNES_BUTTON_NAMES[snes_btn]
                for joy_btn, snes_btn in sorted(active_dpad.items())
            }
            if active_dpad
            else None
        ),
    }


def format_input_mapping(summary: dict[str, object]) -> str:
    """Render a compact one-line summary of the effective face-button map."""
    controller_buttons = summary.get("controller_buttons", {})
    face_labels = []
    for joy_btn in range(4):
        mapped = controller_buttons.get(str(joy_btn))
        if mapped is not None:
            face_labels.append(f"{joy_btn}->{mapped}")

    swaps = []
    if summary.get("swap_ab"):
        swaps.append("AB")
    if summary.get("swap_xy"):
        swaps.append("XY")
    swap_suffix = f" swaps={'+'.join(swaps)}" if swaps else ""
    joined = " ".join(face_labels) if face_labels else "controller map unavailable"
    return f"{joined}{swap_suffix}"


def parse_snes_button_label(label: str) -> tuple[str, ...]:
    names = tuple(part.strip().upper() for part in label.split("+") if part.strip())
    if not names:
        raise ValueError(f"No button names in label: {label!r}")
    for name in names:
        if name not in SNES_BUTTON_NAME_TO_INDEX:
            raise ValueError(f"Unknown button name: {name}")
    return names


def action_from_snes_button_names(
    button_names: list[str] | tuple[str, ...],
    action_size: int = 12,
) -> list[int]:
    action = [0] * action_size
    for name in button_names:
        button = SNES_BUTTON_NAME_TO_INDEX[name]
        if button is not None and 0 <= button < action_size:
            action[button] = 1
    return action


def describe_controller(joystick: Optional["pygame.joystick.Joystick"]) -> dict[str, object]:
    if joystick is None:
        return {"connected": False}

    info: dict[str, object] = {
        "connected": True,
        "name": joystick.get_name(),
        "buttons": joystick.get_numbuttons(),
        "axes": joystick.get_numaxes(),
        "hats": joystick.get_numhats(),
    }
    guid_fn = getattr(joystick, "get_guid", None)
    if callable(guid_fn):
        try:
            info["guid"] = guid_fn()
        except Exception:
            pass
    return info


def controller_debug_snapshot(
    joystick: Optional["pygame.joystick.Joystick"],
    *,
    deadzone: float = 0.5,
) -> dict[str, object]:
    if joystick is None:
        return {
            "connected": False,
            "raw_buttons": [],
            "axes": {},
            "hats": [],
            "mapped_buttons": [],
        }

    action = [0] * 12
    controller_action(joystick, action)
    sanitize_action(action)
    raw_buttons = [
        idx for idx in range(joystick.get_numbuttons())
        if joystick.get_button(idx)
    ]
    axes = {
        idx: round(float(joystick.get_axis(idx)), 3)
        for idx in range(joystick.get_numaxes())
        if abs(float(joystick.get_axis(idx))) >= deadzone / 2
    }
    hats = [
        joystick.get_hat(idx)
        for idx in range(joystick.get_numhats())
    ]
    return {
        "connected": True,
        "raw_buttons": raw_buttons,
        "axes": axes,
        "hats": hats,
        "mapped_buttons": pressed_snes_buttons(action),
    }


def init_controller(pygame) -> Optional["pygame.joystick.Joystick"]:
    """Initialize first available controller."""
    pygame.joystick.init()
    if pygame.joystick.get_count() > 0:
        joy = pygame.joystick.Joystick(0)
        joy.init()
        return joy
    return None


def init_controllers(pygame) -> list["pygame.joystick.Joystick"]:
    """Initialize all available controllers."""
    pygame.joystick.init()
    controllers: list["pygame.joystick.Joystick"] = []
    for i in range(pygame.joystick.get_count()):
        joy = pygame.joystick.Joystick(i)
        joy.init()
        controllers.append(joy)
    return controllers


def _apply_controller_action(
    joystick: Optional["pygame.joystick.Joystick"],
    action: list[int],
    offset: int = 0,
    controller_map: Optional[dict[int, int]] = None,
    fallback_map: dict[int, int] = CONTROLLER_MAP_FALLBACK,
    start_override: Optional[int] = None,
    select_override: Optional[int] = None,
    start_action_index: Optional[int] = None,
    select_action_index: Optional[int] = None,
    axis_map: Optional[dict[int, int]] = None,
    dpad_button_map: Optional[dict[int, int]] = None,
    deadzone: float = 0.5,
) -> None:
    """Update action array from controller input with optional offset."""
    if joystick is None:
        return

    # D-pad via hat
    if joystick.get_numhats() > 0:
        hat = joystick.get_hat(0)
        if hat[0] < 0:
            action[offset + SNES_LEFT] = 1
        if hat[0] > 0:
            action[offset + SNES_RIGHT] = 1
        if hat[1] > 0:
            action[offset + SNES_UP] = 1
        if hat[1] < 0:
            action[offset + SNES_DOWN] = 1

    # D-pad via analog stick
    if joystick.get_numaxes() >= 2:
        axis_x = joystick.get_axis(0)
        axis_y = joystick.get_axis(1)
        if axis_x < -deadzone:
            action[offset + SNES_LEFT] = 1
        if axis_x > deadzone:
            action[offset + SNES_RIGHT] = 1
        if axis_y < -deadzone:
            action[offset + SNES_UP] = 1
        if axis_y > deadzone:
            action[offset + SNES_DOWN] = 1

    # D-pad via buttons on controllers that do not expose hats.
    active_dpad_button_map = _resolved_dpad_button_map(joystick, dpad_button_map)
    if active_dpad_button_map:
        for joy_btn, snes_btn in active_dpad_button_map.items():
            if joy_btn < joystick.get_numbuttons() and joystick.get_button(joy_btn):
                action[offset + snes_btn] = 1

    # Buttons
    active_map = _resolved_controller_map(controller_map)
    start_override = _resolved_button_override(start_override, "RETRO_PAD_START_BUTTON")
    select_override = _resolved_button_override(select_override, "RETRO_PAD_SELECT_BUTTON")
    for joy_btn, snes_btn in active_map.items():
        if joy_btn < joystick.get_numbuttons() and joystick.get_button(joy_btn):
            target = snes_btn
            if snes_btn == SNES_START and start_action_index is not None:
                target = start_action_index
            elif snes_btn == SNES_SELECT and select_action_index is not None:
                target = select_action_index
            action[offset + target] = 1
    for joy_btn, snes_btn in fallback_map.items():
        if joy_btn < joystick.get_numbuttons() and joystick.get_button(joy_btn):
            target = snes_btn
            if snes_btn == SNES_START and start_action_index is not None:
                target = start_action_index
            elif snes_btn == SNES_SELECT and select_action_index is not None:
                target = select_action_index
            action[offset + target] = 1
    if start_override is not None:
        if start_override < joystick.get_numbuttons() and joystick.get_button(start_override):
            target = start_action_index if start_action_index is not None else SNES_START
            action[offset + target] = 1
    if select_override is not None:
        if select_override < joystick.get_numbuttons() and joystick.get_button(select_override):
            target = select_action_index if select_action_index is not None else SNES_SELECT
            action[offset + target] = 1
    if axis_map:
        for axis_idx, snes_btn in axis_map.items():
            if axis_idx < joystick.get_numaxes():
                if joystick.get_axis(axis_idx) > deadzone:
                    action[offset + snes_btn] = 1


def controller_action(
    joystick: Optional["pygame.joystick.Joystick"],
    action: list[int],
    offset: int = 0,
    start_override: Optional[int] = None,
    select_override: Optional[int] = None,
    start_action_index: Optional[int] = None,
    select_action_index: Optional[int] = None,
    controller_map: Optional[dict[int, int]] = None,
    axis_map: Optional[dict[int, int]] = None,
    dpad_button_map: Optional[dict[int, int]] = None,
) -> None:
    """Update action array from controller input."""
    _apply_controller_action(
        joystick,
        action,
        offset=offset,
        start_override=start_override,
        select_override=select_override,
        start_action_index=start_action_index,
        select_action_index=select_action_index,
        controller_map=controller_map,
        axis_map=axis_map,
        dpad_button_map=dpad_button_map,
    )


def keyboard_action(
    keys,
    action: list[int],
    pygame,
    start_action_index: Optional[int] = None,
    select_action_index: Optional[int] = None,
) -> None:
    """Update action array from keyboard input.

    Default mapping:
    - Arrow keys: D-pad
    - Z: B (run/cancel)
    - C: A (confirm/talk)
    - X: Y (use item/tool)
    - V: X (menu)
    - A/Q: L shoulder
    - S/W: R shoulder
    - Enter: Start
    - Shift: Select
    """
    if keys[pygame.K_RIGHT]:
        action[SNES_RIGHT] = 1
    if keys[pygame.K_LEFT]:
        action[SNES_LEFT] = 1
    if keys[pygame.K_DOWN]:
        action[SNES_DOWN] = 1
    if keys[pygame.K_UP]:
        action[SNES_UP] = 1

    if keys[pygame.K_RETURN]:
        start_idx = start_action_index if start_action_index is not None else SNES_START
        action[start_idx] = 1
    if keys[pygame.K_RSHIFT] or keys[pygame.K_LSHIFT]:
        select_idx = select_action_index if select_action_index is not None else SNES_SELECT
        action[select_idx] = 1

    if keys[pygame.K_z]:
        action[SNES_B] = 1
    if keys[pygame.K_c]:
        action[SNES_A] = 1
    if keys[pygame.K_x]:
        action[SNES_Y] = 1
    if keys[pygame.K_v]:
        action[SNES_X] = 1
    if keys[pygame.K_a] or keys[pygame.K_q]:
        action[SNES_L] = 1
    if keys[pygame.K_s] or keys[pygame.K_w]:
        action[SNES_R] = 1


def sanitize_action_offset(action: list[int], offset: int = 0) -> None:
    """Remove conflicting directional inputs for a player slice."""
    if action[offset + SNES_LEFT] and action[offset + SNES_RIGHT]:
        action[offset + SNES_LEFT] = 0
        action[offset + SNES_RIGHT] = 0
    if action[offset + SNES_UP] and action[offset + SNES_DOWN]:
        action[offset + SNES_UP] = 0
        action[offset + SNES_DOWN] = 0


def sanitize_action(action: list[int]) -> None:
    """Remove conflicting directional inputs."""
    sanitize_action_offset(action, offset=0)


def sanitize_action_multi(action: list[int], players: int = 2, stride: int = 12) -> None:
    """Remove conflicting directional inputs across multiple players."""
    for idx in range(players):
        start = idx * stride
        if start + stride <= len(action):
            sanitize_action_offset(action, offset=start)
