"""Tests for retro_harness.controls."""

from __future__ import annotations

from retro_harness.controls import (
    SNES_A,
    SNES_B,
    SNES_BUTTON_NAME_TO_INDEX,
    SNES_DOWN,
    SNES_L,
    SNES_LEFT,
    SNES_R,
    SNES_RIGHT,
    SNES_SELECT,
    SNES_START,
    SNES_UP,
    SNES_X,
    SNES_Y,
    action_from_snes_button_names,
    controller_action,
    controller_debug_snapshot,
    keyboard_action,
    parse_snes_button_label,
)


class _FakeJoystick:
    def __init__(
        self,
        *,
        buttons: set[int] | None = None,
        axes: dict[int, float] | None = None,
        hats: list[tuple[int, int]] | None = None,
        numbuttons: int = 16,
        name: str = "FakePad",
    ) -> None:
        self._buttons = buttons or set()
        self._axes = axes or {}
        self._hats = hats or []
        self._numbuttons = numbuttons
        self._name = name

    def get_name(self) -> str:
        return self._name

    def get_numbuttons(self) -> int:
        return self._numbuttons

    def get_button(self, idx: int) -> int:
        return int(idx in self._buttons)

    def get_numaxes(self) -> int:
        if not self._axes:
            return 0
        return max(self._axes) + 1

    def get_axis(self, idx: int) -> float:
        return float(self._axes.get(idx, 0.0))

    def get_numhats(self) -> int:
        return len(self._hats)

    def get_hat(self, idx: int) -> tuple[int, int]:
        return self._hats[idx]

    def get_guid(self) -> str:
        return "FAKE-GUID"


def test_controller_action_maps_face_buttons():
    cases = {
        0: SNES_B,
        1: SNES_A,
        2: SNES_Y,
        3: SNES_X,
        6: SNES_SELECT,
        7: SNES_START,
    }

    for raw_button, expected in cases.items():
        joy = _FakeJoystick(buttons={raw_button})
        action = [0] * 12
        controller_action(joy, action)
        assert action[expected] == 1


def test_controller_action_auto_maps_dpad_buttons_when_no_hat():
    joy = _FakeJoystick(buttons={11, 14}, hats=[], numbuttons=16)
    action = [0] * 12

    controller_action(joy, action)

    assert action[SNES_UP] == 1
    assert action[SNES_RIGHT] == 1
    assert action[SNES_START] == 0
    assert action[SNES_SELECT] == 0


def test_controller_action_does_not_map_l3_as_start_by_default():
    joy = _FakeJoystick(buttons={9}, hats=[], numbuttons=10)
    action = [0] * 12

    controller_action(joy, action)

    assert action[SNES_START] == 0


def test_controller_action_respects_env_swaps(monkeypatch):
    monkeypatch.setenv("RETRO_PAD_SWAP_AB", "1")
    monkeypatch.setenv("RETRO_PAD_SWAP_XY", "1")
    cases = {
        0: SNES_A,
        1: SNES_B,
        2: SNES_X,
        3: SNES_Y,
    }

    for raw_button, expected in cases.items():
        joy = _FakeJoystick(buttons={raw_button})
        action = [0] * 12
        controller_action(joy, action)
        assert action[expected] == 1


def test_controller_action_respects_env_start_select_and_dpad(monkeypatch):
    monkeypatch.setenv("RETRO_PAD_START_BUTTON", "10")
    monkeypatch.setenv("RETRO_PAD_SELECT_BUTTON", "9")
    monkeypatch.setenv("RETRO_PAD_DPAD_BUTTONS", "4,5,6,7")
    joy = _FakeJoystick(buttons={4, 7, 9, 10}, hats=[], numbuttons=16)
    action = [0] * 12

    controller_action(joy, action)

    assert action[SNES_UP] == 1
    assert action[SNES_RIGHT] == 1
    assert action[SNES_SELECT] == 1
    assert action[SNES_START] == 1


def test_controller_debug_snapshot_reports_mapped_buttons():
    joy = _FakeJoystick(
        buttons={0, 11},
        axes={0: -0.75, 1: 0.0},
        hats=[(0, -1)],
    )

    snapshot = controller_debug_snapshot(joy)

    assert snapshot["connected"] is True
    assert snapshot["raw_buttons"] == [0, 11]
    assert snapshot["hats"] == [(0, -1)]
    assert "B" in snapshot["mapped_buttons"]
    assert "LEFT" in snapshot["mapped_buttons"]
    assert "DOWN" in snapshot["mapped_buttons"]


def test_parse_snes_button_label_uses_shared_name_map():
    assert parse_snes_button_label("A+B") == ("A", "B")
    assert parse_snes_button_label("idle") == ("IDLE",)
    assert SNES_BUTTON_NAME_TO_INDEX["A"] == SNES_A
    assert SNES_BUTTON_NAME_TO_INDEX["B"] == SNES_B
    assert SNES_BUTTON_NAME_TO_INDEX["LEFT"] == SNES_LEFT
    assert SNES_BUTTON_NAME_TO_INDEX["L"] == SNES_L
    assert SNES_BUTTON_NAME_TO_INDEX["LEFT"] != SNES_BUTTON_NAME_TO_INDEX["L"]
    assert SNES_BUTTON_NAME_TO_INDEX["RIGHT"] != SNES_BUTTON_NAME_TO_INDEX["R"]


def test_action_from_snes_button_names_builds_expected_action():
    action = action_from_snes_button_names(["A", "LEFT"], action_size=12)

    assert action[SNES_A] == 1
    assert action[SNES_LEFT] == 1
    assert action[SNES_B] == 0


class _FakePygame:
    K_RIGHT = "right"
    K_LEFT = "left"
    K_DOWN = "down"
    K_UP = "up"
    K_RETURN = "return"
    K_RSHIFT = "rshift"
    K_LSHIFT = "lshift"
    K_z = "z"
    K_c = "c"
    K_x = "x"
    K_v = "v"
    K_a = "a"
    K_s = "s"
    K_q = "q"
    K_w = "w"


class _FakeKeys:
    def __init__(self, pressed: set[str]) -> None:
        self._pressed = pressed

    def __getitem__(self, key: str) -> bool:
        return key in self._pressed


def test_keyboard_action_uses_harvest_snes_layout():
    keys = _FakeKeys({"z", "c", "x", "v", "a", "s"})
    action = [0] * 12

    keyboard_action(keys, action, _FakePygame)

    assert action[SNES_B] == 1
    assert action[SNES_A] == 1
    assert action[SNES_Y] == 1
    assert action[SNES_X] == 1
    assert action[SNES_L] == 1
    assert action[SNES_R] == 1
