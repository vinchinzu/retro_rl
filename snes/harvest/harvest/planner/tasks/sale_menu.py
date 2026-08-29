"""Animal-shop sale menu classification and counter-menu input tapes."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from harvest.core.ram_catalog import field_spec, read_ram_u16
from harvest.tasks.nav import get_pos_from_ram, make_action
from harvest.tasks.primitives import press_a_sequence, press_button_sequence


SHOP_MENU_TEXT_ID = 0x0305
SELL_CHICKEN_TEXT_ID = 0x030B


def dialog_text_id(ram: np.ndarray) -> int:
    return read_ram_u16(ram, field_spec("dialog_text_id").address, live_offset=False)


def near_px(
    ram: np.ndarray, target: Tuple[int, int], tolerance: int
) -> bool:
    pos = get_pos_from_ram(ram)
    return abs(pos.x - target[0]) <= tolerance and abs(pos.y - target[1]) <= tolerance


def sale_request_ready(saw_shop_menu: bool, ram: np.ndarray, request_text_id: int) -> bool:
    """True once the four-option shop menu has advanced to the sale prompt."""
    return saw_shop_menu and dialog_text_id(ram) == request_text_id


def open_counter_menu_actions() -> List[np.ndarray]:
    return list(
        press_a_sequence(
            "right",
            face_frames=4,
            pre_press_settle_frames=0,
            hold_frames=12,
            settle_frames=12,
            hold_face_with_a=True,
        )
    )


def sell_chicken_choice_actions() -> List[np.ndarray]:
    """Select "sell chicken" (dialog 0x030B) from the four-option shop menu."""
    actions: List[np.ndarray] = []
    actions.extend(make_action(right=True, a=True) for _ in range(6))
    actions.extend(make_action(right=True) for _ in range(7))
    actions.extend(make_action(right=True, a=True) for _ in range(3))
    actions.extend(make_action(a=True) for _ in range(4))
    actions.extend(make_action() for _ in range(12))
    actions.extend(press_button_sequence("a", hold_frames=8, settle_frames=19))
    actions.extend(press_button_sequence("a", hold_frames=8, settle_frames=141))
    actions.extend(press_button_sequence("down", hold_frames=11, settle_frames=7))
    actions.extend(press_button_sequence("down", hold_frames=7, settle_frames=16))
    actions.extend(press_button_sequence("a", hold_frames=10, settle_frames=30))
    return actions


def payout_press_actions() -> List[np.ndarray]:
    return list(
        press_button_sequence(
            "a",
            face="down",
            face_frames=8,
            pre_press_settle_frames=0,
            hold_frames=10,
            settle_frames=45,
            hold_face_with_button=True,
        )
    )


def chicken_count_reason(
    start_chickens: int,
    current_chickens: int,
    start_money: int,
    current_money: int,
    *,
    money_label: str = "shipping",
    extra: Optional[str] = None,
) -> str:
    text = (
        f"chickens {start_chickens}->{current_chickens} "
        f"{money_label} {start_money}->{current_money}"
    )
    if extra:
        return f"{text} {extra}"
    return text


__all__ = [
    "SELL_CHICKEN_TEXT_ID",
    "SHOP_MENU_TEXT_ID",
    "chicken_count_reason",
    "dialog_text_id",
    "near_px",
    "open_counter_menu_actions",
    "payout_press_actions",
    "sale_request_ready",
    "sell_chicken_choice_actions",
]
