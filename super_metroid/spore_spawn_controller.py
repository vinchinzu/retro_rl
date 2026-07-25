"""Controller-only post-Torizo route through Spore Spawn.

The room sequence is pre-calculated from the Super Metroid editor export.  This
module owns movement and combat only: it reads typed state and emits ordinary
12-button controller actions.  It never loads emulator state or writes RAM.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np

from retro_harness.actions import buttons, idle_action
from super_metroid.ram import GameplayPhase, SuperMetroidState


class ControllerSession(Protocol):
    frame: int
    state: SuperMetroidState

    def step(self, action: np.ndarray, reason: str) -> SuperMetroidState: ...


@dataclass(frozen=True)
class SporeSpawnEvidence:
    entry_frame: int
    activation_frame: int
    defeat_frame: int
    exit_frame: int
    peak_hp: int
    observed_hp: tuple[int, ...]
    brinstar_boss_bits_before: int
    brinstar_boss_bits_after: int
    vulnerable_spritemaps: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_LJ = ("LEFT", "A", "B", "X")
_RJ = ("RIGHT", "A", "B", "X")
_RR = ("RIGHT", "B", "X")
_LR = ("LEFT", "B", "X")
_J = ("A", "B", "X")

# Map-guided development searches generated these finite controller sequences.
# They are replayed only from checked natural room/coordinate boundaries.
_BIG_PINK_CLIMB = (
    _LJ, _RJ, _RJ, _LJ, _RJ, _RJ, _LJ, _RJ, _RJ, _RR, _RJ, (), (), _LJ,
    _RJ, _RJ, _RJ, _RJ, _RJ, (), _RJ, _RJ, _RJ, _RJ, (), _J, _RJ, _RJ,
    _RJ, _RJ, _LJ, _RR, _J, _LJ, _LJ, _LJ, _LJ, _LJ, (), _LJ, _LJ, _J,
    _LJ, _LJ, _RJ, _LJ, _LJ, _RJ, _LJ, _LJ, _LJ, _RJ, _LJ, _LJ, _RJ,
    _LJ, _LJ, _J, _RJ, _RR, _RR, _RJ, _RJ, _RJ, _RJ, _LJ,
)

_SPORE_EXIT_CLIMB = (
    _LR, _LR, _RJ, _LJ, _RR, _LJ, _LJ, (), _RJ, _RJ, _RJ, _RJ, _RJ, _LJ,
    _LJ, _RJ, _LR, _LJ, _LJ, _LJ, _RJ, _LJ, _LR, _LR, _RR, _RJ, _RJ,
    _RR, _RR, _LR, _LJ, _LJ, _LJ, _RJ, _LJ, _J, _RJ, _LJ, _LR, _RJ,
    _RJ, _RJ, _RJ, _RJ, _J, _RJ, (), _LJ, (), _J, _LJ,
)

_VULNERABLE_SPRITEMAPS = frozenset({0xEEAF, 0xEEC1, 0xEED3, 0xEEE5})


def _hold(
    session: ControllerSession,
    frames: int,
    *names: str,
    reason: str,
) -> SuperMetroidState:
    action = buttons(*names) if names else idle_action()
    state = session.state
    for _ in range(frames):
        state = session.step(action, reason)
    return state


def _require_room(
    session: ControllerSession,
    room_id: int,
    label: str,
    *,
    ordinary: bool = True,
) -> None:
    state = session.state
    if state.room_id != room_id or (
        ordinary and state.phase is not GameplayPhase.ORDINARY_GAMEPLAY
    ):
        raise RuntimeError(
            f"{label}: expected room 0x{room_id:04X}, got {state}"
        )


def _hold_until_room(
    session: ControllerSession,
    target_room_id: int,
    timeout: int,
    *names: str,
    reason: str,
) -> None:
    action = buttons(*names) if names else idle_action()
    for _ in range(timeout):
        if session.state.room_id == target_room_id:
            return
        session.step(action, reason)
    raise TimeoutError(
        f"{reason}: did not reach 0x{target_room_id:04X}: {session.state}"
    )


def play_parlor_to_main_shaft(session: ControllerSession) -> None:
    """Travel from the accepted post-Torizo Parlor settle to Green Brinstar."""
    _require_room(session, 0x92FD, "post-Torizo entry")
    if not session.state.bombs or session.state.max_missiles < 10:
        raise RuntimeError(f"post-Torizo capabilities missing: {session.state}")
    if session.state.samus_x > 956:
        alignment_frames = 10 if session.state.pose == 2 else 15
        _hold(
            session,
            alignment_frames,
            "LEFT",
            reason="post_torizo_parlor_alignment",
        )
        _hold(session, 10, reason="post_torizo_parlor_alignment")
    if not (
        948 <= session.state.samus_x <= 954
        and session.state.samus_y == 651
        and session.state.pose == 2
    ):
        raise RuntimeError(f"post-Torizo Parlor alignment missed: {session.state}")

    for _ in range(2):
        _hold(session, 20, "LEFT", "A", "B", "X", reason="parlor_left_traverse")
        _hold(session, 12, "LEFT", "B", "X", reason="parlor_left_traverse")
    _hold(session, 50, reason="parlor_left_traverse_settle")
    for _ in range(6):
        _hold(session, 30, "RIGHT", "A", reason="parlor_chimney_climb")
        _hold(session, 12, reason="parlor_chimney_climb")
    for _ in range(2):
        _hold(session, 30, "LEFT", "A", reason="parlor_chimney_climb")
        _hold(session, 12, reason="parlor_chimney_climb")
    _hold(session, 100, reason="parlor_chimney_settle")
    _hold(session, 30, "LEFT", "A", reason="parlor_upper_platforms")
    _hold(session, 30, "LEFT", reason="parlor_upper_platforms")
    _hold(session, 60, reason="parlor_upper_platforms")
    _hold(session, 40, "LEFT", "A", reason="parlor_upper_platforms")
    _hold(session, 100, reason="parlor_upper_platforms")
    for names, frames in (
        (("RIGHT", "A"), 30),
        (("RIGHT",), 15),
        (("LEFT",), 10),
        ((), 100),
        (("RIGHT", "A"), 10),
        (("RIGHT",), 40),
        (("LEFT",), 10),
        ((), 100),
        (("RIGHT", "A"), 20),
        (("RIGHT",), 30),
        ((), 100),
        (("LEFT", "A"), 40),
        (("LEFT",), 16),
        ((), 30),
        (("RIGHT", "B"), 21),
        (("RIGHT", "A", "B"), 8),
        (("LEFT",), 8),
        (("LEFT", "A"), 50),
        ((), 40),
        (("RIGHT", "A"), 35),
        ((), 100),
    ):
        _hold(session, frames, *names, reason="parlor_upper_platforms")
    _hold(session, 2, "DOWN", reason="parlor_bomb_tunnel_morph")
    _hold(session, 3, reason="parlor_bomb_tunnel_morph")
    _hold(session, 2, "DOWN", reason="parlor_bomb_tunnel_morph")
    _hold(session, 10, reason="parlor_bomb_tunnel_morph")
    for _ in range(10):
        _hold(session, 45, "RIGHT", "X", reason="parlor_bomb_tunnel")
        _hold(session, 15, "RIGHT", reason="parlor_bomb_tunnel")
    _hold(session, 100, reason="parlor_bomb_tunnel_settle")
    for _ in range(9):
        _hold(session, 50, "LEFT", "A", "B", "X", reason="parlor_terminator_exit")
        _hold(session, 10, "LEFT", "B", "X", reason="parlor_terminator_exit")
    _hold(session, 100, reason="terminator_entry_settle")
    _hold(session, 2, "DOWN", reason="terminator_morph")
    _hold(session, 3, reason="terminator_morph")
    _hold(session, 2, "DOWN", reason="terminator_morph")
    _hold(session, 10, reason="terminator_morph")
    for _ in range(8):
        _hold(session, 45, "LEFT", "X", reason="terminator_bomb_tunnel")
        _hold(session, 15, "LEFT", reason="terminator_bomb_tunnel")
    _require_room(session, 0x990D, "Terminator traversal")

    for _ in range(7):
        _hold(session, 50, "LEFT", "A", "B", "X", reason="terminator_energy_tank")
        _hold(session, 10, "LEFT", "B", "X", reason="terminator_energy_tank")
    for _ in range(30):
        _hold(session, 10, "LEFT", reason="collect_terminator_energy_tank")
        if session.state.max_health >= 199:
            break
    if session.state.max_health < 199:
        raise RuntimeError(f"Terminator Energy Tank was not collected: {session.state}")
    _hold_until_room(
        session,
        0x99BD,
        900,
        "LEFT",
        "A",
        "B",
        "X",
        reason="exit_terminator",
    )
    _hold(session, 180, reason="green_pirates_entry_settle")
    _require_room(session, 0x99BD, "Green Pirates entry")

    _hold(session, 100, "LEFT", "B", "X", reason="green_pirates_descent")
    for direction in ("RIGHT", "LEFT", "RIGHT", "LEFT", "RIGHT", "LEFT", "RIGHT", "LEFT"):
        _hold(
            session,
            80,
            direction,
            "B",
            "X",
            reason="green_pirates_descent",
        )
    _hold(session, 180, reason="lower_mushrooms_entry_settle")
    _require_room(session, 0x9969, "Lower Mushrooms entry")

    for _ in range(13):
        _hold(session, 60, "LEFT", "A", "B", "X", reason="lower_mushrooms")
    _hold(session, 240, reason="green_elevator_entry_settle")
    _require_room(session, 0x9938, "Green elevator entry")
    for _ in range(30):
        state = session.state
        if 118 <= state.samus_x <= 126 and state.velocity_x == 0:
            break
        if state.samus_x < 118:
            direction = "RIGHT"
        elif state.samus_x > 126:
            direction = "LEFT"
        else:
            direction = "RIGHT" if state.velocity_x < 0 else "LEFT"
        _hold(session, 10, direction, reason="green_elevator_center")
        _hold(session, 10, reason="green_elevator_center")
    else:
        raise RuntimeError(f"Could not center on Green Brinstar elevator: {session.state}")
    _hold(session, 10, "DOWN", reason="green_elevator_descend")
    _hold(session, 1_000, reason="green_elevator_descent_settle")
    _require_room(session, 0x9AD9, "Green Brinstar Main Shaft landing")


def play_main_shaft_to_spore_spawn(session: ControllerSession) -> SporeSpawnEvidence:
    """Take the editor-planned route, defeat Spore Spawn, and exit naturally."""
    _require_room(session, 0x9AD9, "Main Shaft route entry")

    _hold(session, 1_000, reason="main_shaft_entry_settle")
    for names in (
        ("RIGHT", "B"),
        ("LEFT", "B"),
        ("RIGHT", "B"),
        ("LEFT", "B"),
    ):
        _hold(session, 60, *names, reason="main_shaft_descent")
    _hold(session, 50, reason="main_shaft_descent_settle")
    for names in (
        ("RIGHT", "B"),
        ("LEFT", "B"),
        ("RIGHT", "B"),
        ("LEFT", "B"),
        ("RIGHT", "B"),
    ):
        _hold(session, 80, *names, reason="main_shaft_dachora_level")
    _hold(session, 30, reason="main_shaft_dachora_door_settle")
    _hold(session, 1, "SELECT", reason="select_missiles")
    _hold(session, 10, reason="select_missiles_settle")
    for _ in range(15):
        _hold(session, 2, "X", reason="open_dachora_red_door")
        _hold(session, 15, reason="open_dachora_red_door")
    _hold(session, 100, "RIGHT", "B", reason="enter_dachora")
    _hold(session, 250, reason="dachora_entry_settle")
    _require_room(session, 0x9CB3, "Dachora entry")

    _hold(session, 350, "RIGHT", "A", "B", "X", reason="cross_dachora")
    _hold(session, 2, "DOWN", reason="dachora_tunnel_morph")
    _hold(session, 3, reason="dachora_tunnel_morph")
    _hold(session, 2, "DOWN", reason="dachora_tunnel_morph")
    _hold(session, 10, reason="dachora_tunnel_morph")
    for _ in range(15):
        _hold(session, 45, "RIGHT", "X", reason="bomb_dachora_tunnel")
        _hold(session, 15, "RIGHT", reason="bomb_dachora_tunnel")
    _hold(session, 160, "RIGHT", "A", "B", "X", reason="exit_dachora")
    _hold(session, 300, reason="big_pink_entry_settle")
    _require_room(session, 0x9D19, "Big Pink entry")

    _hold(session, 2, "UP", reason="unmorph_big_pink")
    _hold(session, 10, reason="unmorph_big_pink")
    _hold(session, 180, "RIGHT", "A", "B", "X", reason="big_pink_climb")
    _hold(session, 80, "LEFT", "A", "B", "X", reason="big_pink_climb")
    for names in _BIG_PINK_CLIMB:
        _hold(session, 16, *names, reason="big_pink_map_guided_climb")
    if not (session.state.samus_y <= 150 and session.state.samus_x >= 780):
        raise RuntimeError(f"Big Pink climb missed upper-right door: {session.state}")
    _hold(session, 100, "RIGHT", "B", "X", reason="big_pink_red_door_approach")
    for _ in range(15):
        _hold(session, 2, "X", reason="open_kihunter_red_door")
        _hold(session, 15, reason="open_kihunter_red_door")
    _hold(session, 150, "RIGHT", "B", reason="enter_kihunter")
    _hold(session, 300, reason="kihunter_entry_settle")
    _require_room(session, 0x9D9C, "Spore Kihunter entry")

    for index in range(8):
        direction = "RIGHT" if index % 2 == 0 else "LEFT"
        _hold(
            session,
            180,
            direction,
            "A",
            "B",
            "X",
            reason="clear_spore_kihunters",
        )
    for index in range(240):
        names = (
            ("UP", "X"),
            ("LEFT", "UP", "X"),
            ("RIGHT", "UP", "X"),
            ("LEFT", "X"),
            ("RIGHT", "X"),
        )[index % 5]
        _hold(session, 2, *names, reason="aim_at_spore_kihunters")
        _hold(
            session,
            8,
            *tuple(name for name in names if name != "X"),
            reason="aim_at_spore_kihunters",
        )
    _hold(session, 300, reason="kihunter_clear_settle")
    if session.state.enemies_killed < 3:
        raise RuntimeError(f"Spore Kihunters did not clear naturally: {session.state}")
    _hold(session, 80, "RIGHT", "B", reason="kihunter_boss_door_runway")
    _hold(session, 100, "RIGHT", "A", "B", "X", reason="kihunter_boss_door_jump")
    _hold(session, 10, reason="release_kihunter_jump")
    _hold(session, 80, "RIGHT", "A", "B", "X", reason="align_spore_spawn_door")
    _hold(session, 10, reason="release_kihunter_door_align")
    _hold(session, 30, "LEFT", "B", reason="center_under_spore_spawn_door")
    _hold(session, 60, reason="center_under_spore_spawn_door")
    for _ in range(15):
        _hold(session, 2, "UP", "X", reason="open_spore_spawn_door")
        _hold(session, 10, "UP", reason="open_spore_spawn_door")
    _hold(session, 10, reason="release_spore_spawn_door_shot")
    _hold(session, 120, "UP", "A", "B", reason="enter_spore_spawn")
    _hold(session, 300, reason="spore_spawn_entry_settle")
    _require_room(session, 0x9DC7, "Spore Spawn entry")
    if session.state.enemy0_hp < 960:
        raise RuntimeError(f"Spore Spawn did not activate at 960 HP: {session.state}")

    entry_frame = session.frame
    activation_frame = session.frame
    peak_hp = session.state.enemy0_hp
    observed_hp = {session.state.enemy0_hp}
    boss_bits_before = session.state.boss_bits[1]
    seen_spritemaps: set[int] = set()
    jump_direction = "RIGHT"
    jump_hold = 0
    for index in range(30_000):
        state = session.state
        peak_hp = max(peak_hp, state.enemy0_hp)
        observed_hp.add(state.enemy0_hp)
        if state.enemy0_spritemap in _VULNERABLE_SPRITEMAPS:
            seen_spritemaps.add(state.enemy0_spritemap)
        if state.samus_x <= 65:
            jump_direction = "RIGHT"
        elif state.samus_x >= 191:
            jump_direction = "LEFT"
        if state.samus_y >= 710 and jump_hold == 0:
            jump_hold = 36
        hold_jump = jump_hold > 0
        jump_hold = max(0, jump_hold - 1)
        if state.samus_y >= 710:
            names = (
                (jump_direction, "A", "B")
                if hold_jump
                else (jump_direction, "A")
            )
        else:
            aim_direction = "LEFT" if state.enemy0_x < state.samus_x else "RIGHT"
            names_list = [aim_direction, "UP"]
            if hold_jump:
                names_list.extend(("A", "B"))
            if (
                state.enemy0_spritemap in _VULNERABLE_SPRITEMAPS
                and index % 4 == 0
            ):
                names_list.append("X")
            names = tuple(names_list)
        _hold(session, 1, *names, reason="fight_spore_spawn")
        if session.state.enemy0_hp == 0:
            observed_hp.add(0)
            break
    else:
        raise TimeoutError(f"Spore Spawn HP never reached zero: {session.state}")
    defeat_frame = session.frame

    _hold(session, 600, reason="spore_spawn_death_settle")
    for names in _SPORE_EXIT_CLIMB:
        _hold(session, 16, *names, reason="spore_exit_map_guided_climb")
    if not (session.state.samus_y <= 150 and session.state.samus_x >= 170):
        raise RuntimeError(f"Spore exit climb missed upper-right door: {session.state}")
    for _ in range(20):
        _hold(session, 2, "RIGHT", "X", reason="open_spore_exit_door")
        _hold(session, 8, "RIGHT", reason="open_spore_exit_door")
    _hold(session, 300, reason="spore_spawn_exit_settle")
    _require_room(session, 0x9B5B, "Spore Spawn natural exit")

    return SporeSpawnEvidence(
        entry_frame=entry_frame,
        activation_frame=activation_frame,
        defeat_frame=defeat_frame,
        exit_frame=session.frame,
        peak_hp=peak_hp,
        observed_hp=tuple(sorted(observed_hp, reverse=True)),
        brinstar_boss_bits_before=boss_bits_before,
        brinstar_boss_bits_after=session.state.boss_bits[1],
        vulnerable_spritemaps=tuple(sorted(seen_spritemaps)),
    )


def play_post_torizo_to_spore_spawn(
    session: ControllerSession,
) -> SporeSpawnEvidence:
    """Run the complete checked post-Torizo controller slice."""
    play_parlor_to_main_shaft(session)
    return play_main_shaft_to_spore_spawn(session)
