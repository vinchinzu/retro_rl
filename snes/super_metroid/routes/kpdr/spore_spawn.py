"""Post-Torizo approach + Spore room leave (KPDR continuous).

Movement only. The Survival fight is
:func:`super_metroid.combat.spore_spawn.play_spore_spawn_floor_bounce`.
The post-Torizo Parlor escape is :mod:`kpdr.alcatraz_escape`. Gauntlet's
right-side parlor chimney lives in ``gauntlet.parlor_to_landing``.
This module never loads emulator state or writes RAM.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from super_metroid.combat.spore_spawn import play_spore_spawn_floor_bounce
from super_metroid.ram import GameplayPhase
from super_metroid.routes.controller_common import (
    hold_until,
    require_room,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr.alcatraz_escape import play_alcatraz_escape
from super_metroid.routes.kpdr.room_ids import ROOM_SPORE_SPAWN, ROOM_SUPER, ROOM_TERMINATOR
from super_metroid.routes.runtime import ControllerSession, hold

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


def _require_room(
    session: ControllerSession,
    room_id: int,
    label: str,
    *,
    ordinary: bool = True,
) -> None:
    """Room assert via controller_common plus optional ordinary-phase gate.

    Spore route requires ORDINARY_GAMEPLAY as well as room id.
    ``require_room`` only checks room; phase gate stays local.
    """
    require_room(session, room_id, label)
    if ordinary and session.state.phase is not GameplayPhase.ORDINARY_GAMEPLAY:
        raise RuntimeError(
            f"{label}: expected room 0x{room_id:04X}, got {session.state}"
        )


def play_parlor_to_main_shaft(session: ControllerSession) -> None:
    """Travel from the accepted post-Torizo Parlor settle to Green Brinstar."""
    _require_room(session, 0x92FD, "post-Torizo entry")
    if not session.state.bombs or session.state.max_missiles < 10:
        raise RuntimeError(f"post-Torizo capabilities missing: {session.state}")
    play_alcatraz_escape(session)
    for _ in range(9):
        hold(session, 50, "LEFT", "A", "B", "X", reason="parlor_terminator_exit")
        hold(session, 10, "LEFT", "B", "X", reason="parlor_terminator_exit")
    hold(session, 100, reason="terminator_entry_settle")
    hold(session, 2, "DOWN", reason="terminator_morph")
    hold(session, 3, reason="terminator_morph")
    hold(session, 2, "DOWN", reason="terminator_morph")
    hold(session, 10, reason="terminator_morph")
    for _ in range(8):
        hold(session, 45, "LEFT", "X", reason="terminator_bomb_tunnel")
        hold(session, 15, "LEFT", reason="terminator_bomb_tunnel")
    wait_ordinary_room(
        session,
        ROOM_TERMINATOR,
        settle_frames=240,
        label="terminator_traversal",
    )

    for _ in range(7):
        hold(session, 50, "LEFT", "A", "B", "X", reason="terminator_energy_tank")
        hold(session, 10, "LEFT", "B", "X", reason="terminator_energy_tank")
    for _ in range(30):
        hold(session, 10, "LEFT", reason="collect_terminator_energy_tank")
        if session.state.max_health >= 199:
            break
    if session.state.max_health < 199:
        raise RuntimeError(f"Terminator Energy Tank was not collected: {session.state}")
    hold_until(
        session,
        lambda state: state.room_id == 0x99BD,
        "LEFT",
        "A",
        "B",
        "X",
        timeout=600,
        reason="exit_terminator",
    )
    hold(session, 180, reason="green_pirates_entry_settle")
    _require_room(session, 0x99BD, "Green Pirates entry")

    hold(session, 100, "LEFT", "B", "X", reason="green_pirates_descent")
    for direction in ("RIGHT", "LEFT", "RIGHT", "LEFT", "RIGHT", "LEFT", "RIGHT", "LEFT"):
        hold(
            session,
            80,
            direction,
            "B",
            "X",
            reason="green_pirates_descent",
        )
    hold(session, 180, reason="lower_mushrooms_entry_settle")
    _require_room(session, 0x9969, "Lower Mushrooms entry")

    for _ in range(13):
        hold(session, 60, "LEFT", "A", "B", "X", reason="lower_mushrooms")
    hold(session, 240, reason="green_elevator_entry_settle")
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
        hold(session, 10, direction, reason="green_elevator_center")
        hold(session, 10, reason="green_elevator_center")
    else:
        raise RuntimeError(f"Could not center on Green Brinstar elevator: {session.state}")
    hold(session, 10, "DOWN", reason="green_elevator_descend")
    hold(session, 1_000, reason="green_elevator_descent_settle")
    _require_room(session, 0x9AD9, "Green Brinstar Main Shaft landing")

def play_main_shaft_to_spore_spawn(session: ControllerSession) -> SporeSpawnEvidence:
    """Take the editor-planned route, defeat Spore Spawn, and exit naturally."""
    _require_room(session, 0x9AD9, "Main Shaft route entry")

    # Wave-4 04A guarded settle (x 118–126, 360f) timed out continuous at
    # x=128 y=680 pose=0 — too tight for elevator land variance. Restore fixed
    # settle; pure/docs may re-attempt with a wider band later.
    hold(session, 1_000, reason="main_shaft_entry_settle")
    for names in (
        ("RIGHT", "B"),
        ("LEFT", "B"),
        ("RIGHT", "B"),
        ("LEFT", "B"),
    ):
        hold(session, 60, *names, reason="main_shaft_descent")
    hold(session, 50, reason="main_shaft_descent_settle")
    for names in (
        ("RIGHT", "B"),
        ("LEFT", "B"),
        ("RIGHT", "B"),
        ("LEFT", "B"),
        ("RIGHT", "B"),
    ):
        hold(session, 80, *names, reason="main_shaft_dachora_level")
    hold(session, 30, reason="main_shaft_dachora_door_settle")
    hold(session, 1, "SELECT", reason="select_missiles")
    hold(session, 10, reason="select_missiles_settle")
    for _ in range(15):
        hold(session, 2, "X", reason="open_dachora_red_door")
        hold(session, 15, reason="open_dachora_red_door")
    hold(session, 100, "RIGHT", "B", reason="enter_dachora")
    hold(session, 250, reason="dachora_entry_settle")
    _require_room(session, 0x9CB3, "Dachora entry")

    hold(session, 350, "RIGHT", "A", "B", "X", reason="cross_dachora")
    hold(session, 2, "DOWN", reason="dachora_tunnel_morph")
    hold(session, 3, reason="dachora_tunnel_morph")
    hold(session, 2, "DOWN", reason="dachora_tunnel_morph")
    hold(session, 10, reason="dachora_tunnel_morph")
    for _ in range(15):
        hold(session, 45, "RIGHT", "X", reason="bomb_dachora_tunnel")
        hold(session, 15, "RIGHT", reason="bomb_dachora_tunnel")
    hold(session, 160, "RIGHT", "A", "B", "X", reason="exit_dachora")
    hold(session, 300, reason="big_pink_entry_settle")
    _require_room(session, 0x9D19, "Big Pink entry")

    hold(session, 2, "UP", reason="unmorph_big_pink")
    hold(session, 10, reason="unmorph_big_pink")
    hold(session, 180, "RIGHT", "A", "B", "X", reason="big_pink_climb")
    hold(session, 80, "LEFT", "A", "B", "X", reason="big_pink_climb")
    for names in _BIG_PINK_CLIMB:
        hold(session, 16, *names, reason="big_pink_map_guided_climb")
    if not (session.state.samus_y <= 150 and session.state.samus_x >= 780):
        raise RuntimeError(f"Big Pink climb missed upper-right door: {session.state}")
    hold(session, 100, "RIGHT", "B", "X", reason="big_pink_red_door_approach")
    for _ in range(15):
        hold(session, 2, "X", reason="open_kihunter_red_door")
        hold(session, 15, reason="open_kihunter_red_door")
    hold(session, 150, "RIGHT", "B", reason="enter_kihunter")
    hold(session, 300, reason="kihunter_entry_settle")
    _require_room(session, 0x9D9C, "Spore Kihunter entry")

    for index in range(8):
        direction = "RIGHT" if index % 2 == 0 else "LEFT"
        hold(
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
        hold(session, 2, *names, reason="aim_at_spore_kihunters")
        hold(
            session,
            8,
            *tuple(name for name in names if name != "X"),
            reason="aim_at_spore_kihunters",
        )
    hold(session, 300, reason="kihunter_clear_settle")
    if session.state.enemies_killed < 3:
        raise RuntimeError(f"Spore Kihunters did not clear naturally: {session.state}")
    hold(session, 80, "RIGHT", "B", reason="kihunter_boss_door_runway")
    hold(session, 100, "RIGHT", "A", "B", "X", reason="kihunter_boss_door_jump")
    hold(session, 10, reason="release_kihunter_jump")
    hold(session, 80, "RIGHT", "A", "B", "X", reason="align_spore_spawn_door")
    hold(session, 10, reason="release_kihunter_door_align")
    hold(session, 30, "LEFT", "B", reason="center_under_spore_spawn_door")
    hold(session, 60, reason="center_under_spore_spawn_door")
    for _ in range(15):
        hold(session, 2, "UP", "X", reason="open_spore_spawn_door")
        hold(session, 10, "UP", reason="open_spore_spawn_door")
    hold(session, 10, reason="release_spore_spawn_door_shot")
    hold(session, 120, "UP", "A", "B", reason="enter_spore_spawn")
    hold(session, 300, reason="spore_spawn_entry_settle")
    _require_room(session, ROOM_SPORE_SPAWN, "Spore Spawn entry")
    fight = play_spore_spawn_floor_bounce(session)

    hold(session, 600, reason="spore_spawn_death_settle")
    for names in _SPORE_EXIT_CLIMB:
        hold(session, 16, *names, reason="spore_exit_map_guided_climb")
    if not (session.state.samus_y <= 150 and session.state.samus_x >= 170):
        raise RuntimeError(f"Spore exit climb missed upper-right door: {session.state}")
    for _ in range(20):
        hold(session, 2, "RIGHT", "X", reason="open_spore_exit_door")
        hold(session, 8, "RIGHT", reason="open_spore_exit_door")
    hold(session, 300, reason="spore_spawn_exit_settle")
    _require_room(session, ROOM_SUPER, "Spore Spawn natural exit")

    return SporeSpawnEvidence(
        entry_frame=fight.entry_frame,
        activation_frame=fight.activation_frame,
        defeat_frame=fight.defeat_frame,
        exit_frame=session.frame,
        peak_hp=fight.peak_hp,
        observed_hp=fight.observed_hp,
        brinstar_boss_bits_before=fight.brinstar_boss_bits_before,
        brinstar_boss_bits_after=session.state.boss_bits[1],
        vulnerable_spritemaps=fight.vulnerable_spritemaps,
    )


def play_post_torizo_to_spore_spawn(
    session: ControllerSession,
) -> SporeSpawnEvidence:
    """Run the complete checked post-Torizo controller slice."""
    play_parlor_to_main_shaft(session)
    return play_main_shaft_to_spore_spawn(session)
