"""Controller-only post-Torizo route through Spore Spawn (KPDR continuous).

The room sequence is pre-calculated from the Super Metroid editor export.  This
module owns movement and combat only: it reads typed state and emits ordinary
12-button controller actions.  It never loads emulator state or writes RAM.

Parlor left (Alcatraz) climb after Bomb Torizo reuses the shared wall-jump
skill from :mod:`controller_common` (same pulse shape as Bubble Phase D).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from super_metroid.ram import GameplayPhase
from super_metroid.routes.controller_common import (
    WallJumpTiming,
    consecutive_walljumps,
    hold_until,
    require_room,
    settle_hold,
)
from super_metroid.routes.runtime import ControllerSession, hold

# Post-Torizo Alcatraz chimney (pre-Hi-Jump): open-loop into-wall + A pulses.
# Frame-matched to the legacy RIGHT×6 / LEFT×2 holds; expressed as named
# WallJumpTiming so the skill is visible on the continuous spine (second
# consumer after Bubble R15 double WJ).
_PARLOR_CHIMNEY_RIGHT = WallJumpTiming(
    into="RIGHT",
    flip="RIGHT",
    into_frames=30,
    amid_frames=0,
    flip_frames=0,
    delay_into_frames=0,
)
_PARLOR_CHIMNEY_LEFT = WallJumpTiming(
    into="LEFT",
    flip="LEFT",
    into_frames=30,
    amid_frames=0,
    flip_frames=0,
    delay_into_frames=0,
)
# Six right-wall pulses then two left-wall pulses; 12f settle between.
_PARLOR_CHIMNEY_WJ: tuple[WallJumpTiming, ...] = (
    *(_PARLOR_CHIMNEY_RIGHT for _ in range(6)),
    *(_PARLOR_CHIMNEY_LEFT for _ in range(2)),
)
_PARLOR_CHIMNEY_GAP = 12

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

# Mouth open/close transition plus fully-open hold spritemaps. The old set
# (EEAF/EEC1/EED3/EEE5 only) missed the long open hold windows EF3D/EF4F/EF61
# where multi-missile damage is available.
_VULNERABLE_SPRITEMAPS = frozenset(
    {
        0xEE79,
        0xEE8B,
        0xEE9D,
        0xEEAF,
        0xEEC1,
        0xEED3,
        0xEEE5,
        0xEF3D,
        0xEF4F,
        0xEF61,
    }
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
    if session.state.samus_x > 956:
        alignment_frames = 10 if session.state.pose == 2 else 15
        hold(
            session,
            alignment_frames,
            "LEFT",
            reason="post_torizo_parlor_alignment",
        )
        hold(session, 10, reason="post_torizo_parlor_alignment")
    if not (
        948 <= session.state.samus_x <= 954
        and session.state.samus_y == 651
        and session.state.pose == 2
    ):
        raise RuntimeError(f"post-Torizo Parlor alignment missed: {session.state}")

    for _ in range(2):
        hold(session, 20, "LEFT", "A", "B", "X", reason="parlor_left_traverse")
        hold(session, 12, "LEFT", "B", "X", reason="parlor_left_traverse")
    hold(session, 50, reason="parlor_left_traverse_settle")
    # Alcatraz left climb: shared consecutive wall-jump skill (visible WJ).
    consecutive_walljumps(
        session,
        _PARLOR_CHIMNEY_WJ,
        reason="parlor_chimney_wj",
        gap_frames=_PARLOR_CHIMNEY_GAP,
    )
    # Final pulse had no trailing gap in the old open-loop (gap only between);
    # add one post-chain settle matching the last idle of the old 8×(30+12).
    settle_hold(session, _PARLOR_CHIMNEY_GAP, reason="parlor_chimney_wj_tail")
    hold(session, 100, reason="parlor_chimney_settle")
    hold(session, 30, "LEFT", "A", reason="parlor_upper_platforms")
    hold(session, 30, "LEFT", reason="parlor_upper_platforms")
    hold(session, 60, reason="parlor_upper_platforms")
    hold(session, 40, "LEFT", "A", reason="parlor_upper_platforms")
    hold(session, 100, reason="parlor_upper_platforms")
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
        hold(session, frames, *names, reason="parlor_upper_platforms")
    hold(session, 2, "DOWN", reason="parlor_bomb_tunnel_morph")
    hold(session, 3, reason="parlor_bomb_tunnel_morph")
    hold(session, 2, "DOWN", reason="parlor_bomb_tunnel_morph")
    hold(session, 10, reason="parlor_bomb_tunnel_morph")
    for _ in range(10):
        hold(session, 45, "RIGHT", "X", reason="parlor_bomb_tunnel")
        hold(session, 15, "RIGHT", reason="parlor_bomb_tunnel")
    hold(session, 100, reason="parlor_bomb_tunnel_settle")
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
    _require_room(session, 0x990D, "Terminator traversal")

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
    _require_room(session, 0x9DC7, "Spore Spawn entry")
    if session.state.enemy0_hp < 960:
        raise RuntimeError(f"Spore Spawn did not activate at 960 HP: {session.state}")

    entry_frame = session.frame
    activation_frame = session.frame
    peak_hp = session.state.enemy0_hp
    observed_hp = {session.state.enemy0_hp}
    boss_bits_before = session.state.boss_bits[1]
    seen_spritemaps: set[int] = set()
    # Floor bounce + aim-up missiles during open windows. Unlimited energy
    # means survival is free; damage requires airborne proximity to the core
    # (shell blocks floor shots). Continuous power-on fight ~5.2k frames
    # (~86s) vs the prior ~23k (~6+ min) single-hit-per-window loop.
    jump_direction = "RIGHT"
    jump_hold = 0
    for index in range(12_000):
        state = session.state
        peak_hp = max(peak_hp, state.enemy0_hp)
        observed_hp.add(state.enemy0_hp)
        mouth_open = state.enemy0_spritemap in _VULNERABLE_SPRITEMAPS
        if mouth_open:
            seen_spritemaps.add(state.enemy0_spritemap)
        # Bounce across the floor so open windows still cross under the core.
        if state.samus_x <= 65:
            jump_direction = "RIGHT"
        elif state.samus_x >= 191:
            jump_direction = "LEFT"
        if state.samus_y >= 710 and jump_hold == 0:
            # Slightly longer hold while open keeps height for multi-missile
            # windows (missile cadence ~every other frame while open).
            jump_hold = 52 if mouth_open else 44
        hold_jump = jump_hold > 0
        jump_hold = max(0, jump_hold - 1)
        fire = mouth_open and index % 2 == 0
        aim_direction = "LEFT" if state.enemy0_x < state.samus_x else "RIGHT"
        if state.samus_y >= 710:
            # Launch jump; still fire if the mouth opens during takeoff.
            names_list = [jump_direction, "A"]
            if hold_jump:
                names_list.append("B")
            if fire:
                names_list.extend(("UP", "X"))
            names = tuple(names_list)
        else:
            # Airborne: hold UP to unspin so missiles can fire, face the core.
            names_list = [aim_direction, "UP"]
            if hold_jump:
                names_list.extend(("A", "B"))
            if fire:
                names_list.append("X")
            names = tuple(names_list)
        hold(session, 1, *names, reason="fight_spore_spawn")
        if session.state.enemy0_hp == 0:
            observed_hp.add(0)
            break
    else:
        raise TimeoutError(f"Spore Spawn HP never reached zero: {session.state}")
    defeat_frame = session.frame

    hold(session, 600, reason="spore_spawn_death_settle")
    for names in _SPORE_EXIT_CLIMB:
        hold(session, 16, *names, reason="spore_exit_map_guided_climb")
    if not (session.state.samus_y <= 150 and session.state.samus_x >= 170):
        raise RuntimeError(f"Spore exit climb missed upper-right door: {session.state}")
    for _ in range(20):
        hold(session, 2, "RIGHT", "X", reason="open_spore_exit_door")
        hold(session, 8, "RIGHT", reason="open_spore_exit_door")
    hold(session, 300, reason="spore_spawn_exit_settle")
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
