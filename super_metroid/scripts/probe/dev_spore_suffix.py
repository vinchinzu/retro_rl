"""Development-only driver for the post-Torizo Spore Spawn suffix.

This intentionally starts from editor save states so room navigation can be
developed quickly.  Its output is never acceptance evidence; accepted runs
must compose the resulting raw controller policy after the power-on prefix.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from retro_harness.actions import buttons, idle_action  # noqa: E402
from retro_harness.env import make_env  # noqa: E402
from super_metroid.assist import (  # noqa: E402
    UnlimitedAmmoAssist,
    UnlimitedResourcesAssist,
)
from super_metroid.paths import GAME, GAME_DIR  # noqa: E402
from super_metroid.ram import SuperMetroidState, parse_state  # noqa: E402
from super_metroid.routes.spore_spawn_controller import (  # noqa: E402
    play_parlor_to_main_shaft,
    play_post_torizo_to_spore_spawn,
)


DEBUG_DIR = _REPO_ROOT / "super_metroid/debug/spore/suffix"
MAIN_SHAFT_STATE = "Green Brinstar Main Shaft [from Green Brinstar Elevator Room]"
PARLOR_STATE = "Parlor and Alcatraz [from Flyway]"
LJ = ("LEFT", "A", "B", "X")
RJ = ("RIGHT", "A", "B", "X")
RR = ("RIGHT", "B", "X")
LR = ("LEFT", "B", "X")
J = ("A", "B", "X")
_BIG_PINK_SEARCH_SEED = (
    LJ, RJ, RJ, LJ, RJ, RJ, LJ, RJ, RJ, RR, RJ, (), (), LJ, RJ, RJ, RJ,
    RJ, RJ, (), RJ, RJ, RJ, RJ, (), J, RJ, RJ, RJ,
    RJ, LJ, RR, J, LJ, LJ, LJ,
    LJ, LJ, (), LJ,
    LJ, J, LJ, LJ, RJ, LJ, LJ, RJ, LJ, LJ, LJ, RJ, LJ, LJ, RJ, LJ,
    LJ, J, RJ, RR, RR, RJ, RJ, RJ, RJ, LJ,
)
_SPORE_EXIT_SEARCH_SEED = (
    LR, LR, RJ, LJ, RR, LJ, LJ, (),
    RJ, RJ, RJ, RJ, RJ, LJ, LJ, RJ,
    LR, LJ, LJ, LJ, RJ, LJ, LR, LR,
    RR, RJ, RJ, RR, RR, LR, LJ, LJ,
    LJ, RJ, LJ, J, RJ, LJ, LR, RJ,
    RJ, RJ, RJ, RJ, J, RJ, (), LJ,
    (), J, LJ,
)


@dataclass
class DevSession:
    env: object
    assist: UnlimitedAmmoAssist
    observation: np.ndarray
    frame: int = 0

    @property
    def state(self) -> SuperMetroidState:
        return parse_state(self.env.get_ram(), frame=self.frame)

    def step(self, action: np.ndarray, reason: str) -> SuperMetroidState:
        del reason
        self.observation, _, _, _, _ = self.env.step(action)
        self.frame += 1
        state = self.state
        self.assist.apply(self.env.data, state)
        return state

    def hold(self, frames: int, *names: str, reason: str) -> SuperMetroidState:
        action = buttons(*names) if names else idle_action()
        state = self.state
        for _ in range(frames):
            state = self.step(action, reason)
        return state

    def log(self, label: str) -> None:
        state = self.state
        ram = self.env.get_ram()
        enemy_words = {
            "spritemap": 0x0F8E,
            "timer": 0x0F90,
            "instruction": 0x0F92,
            "instruction_timer": 0x0F94,
            "flash": 0x0F9C,
            "invincibility": 0x0FA0,
            "ai": 0x0FA8,
        }
        enemy_detail = {}
        for name, address in enemy_words.items():
            enemy_detail[name] = (
                int(ram[address]) | int(ram[address + 1]) << 8
            )
        enemies = []
        for slot in range(8):
            offset = slot * 0x40
            x = int(ram[0x0F7A + offset]) | int(ram[0x0F7B + offset]) << 8
            y = int(ram[0x0F7E + offset]) | int(ram[0x0F7F + offset]) << 8
            hp = int(ram[0x0F8C + offset]) | int(ram[0x0F8D + offset]) << 8
            if 0 < hp < 0x8000:
                enemies.append(f"{slot}:{x},{y}/{hp}")
        print(
            f"{label:<28} frame={self.frame:<6} room=0x{state.room_id:04X} "
            f"phase={state.phase.value:<18} x={state.samus_x:<4} "
            f"y={state.samus_y:<4} hp={state.health:<3} "
            f"ammo={state.missiles}/{state.max_missiles} "
            f"selected={state.selected_item:<2} "
            f"pose={state.pose:<3} "
            f"enemies={state.enemies_killed}/{state.num_enemies} "
            f"alive=[{' '.join(enemies)}] enemy0_detail={enemy_detail}"
        )

    def snapshot(self, label: str) -> None:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        output = DEBUG_DIR / f"{label}.png"
        cv2.imwrite(str(output), cv2.cvtColor(self.observation, cv2.COLOR_RGB2BGR))
        print(f"snapshot: {output}")


@dataclass(frozen=True)
class ProbeNode:
    state: bytes
    actions: tuple[tuple[str, ...], ...]
    samus_x: int
    samus_y: int
    pose: int
    velocity_y: int


def beam_search_target(
    session: DevSession,
    *,
    target_x: int,
    target_y: int,
    room_id: int = 0x9D19,
    depth: int = 24,
    width: int = 24,
    macro_frames: int = 16,
) -> tuple[tuple[str, ...], ...]:
    """Development search for a low-y controller sequence from the live state."""
    action_names = (
        ("LEFT", "A", "B", "X"),
        ("RIGHT", "A", "B", "X"),
        ("LEFT", "B", "X"),
        ("RIGHT", "B", "X"),
        ("A", "B", "X"),
        (),
    )
    start = session.state
    frontier = [
        ProbeNode(
            state=session.env.em.get_state(),
            actions=(),
            samus_x=start.samus_x,
            samus_y=start.samus_y,
            pose=start.pose,
            velocity_y=start.velocity_y,
        )
    ]
    def score(node: ProbeNode) -> tuple[int, int, int]:
        return (
            abs(node.samus_x - target_x) + 2 * abs(node.samus_y - target_y),
            abs(node.samus_y - target_y),
            abs(node.velocity_y),
        )

    best = frontier[0]
    for layer in range(depth):
        children: list[ProbeNode] = []
        for parent in frontier:
            for names in action_names:
                session.env.em.set_state(parent.state)
                session.hold(
                    macro_frames,
                    *names,
                    reason="development_beam_search",
                )
                state = session.state
                if state.room_id != room_id or state.dead:
                    continue
                children.append(
                    ProbeNode(
                        state=session.env.em.get_state(),
                        actions=(*parent.actions, names),
                        samus_x=state.samus_x,
                        samus_y=state.samus_y,
                        pose=state.pose,
                        velocity_y=state.velocity_y,
                    )
                )
        by_cell: dict[tuple[int, int, int, int], ProbeNode] = {}
        for child in children:
            key = (
                child.samus_x // 12,
                child.samus_y // 12,
                child.pose,
                child.velocity_y // 128,
            )
            incumbent = by_cell.get(key)
            if incumbent is None or score(child) < score(incumbent):
                by_cell[key] = child
        frontier = sorted(by_cell.values(), key=score)[:width]
        if not frontier:
            break
        if score(frontier[0]) < score(best):
            best = frontier[0]
        print(
            f"beam target=({target_x},{target_y}) layer={layer + 1:<2} "
            f"best=({best.samus_x},{best.samus_y}) "
            f"score={score(best)[0]}"
        )
        if score(best)[0] <= 24:
            break
    return best.actions


def reach_big_pink(session: DevSession) -> None:
    """Replay the discovered Main Shaft and Dachora controller path."""
    session.hold(1_000, reason="main_shaft_entry_settle")
    session.log("main shaft settled")

    for names in (
        ("RIGHT", "B"),
        ("LEFT", "B"),
        ("RIGHT", "B"),
        ("LEFT", "B"),
    ):
        session.hold(60, *names, reason="main_shaft_descent")
    session.hold(50, reason="main_shaft_descent_settle")
    for names in (
        ("RIGHT", "B"),
        ("LEFT", "B"),
        ("RIGHT", "B"),
        ("LEFT", "B"),
        ("RIGHT", "B"),
    ):
        session.hold(80, *names, reason="main_shaft_dachora_level")
    session.hold(30, reason="main_shaft_dachora_door_settle")
    session.log("main shaft door level")

    session.hold(1, "SELECT", reason="select_missiles")
    session.hold(10, reason="select_missiles_settle")
    for _ in range(15):
        session.hold(2, "X", reason="open_dachora_red_door")
        session.hold(15, reason="open_dachora_red_door")
    session.hold(100, "RIGHT", "B", reason="enter_dachora")
    session.hold(250, reason="dachora_entry_settle")
    session.log("dachora settled")
    session.snapshot("dachora_entry")

    session.hold(350, "RIGHT", "A", "B", "X", reason="cross_dachora")
    session.log("dachora right wall")
    session.hold(2, "DOWN", reason="morph")
    session.hold(3, reason="morph")
    session.hold(2, "DOWN", reason="morph")
    session.hold(10, reason="morph")
    for _ in range(15):
        session.hold(45, "RIGHT", "X", reason="bomb_dachora_tunnel")
        session.hold(15, "RIGHT", reason="bomb_dachora_tunnel")
    session.log("dachora tunnel clear")
    session.hold(160, "RIGHT", "A", "B", "X", reason="exit_dachora")
    session.hold(300, reason="big_pink_entry_settle")
    session.log("big pink settled")
    session.snapshot("big_pink_entry")


def climb_big_pink(session: DevSession, *, search: bool) -> None:
    """Probe the central shaft from the Dachora entry toward the top-right door."""
    session.hold(2, "UP", reason="unmorph_big_pink")
    session.hold(10, reason="unmorph_big_pink")
    session.log("big pink unmorphed")
    for index, direction in enumerate(("RIGHT", "LEFT")):
        session.hold(
            180 if index == 0 else 80,
            direction,
            "A",
            "B",
            "X",
            reason="big_pink_climb_upper",
        )
        session.log(f"big pink climb {index + 1}")
    if search:
        all_actions: list[tuple[str, ...]] = list(_BIG_PINK_SEARCH_SEED)
        for names in _BIG_PINK_SEARCH_SEED:
            session.hold(16, *names, reason="big_pink_beam_seed")
        session.log("big pink beam seed")
        print(f"beam combined actions: {tuple(all_actions)}")
        session.hold(100, "RIGHT", "B", "X", reason="big_pink_red_door_approach")
        session.log("big pink red door")
        for _ in range(15):
            session.hold(2, "X", reason="open_kihunter_red_door")
            session.hold(15, reason="open_kihunter_red_door")
        session.hold(150, "RIGHT", "B", reason="enter_kihunter")
        session.hold(300, reason="kihunter_entry_settle")
        session.log("kihunter settled")
        session.snapshot("kihunter_entry")
        for index in range(8):
            direction = "RIGHT" if index % 2 == 0 else "LEFT"
            session.hold(
                180,
                direction,
                "A",
                "B",
                "X",
                reason="clear_spore_kihunters",
            )
            session.log(f"kihunter pass {index + 1}")
        for index in range(240):
            names = (
                ("UP", "X"),
                ("LEFT", "UP", "X"),
                ("RIGHT", "UP", "X"),
                ("LEFT", "X"),
                ("RIGHT", "X"),
            )[index % 5]
            session.hold(2, *names, reason="aim_at_spore_kihunters")
            session.hold(
                8,
                *tuple(name for name in names if name != "X"),
                reason="aim_at_spore_kihunters",
            )
            if index % 30 == 29:
                session.log(f"kihunter aim {index + 1}")
        session.hold(300, reason="kihunter_clear_settle")
        session.log("kihunter clear settled")
        session.snapshot("kihunter_clear")
        session.hold(80, "RIGHT", "B", reason="kihunter_boss_door_runway")
        session.log("kihunter runway")
        session.hold(
            100,
            "RIGHT",
            "A",
            "B",
            "X",
            reason="kihunter_boss_door_jump",
        )
        session.log("kihunter door jump")
        session.hold(10, reason="release_kihunter_jump")
        session.hold(
            80,
            "RIGHT",
            "A",
            "B",
            "X",
            reason="align_spore_spawn_door",
        )
        session.log("kihunter door align")
        session.hold(10, reason="release_kihunter_door_align")
        session.hold(30, "LEFT", "B", reason="center_under_spore_spawn_door")
        session.hold(60, reason="center_under_spore_spawn_door")
        session.log("kihunter under door")
        for _ in range(15):
            session.hold(2, "UP", "X", reason="open_spore_spawn_door")
            session.hold(10, "UP", reason="open_spore_spawn_door")
        session.hold(10, reason="release_spore_spawn_door_shot")
        session.hold(120, "UP", "A", "B", reason="enter_spore_spawn")
        session.hold(300, reason="spore_spawn_entry_settle")
        session.log("spore spawn settled")
        session.snapshot("spore_spawn_entry")
        vulnerable_spritemaps = {0xEEAF, 0xEEC1, 0xEED3, 0xEEE5}
        captured_spritemaps: set[int] = set()
        jump_direction = "RIGHT"
        jump_hold = 0
        for index in range(30_000):
            ram = session.env.get_ram()
            spritemap = int(ram[0x0F8E]) | int(ram[0x0F8F]) << 8
            if (
                spritemap in vulnerable_spritemaps
                and spritemap not in captured_spritemaps
            ):
                captured_spritemaps.add(spritemap)
                session.snapshot(f"spore_spritemap_{spritemap:04x}")
            boss_x = session.state.enemy0_x
            if session.state.samus_x <= 65:
                jump_direction = "RIGHT"
            elif session.state.samus_x >= 191:
                jump_direction = "LEFT"
            if session.state.samus_y >= 710 and jump_hold == 0:
                jump_hold = 36
            hold_jump = jump_hold > 0
            jump_hold = max(0, jump_hold - 1)
            if session.state.samus_y >= 710:
                names = (
                    (jump_direction, "A", "B")
                    if hold_jump
                    else (jump_direction, "A")
                )
            else:
                aim_direction = (
                    "LEFT"
                    if boss_x < session.state.samus_x
                    else "RIGHT"
                )
                fire = spritemap in vulnerable_spritemaps and index % 4 == 0
                names_list = [aim_direction, "UP"]
                if hold_jump:
                    names_list.extend(("A", "B"))
                if fire:
                    names_list.append("X")
                names = tuple(names_list)
            session.hold(1, *names, reason="fight_spore_spawn")
            if index % 1_200 == 1_199:
                session.log(f"spore fight {index + 1}")
            if session.state.enemy0_hp == 0:
                break
        session.hold(600, reason="spore_spawn_death_settle")
        session.log("spore spawn death settle")
        session.snapshot("spore_spawn_death")
        if search:
            for names in _SPORE_EXIT_SEARCH_SEED:
                session.hold(16, *names, reason="spore_exit_beam_seed")
            session.log("spore exit beam seed")
            for _ in range(20):
                session.hold(2, "RIGHT", "X", reason="open_spore_exit_door")
                session.hold(8, "RIGHT", reason="open_spore_exit_door")
            session.hold(300, reason="spore_spawn_exit_settle")
            session.log("spore spawn exit settled")
            session.snapshot("spore_spawn_exit")
            return
        for index in range(20):
            direction = "LEFT" if index % 2 == 0 else "RIGHT"
            session.hold(
                45,
                direction,
                "A",
                "B",
                "X",
                reason="climb_out_of_spore_spawn",
            )
            session.hold(40, reason="release_spore_spawn_climb_jump")
            session.log(f"spore exit climb {index + 1}")
            if session.state.room_id != 0x9DC7:
                break
        session.hold(300, reason="spore_spawn_exit_settle")
        session.log("spore spawn exit settled")
        session.snapshot("spore_spawn_exit")
        return
    for index in range(24):
        direction = "LEFT" if index % 2 == 0 else "RIGHT"
        session.hold(
            16,
            direction,
            "A",
            "B",
            "X",
            reason="big_pink_wall_jump",
        )
        if index % 4 == 3:
            session.log(f"big pink wall jump {index + 1}")
    session.hold(100, reason="big_pink_climb_settle")
    session.log("big pink climb settled")
    session.snapshot("big_pink_climb_probe")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default=MAIN_SHAFT_STATE)
    parser.add_argument(
        "--game-dir",
        type=Path,
        default=GAME_DIR,
        help="directory containing custom_integrations/ (default: super_metroid)",
    )
    parser.add_argument("--beam", action="store_true")
    parser.add_argument("--parlor-to-main", action="store_true")
    parser.add_argument("--parlor-to-spore", action="store_true")
    parser.add_argument("--simulate-power-on-handoff", action="store_true")
    parser.add_argument("--simulate-left-handoff", action="store_true")
    args = parser.parse_args()

    env = make_env(GAME, args.state, args.game_dir, render_mode="rgb_array")
    try:
        observation, _ = env.reset()
        session = DevSession(env, UnlimitedResourcesAssist(), observation)
        session.log("reset")
        if args.simulate_power_on_handoff:
            session.hold(15, "RIGHT", reason="simulate_power_on_handoff")
            session.hold(10, reason="simulate_power_on_handoff")
            session.log("simulated power-on handoff")
        if args.simulate_left_handoff:
            session.hold(2, "LEFT", reason="simulate_left_handoff")
            session.hold(10, reason="simulate_left_handoff")
            session.log("simulated left handoff")
        if args.parlor_to_spore:
            evidence = play_post_torizo_to_spore_spawn(session)
            session.log("production suffix complete")
            print(f"spore evidence: {evidence.to_dict()}")
            session.snapshot("production_suffix_complete")
            return
        if args.parlor_to_main:
            play_parlor_to_main_shaft(session)
            session.log("main shaft from parlor")
            session.snapshot("main_shaft_from_parlor")
            return
        reach_big_pink(session)
        climb_big_pink(session, search=args.beam)
    finally:
        env.close()


if __name__ == "__main__":
    main()
