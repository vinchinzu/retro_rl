"""Build the disclosed recon fixture for isolated Level 7 interior probing.

rr-8t4.2 (L7-B).  Unlike the L8 builder this one does NOT start from a
pre-settled save-state: it **walks the fixture-live** ``0x79 -> 0x69 -> 0x6A
-> 0x6B`` chain from the ``Level7Entrance`` pin (``EntryNorthDoorController``
-> ``Room69EastController`` -> ``Room6AEastController``, all 2/2 green),
clears the six ``0x6B`` goriya ``0x05`` with the shared goriya micro, then
discloses only the resources onward recon needs:

  * ``ADDR_FOOD`` (``$065D``) 0 -> 1   -- the Hungry Goriya blocker
  * ``ADDR_BOMBS`` (``$0658``) -> min(8, existing max-bomb capacity) -- wall skips
  * ``ADDR_KEYS``  (``$066E``) -> 4 -- the pre-fifth-lock recon keys

The builder deliberately does not write Candle, Whistle, Triforce, doors,
room/screen, health, or heart capacity.  ``ADDR_MAX_BOMBS`` is read, never
written.  The traverse runs under the standard ``UnlimitedHealthAssist``
(same as the 2/2 green room walks); that is a traversal aid, disclosed in the
provenance notes, not a fixture write -- ``deaths`` must be 0 and
``progression_writes`` / ``capacity_writes`` must be 0.

This is not a natural route state and is never route eligible.

    QT_QPA_PLATFORM=offscreen uv run python \\
        nes/zelda_i/scratch/build_level7_interior_recon_fixture.py
"""

from __future__ import annotations

from typing import Any

from retro_harness.env import make_env, reset_obs, save_state, state_path
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless
from zelda_i.combat import nearest_enemy
from zelda_i.dungeon.behaviors import EnemyKind, engagement_hint
from zelda_i.dungeon.trace import compact_snapshot, write_state_provenance
from zelda_i.level7.path import (
    EntryNorthDoorController,
    Room6AEastController,
    Room69EastController,
    live_goriyas,
)
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import (
    ADDR_BOMBS,
    ADDR_CANDLE,
    ADDR_FOOD,
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_MAX_BOMBS,
    ADDR_MODE,
    ADDR_SCREEN,
    ADDR_TRIFORCE,
    ADDR_WHISTLE,
    PLAY_MODE,
    read_snapshot,
    read_u8,
)
from zelda_i.runner import make_assist

BEAD = "rr-8t4.2"
SOURCE_STATE = "Level7Entrance"
FIXTURE_NAME = "Level7InteriorReconFixture"

LEVEL7 = 7
ENTRY_ROOM = 0x79
ROOM_69 = 0x69
ROOM_6A = 0x6A
ROOM_6B = 0x6B
TF_BEFORE_LEVEL7 = 0x3F
PREFERRED_BOMBS = 8
RECON_KEYS = 4
SETTLE_FRAMES = 30
FIGHT_MAX_FRAMES = 4000


def _assign(env: Any, address: int, value: int) -> None:
    env.unwrapped.data.memory.assign(int(address), "|u1", int(value) & 0xFF)


def _drive_chain(env: Any, assist: Any) -> tuple[bool, int]:
    chain = [
        EntryNorthDoorController(),
        Room69EastController(),
        Room6AEastController(),
    ]
    idx = 0
    ctl = chain[idx]
    budget = sum(c.max_frames for c in chain)
    for frame in range(budget):
        snap = read_snapshot(env.get_ram())
        env.step(ctl.step(snap).action)
        if assist is not None:
            assist.apply_env(env, frame=frame)
        s = read_snapshot(env.get_ram())
        if s.screen == ROOM_6B and s.mode == PLAY_MODE and not s.transitioning:
            return True, frame
        if ctl.success and idx < len(chain) - 1:
            idx += 1
            ctl = chain[idx]
        elif ctl.failed:
            return False, frame
    return False, -1


def _clear_goriyas(env: Any, assist: Any, base: int) -> tuple[int, str]:
    saw = False
    for i in range(FIGHT_MAX_FRAMES):
        s = read_snapshot(env.get_ram())
        if s.screen != ROOM_6B:
            return base + i, f"left_to_0x{s.screen:02x}"
        live = live_goriyas(s)
        if live:
            saw = True
        if not live:
            if not saw or i < 90:
                env.step(nes_action("RIGHT"))
                if assist is not None:
                    assist.apply_env(env, frame=base + i)
                continue
            return base + i, "clear"
        tgt = nearest_enemy(s.link_x, s.link_y, live)
        hint = engagement_hint(EnemyKind.GORIYA, s, tgt)
        btn = hint.face
        act = nes_action(btn, "A") if (i % 8) < 4 else nes_action(btn)
        env.step(act)
        if assist is not None:
            assist.apply_env(env, frame=base + i)
    return base + FIGHT_MAX_FRAMES, "timeout"


def main() -> int:
    configure_headless()
    assist = make_assist(True)
    env = make_env(GAME, SOURCE_STATE, GAME_DIR, render_mode="rgb_array")
    try:
        reset_obs(env)
        before = read_snapshot(env.get_ram())
        print(f"source ({SOURCE_STATE}) glance: {compact_snapshot(before)}")
        if not (
            before.level == LEVEL7
            and before.mode == PLAY_MODE
            and before.screen == ENTRY_ROOM
            and int(read_u8(env.get_ram(), ADDR_FOOD)) == 0
        ):
            raise SystemExit("source pin mismatch: expected L7 play 0x79 Food 0")

        ok, reached_frame = _drive_chain(env, assist)
        mid = read_snapshot(env.get_ram())
        print(
            f"chain reached_6b={ok} frame={reached_frame} "
            f"screen=0x{mid.screen:02x} xy=({mid.link_x},{mid.link_y}) "
            f"goriya={[hex(int(o.type_id)) for o in live_goriyas(mid)]}"
        )
        if not ok:
            raise SystemExit("chain did not reach 0x6B")

        fight_frame, status = _clear_goriyas(env, assist, reached_frame)
        cleared = read_snapshot(env.get_ram())
        print(
            f"goriya clear: {status} at frame {fight_frame} "
            f"screen=0x{cleared.screen:02x} xy=({cleared.link_x},{cleared.link_y})"
        )
        if status != "clear" or cleared.screen != ROOM_6B:
            raise SystemExit(f"goriya clear failed: {status}")

        for _ in range(SETTLE_FRAMES):
            env.step(nes_idle_action())
            if assist is not None:
                assist.apply_env(env, frame=fight_frame)

        ram = env.get_ram()
        pre = read_snapshot(ram)
        if not (
            pre.level == LEVEL7
            and pre.mode == PLAY_MODE
            and pre.screen == ROOM_6B
            and pre.triforce == 0
        ):
            raise SystemExit("pre-poke pin mismatch (expected L7 play 0x6B TF 0)")

        max_bombs = int(read_u8(ram, ADDR_MAX_BOMBS))
        bombs = min(PREFERRED_BOMBS, max_bombs) if max_bombs > 0 else PREFERRED_BOMBS
        write_specs = (
            ("food_for_hungry_goriya", ADDR_FOOD, 1),
            ("bombs_for_wall_skips", ADDR_BOMBS, bombs),
            ("keys_for_pre_fifth_lock", ADDR_KEYS, RECON_KEYS),
        )
        fixture_writes: list[dict[str, Any]] = []
        for name, address, target in write_specs:
            value_before = int(read_u8(env.get_ram(), address))
            _assign(env, address, target)
            value_after = int(read_u8(env.get_ram(), address))
            if value_after != int(target):
                raise SystemExit(f"write did not take: {name}")
            fixture_writes.append(
                {
                    "name": name,
                    "address": int(address),
                    "address_hex": f"0x{address:04X}",
                    "before": value_before,
                    "from": value_before,
                    "to": value_after,
                }
            )

        for _ in range(SETTLE_FRAMES):
            env.step(nes_idle_action())
            if assist is not None:
                assist.apply_env(env, frame=fight_frame)

        ram = env.get_ram()
        after = read_snapshot(ram)
        print(f"fixture glance: {compact_snapshot(after)}")
        if not (
            after.level == LEVEL7
            and after.mode == PLAY_MODE
            and after.screen == ROOM_6B
            and after.triforce == 0
            and int(read_u8(ram, ADDR_FOOD)) == 1
            and int(read_u8(ram, ADDR_CANDLE)) == 0
            and int(read_u8(ram, ADDR_WHISTLE)) == 1
        ):
            raise SystemExit("fixture pin mismatch after resource writes")

        telem = assist.telemetry
        if int(telem.deaths) != 0:
            raise SystemExit(f"traverse recorded deaths={telem.deaths}")
        if int(telem.progression_writes) or int(telem.capacity_writes):
            raise SystemExit("traverse recorded progression/capacity writes")

        path = save_state(env, GAME_DIR, GAME, FIXTURE_NAME)
        source_path = state_path(GAME_DIR, GAME, SOURCE_STATE)
        inventory = {
            "bombs": int(read_u8(ram, ADDR_BOMBS)),
            "max_bombs": max_bombs,
            "keys": int(read_u8(ram, ADDR_KEYS)),
            "food": int(read_u8(ram, ADDR_FOOD)),
            "candle": int(read_u8(ram, ADDR_CANDLE)),
            "whistle": int(read_u8(ram, ADDR_WHISTLE)),
        }
        result = {
            "ok": True,
            "source_state": SOURCE_STATE,
            "fixture_state": FIXTURE_NAME,
            "state": compact_snapshot(after),
            "inventory": inventory,
            "fixture_writes": fixture_writes,
            "chain_reached_6b_frame": reached_frame,
            "goriya_clear_frame": fight_frame,
            "assist": assist.report(),
            "protected_unchanged": {
                "level": int(read_u8(ram, ADDR_LEVEL)) == LEVEL7,
                "mode": int(read_u8(ram, ADDR_MODE)) == PLAY_MODE,
                "screen": int(read_u8(ram, ADDR_SCREEN)) == ROOM_6B,
                "triforce": int(read_u8(ram, ADDR_TRIFORCE)) == 0,
                "candle": int(read_u8(ram, ADDR_CANDLE)) == 0,
                "whistle": int(read_u8(ram, ADDR_WHISTLE)) == 1,
            },
        }
        write_state_provenance(
            path,
            source_state_path=source_path if source_path.exists() else None,
            request={
                "bead": BEAD,
                "phase": "level7_interior_recon_resources",
                "track": "recon_fixture",
                "route_eligible": False,
                "fixture_only": True,
                "natural_entry": False,
                "fixture_writes": fixture_writes,
                "notes": [
                    "WALKED the fixture-live 0x79->0x69->0x6A->0x6B chain from "
                    "Level7Entrance (EntryNorthDoorController, Room69EastController, "
                    "Room6AEastController); no set_state / teleport.",
                    "Six 0x6B goriya 0x05 cleared with the shared goriya micro.",
                    "Traverse ran under the standard UnlimitedHealthAssist "
                    "(traversal aid, same as the 2/2 green room walks); deaths=0, "
                    "progression_writes=0, capacity_writes=0. Not a fixture write.",
                    "Bomb target is min(preferred 8, existing max-bomb capacity); "
                    "ADDR_MAX_BOMBS is read, never written.",
                    "Keys are disclosed pre-fifth-lock recon resource.",
                    "No Candle, Whistle, Triforce, room/screen, door, health, or "
                    "heart-capacity writes.",
                ],
            },
            selected_trial=result,
            natural_entry=False,
        )
        print(f"saved {path}")
        print(f"saved {path.with_suffix('.provenance.json')}")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
