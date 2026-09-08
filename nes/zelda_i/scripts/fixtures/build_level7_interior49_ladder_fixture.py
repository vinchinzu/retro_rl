"""Build Level7Interior49LadderReconFixture from Level7Interior49ReconFixture.

rr-8t4.2 (L7-B).  Disclosed recon-only ``ADDR_LADDER`` ``$0663`` 0->1 so the
0x49 water moat (tile 0xF4 ~y120) is walkable.  The Stepladder is an L4 item
the real Survival route already earned; this recon pin is a stripped
Level7Entrance poke chain that never carried L4 inventory.

Does NOT write Candle, Food, Whistle, Triforce, doors, max_bombs, room,
health, or any other undiscovered item.  Not Clean.  Not on any spine.

    QT_QPA_PLATFORM=offscreen PYTHONPATH=nes:. uv run python \\
        nes/zelda_i/scratch/build_level7_interior49_ladder_fixture.py
"""

from __future__ import annotations

from typing import Any

from retro_harness.env import make_env, reset_obs, save_state, state_path
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless
from zelda_i.dungeon.trace import compact_snapshot, write_state_provenance
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import (
    ADDR_BOMBS,
    ADDR_CANDLE,
    ADDR_FOOD,
    ADDR_KEYS,
    ADDR_LADDER,
    ADDR_WHISTLE,
    PLAY_MODE,
    read_snapshot,
    read_u8,
)
from zelda_i.runner import make_assist

BEAD = "rr-8t4.2"
SOURCE_STATE = "Level7Interior49ReconFixture"
FIXTURE_NAME = "Level7Interior49LadderReconFixture"
ROOM = 0x49
SETTLE_FRAMES = 16


def _assign(env: Any, address: int, value: int) -> None:
    env.unwrapped.data.memory.assign(int(address), "|u1", int(value) & 0xFF)


def main() -> int:
    configure_headless()
    assist = make_assist(True)
    env = make_env(GAME, SOURCE_STATE, GAME_DIR, render_mode="rgb_array")
    try:
        reset_obs(env)
        for _ in range(2):
            env.step(nes_idle_action())
        ram = env.get_ram()
        before = read_snapshot(ram)
        ladder_before = int(read_u8(ram, ADDR_LADDER))
        print(f"source ({SOURCE_STATE}) glance: {compact_snapshot(before)}")
        print(
            f"ladder={ladder_before} candle={int(read_u8(ram, ADDR_CANDLE))} "
            f"food={int(read_u8(ram, ADDR_FOOD))} whistle={int(read_u8(ram, ADDR_WHISTLE))} "
            f"keys={int(read_u8(ram, ADDR_KEYS))} bombs={int(read_u8(ram, ADDR_BOMBS))}"
        )
        if not (
            before.level == 7
            and before.mode == PLAY_MODE
            and before.screen == ROOM
            and ladder_before == 0
            and int(read_u8(ram, ADDR_CANDLE)) == 0
            and int(read_u8(ram, ADDR_FOOD)) == 1
            and int(read_u8(ram, ADDR_WHISTLE)) == 1
        ):
            raise SystemExit(
                "source pin mismatch: expected L7 play 0x49 ladder=0 candle=0 food=1 whistle=1"
            )

        _assign(env, ADDR_LADDER, 1)
        ladder_after = int(read_u8(env.get_ram(), ADDR_LADDER))
        if ladder_after != 1:
            raise SystemExit("ADDR_LADDER write did not take")
        fixture_writes = [
            {
                "field": "ladder",
                "name": "ladder_for_0x49_moat",
                "address": ADDR_LADDER,
                "address_hex": f"0x{ADDR_LADDER:04X}",
                "before": ladder_before,
                "from": ladder_before,
                "to": ladder_after,
            }
        ]

        for _ in range(SETTLE_FRAMES):
            env.step(nes_idle_action())
            assist.apply_env(env, frame=0)

        ram = env.get_ram()
        after = read_snapshot(ram)
        print(f"fixture glance: {compact_snapshot(after)}")
        print(f"ladder={int(read_u8(ram, ADDR_LADDER))}")
        if not (
            after.level == 7
            and after.mode == PLAY_MODE
            and after.screen == ROOM
            and int(read_u8(ram, ADDR_LADDER)) == 1
            and int(read_u8(ram, ADDR_CANDLE)) == 0
            and int(read_u8(ram, ADDR_FOOD)) == 1
            and int(read_u8(ram, ADDR_WHISTLE)) == 1
            and after.triforce == 0
        ):
            raise SystemExit("fixture pin mismatch after ladder write")

        telem = assist.telemetry
        if int(telem.deaths) != 0:
            raise SystemExit(f"deaths={telem.deaths}")
        if int(telem.progression_writes) or int(telem.capacity_writes):
            raise SystemExit("progression/capacity writes")

        path = save_state(env, GAME_DIR, GAME, FIXTURE_NAME)
        source_path = state_path(GAME_DIR, GAME, SOURCE_STATE)
        write_state_provenance(
            path,
            source_state_path=source_path if source_path.exists() else None,
            request={
                "bead": BEAD,
                "phase": "level7_interior_49_ladder_recon",
                "track": "recon_fixture",
                "route_eligible": False,
                "fixture_only": True,
                "natural_entry": False,
                "development_only": True,
                "fixture_writes": fixture_writes,
                "notes": [
                    "Derived from Level7Interior49ReconFixture. Disclosed "
                    "ADDR_LADDER $0663 0->1 so the 0x49 full-width water moat "
                    "is walkable. The Stepladder is an L4 item the real Survival "
                    "route already earned; this recon pin never carried L4 inventory.",
                    "No Candle/Food/Whistle/TF/door/max_bombs/room/health writes.",
                    "UnlimitedHealthAssist settle aid only; deaths=0, "
                    "progression_writes=0, capacity_writes=0.",
                ],
            },
            selected_trial={
                "ok": True,
                "state": compact_snapshot(after),
                "glance": {
                    "screen": f"0x{int(after.screen):02x}",
                    "mode": int(after.mode),
                    "xy": [int(after.link_x), int(after.link_y)],
                    "keys": int(read_u8(ram, ADDR_KEYS)),
                    "bombs": int(read_u8(ram, ADDR_BOMBS)),
                    "food": int(read_u8(ram, ADDR_FOOD)),
                    "candle": int(read_u8(ram, ADDR_CANDLE)),
                    "whistle": int(read_u8(ram, ADDR_WHISTLE)),
                    "ladder": int(read_u8(ram, ADDR_LADDER)),
                    "triforce": int(after.triforce),
                },
                "fixture_writes": fixture_writes,
            },
            natural_entry=False,
        )
        print(f"saved {path}")
        print(f"saved {path.with_suffix('.provenance.json')}")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
