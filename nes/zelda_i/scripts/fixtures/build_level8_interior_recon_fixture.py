"""Build the disclosed resource fixture for isolated Level 8 probing.

rr-6o7.2. Starts from ``Level8EntranceReconFixture`` (already settled in
Level 8 room ``0x7E``) and discloses only the combat/economy resources needed
to probe the first interior gates: Magical Sword, bombs capped by the
existing max-bomb capacity, Bow, one wooden arrow, rupees, and pre-Magical-Key
keys. This is not a natural route state and is never route eligible.

The builder deliberately does not write Magic Key, triforce, room/screen,
door flags, health, or heart capacity. Every resource write records its
``before``/``from`` and ``to`` values in the provenance sidecar.

    QT_QPA_PLATFORM=offscreen uv run python \\
        nes/zelda_i/scratch/build_level8_interior_recon_fixture.py
"""

from __future__ import annotations

from typing import Any

from retro_harness.env import make_env, reset_obs, save_state, state_path
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless
from zelda_i.dungeon.trace import compact_snapshot, write_state_provenance
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOMBS,
    ADDR_BOW,
    ADDR_KEYS,
    ADDR_MAX_BOMBS,
    ADDR_RUPEES,
    ADDR_SWORD,
    ADDR_TRIFORCE,
    ADDR_LEVEL,
    ADDR_MODE,
    ADDR_SCREEN,
    PLAY_MODE,
    read_snapshot,
    read_u8,
)

BEAD = "rr-6o7.2"
SOURCE_STATE = "Level8EntranceReconFixture"
FIXTURE_NAME = "Level8InteriorReconFixture"

LEVEL8 = 8
ENTRY_ROOM = 0x7E
POST_L7_TRIFORCE = 0x7F
MAGICAL_SWORD = 3
WOODEN_ARROWS = 1
PREFERRED_BOMBS = 8
PREFERRED_KEYS = 9
RUPEES = 255
SETTLE_FRAMES = 30


def _assign(env: Any, address: int, value: int) -> None:
    env.unwrapped.data.memory.assign(int(address), "|u1", int(value) & 0xFF)


def main() -> int:
    configure_headless()
    env = make_env(GAME, SOURCE_STATE, GAME_DIR, render_mode="rgb_array")
    try:
        obs, *_ = reset_obs(env)
        before_snap = read_snapshot(env.get_ram())
        before_ram = env.get_ram()
        print(f"source ({SOURCE_STATE}) glance: {compact_snapshot(before_snap)}")

        if not (
            before_snap.level == LEVEL8
            and before_snap.mode == PLAY_MODE
            and before_snap.screen == ENTRY_ROOM
            and before_snap.triforce == POST_L7_TRIFORCE
        ):
            raise SystemExit(
                "source pin mismatch: expected settled L8 room 0x7E with TF 0x7F"
            )

        max_bombs = int(read_u8(before_ram, ADDR_MAX_BOMBS))
        bombs = min(PREFERRED_BOMBS, max_bombs)
        write_specs = (
            ("magical_sword", ADDR_SWORD, MAGICAL_SWORD),
            ("bombs_for_recon", ADDR_BOMBS, bombs),
            ("bow_for_recon", ADDR_BOW, 1),
            ("wooden_arrows_for_recon", ADDR_ARROWS, WOODEN_ARROWS),
            ("rupees_for_recon", ADDR_RUPEES, RUPEES),
            ("keys_for_pre_magic_key_doors", ADDR_KEYS, PREFERRED_KEYS),
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
            obs, *_ = env.step(nes_idle_action())

        ram = env.get_ram()
        after = read_snapshot(ram)
        print(f"fixture glance: {compact_snapshot(after)}")
        if not (
            after.level == LEVEL8
            and after.mode == PLAY_MODE
            and after.screen == ENTRY_ROOM
            and after.triforce == POST_L7_TRIFORCE
        ):
            raise SystemExit(
                "fixture pin mismatch after resource writes: expected L8 0x7E TF 0x7F"
            )

        path = save_state(env, GAME_DIR, GAME, FIXTURE_NAME)
        source_path = state_path(GAME_DIR, GAME, SOURCE_STATE)
        inventory = {
            "sword": int(read_u8(ram, ADDR_SWORD)),
            "bombs": int(read_u8(ram, ADDR_BOMBS)),
            "max_bombs": int(read_u8(ram, ADDR_MAX_BOMBS)),
            "bow": int(read_u8(ram, ADDR_BOW)),
            "arrows": int(read_u8(ram, ADDR_ARROWS)),
            "rupees": int(read_u8(ram, ADDR_RUPEES)),
            "keys": int(read_u8(ram, ADDR_KEYS)),
        }
        result = {
            "ok": True,
            "source_state": SOURCE_STATE,
            "fixture_state": FIXTURE_NAME,
            "state": compact_snapshot(after),
            "inventory": inventory,
            "fixture_writes": fixture_writes,
            "protected_unchanged": {
                "level": int(read_u8(ram, ADDR_LEVEL)) == LEVEL8,
                "mode": int(read_u8(ram, ADDR_MODE)) == PLAY_MODE,
                "screen": int(read_u8(ram, ADDR_SCREEN)) == ENTRY_ROOM,
                "triforce": int(read_u8(ram, ADDR_TRIFORCE)) == POST_L7_TRIFORCE,
            },
        }
        write_state_provenance(
            path,
            source_state_path=source_path if source_path.exists() else None,
            request={
                "bead": BEAD,
                "phase": "level8_interior_recon_resources",
                "track": "recon_fixture",
                "route_eligible": False,
                "fixture_only": True,
                "natural_entry": False,
                "fixture_writes": fixture_writes,
                "notes": [
                    "Derived from the fixture-only Level8EntranceReconFixture.",
                    "Bomb target is min(preferred 8, existing max-bomb capacity).",
                    "Keys are a disclosed pre-Magical-Key recon resource; Magic Key is untouched.",
                    "No Magic Key, triforce, room, door flags, health, or heart capacity writes.",
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
