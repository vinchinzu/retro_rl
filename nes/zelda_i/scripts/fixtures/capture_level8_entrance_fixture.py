"""rr-6o7.1: capture the first-ever live L8 interior state.

Fixture-only recon. From ``Level8BushWithCandleFixture`` (candle
owned+selected, triforce 0x7F, disclosed pokes only), reproduces the
sweep-discovered burn: stand near (136, 93) on OW 0x6D, face RIGHT, fire
Red Candle (B), then continue walking RIGHT. This is not a natural walk —
Link's stand position on 0x6D is teleported there via the Stage-1 fixture,
not walked from a measured post-L7 handoff — so the result is
``route_eligible: false`` / ``natural_entry: false`` fixture evidence, not a
route promotion.

Saves the resulting live L8 interior as ``Level8EntranceReconFixture`` with
a full provenance sidecar (recon_fixture schema, house style from
``zelda_i/level9/stair_session.py``).

    uv run python nes/zelda_i/scratch/capture_level8_entrance_fixture.py
"""

from __future__ import annotations

from typing import Any

from retro_harness.env import make_env, reset_obs, resync_custom_state, save_state, state_path
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless
from zelda_i.dungeon.trace import compact_snapshot, write_state_provenance
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import ADDR_LINK_X, ADDR_LINK_Y, read_snapshot, read_u8

BEAD = "rr-6o7.1"
FIXTURE_SOURCE = "Level8BushWithCandleFixture"
FIXTURE_NAME = "Level8EntranceReconFixture"
LEVEL8 = 8
PLAY_MODE = 5
ADDR_CANDLE_USED = 0x0513

STAND_X, STAND_Y = 136, 93
FACING = "RIGHT"
PUSH = "RIGHT"

FIXTURE_WRITES: list[dict[str, Any]] = [
    {
        "name": "teleport_to_burn_stand",
        "addresses": [ADDR_LINK_X, ADDR_LINK_Y],
        "address_hex": [f"0x{ADDR_LINK_X:04X}", f"0x{ADDR_LINK_Y:04X}"],
        "values": [STAND_X, STAND_Y],
        "note": "not a walked position; teleport from Stage-1 fixture stand",
    },
]


def _assign(env: Any, address: int, value: int) -> None:
    env.unwrapped.data.memory.assign(int(address), "|u1", int(value) & 0xFF)


def main() -> int:
    configure_headless()
    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    reset_obs(env)
    resync_custom_state(env, GAME_DIR, GAME, FIXTURE_SOURCE)

    _assign(env, ADDR_LINK_X, STAND_X)
    _assign(env, ADDR_LINK_Y, STAND_Y)
    for _ in range(10):
        env.step(nes_idle_action())
    for _ in range(4):
        env.step(nes_action(FACING))
    for _ in range(6):
        env.step(nes_action("B"))
    for _ in range(70):
        env.step(nes_idle_action())
    candle_used_after_fire = bool(read_u8(env.get_ram(), ADDR_CANDLE_USED))

    for _ in range(40):
        env.step(nes_action(PUSH))

    reached_frame = None
    for i in range(400):
        env.step(nes_idle_action())
        snap = read_snapshot(env.get_ram())
        if snap.level == LEVEL8 and snap.mode == PLAY_MODE:
            reached_frame = i
            break
    else:
        raise SystemExit("did not reach live L8 interior — recipe regressed, aborting capture")

    ram = env.get_ram()
    snap = read_snapshot(ram)
    live_objects = [
        {
            "slot": obj.slot,
            "type_id": obj.type_id,
            "x": obj.x,
            "y": obj.y,
            "facing": obj.facing,
            "hp": obj.hp,
            "state": obj.state,
        }
        for obj in snap.objects
        if obj.type_id != 0 or obj.slot == 0
    ]
    result = {
        "ok": True,
        "candle_used_after_fire": candle_used_after_fire,
        "reached_frame_after_push": reached_frame,
        "entry_room": int(snap.screen),
        "entry_room_hex": hex(int(snap.screen)),
        "state": compact_snapshot(snap),
        "objects": live_objects,
        "cur_opened_doors": int(snap.cur_opened_doors),
        "open_doorway_mask": int(snap.open_doorway_mask),
        "room_item_id": int(snap.room_item_id),
        "room_obj_count": int(snap.room_obj_count),
        "room_all_dead": int(snap.room_all_dead),
    }
    print("captured live L8 interior:", result)

    path = save_state(env, GAME_DIR, GAME, FIXTURE_NAME)
    source_path = state_path(GAME_DIR, GAME, FIXTURE_SOURCE)
    write_state_provenance(
        path,
        source_state_path=source_path if source_path.exists() else None,
        request={
            "bead": BEAD,
            "phase": "level8_first_live_interior_entry",
            "track": "recon_fixture",
            "route_eligible": False,
            "fixture_only": True,
            "natural_entry": False,
            "fixture_writes": FIXTURE_WRITES,
            "burn_recipe": {
                "stand_xy": [STAND_X, STAND_Y],
                "facing": FACING,
                "push_direction": PUSH,
                "notes": [
                    "sourced from a 5856-trial live sweep, nes/zelda_i/logs/level8_bush_burn_sweep.json",
                    "sweep confirmed mode-16 mouth opened from (120,93)/(128,93) RIGHT, (184,93)/(192,93)/(200,93) LEFT, and (160,77) DOWN (entry_room null in sweep); only (136,93) RIGHT/RIGHT was live-walked into room 0x7E",
                    "sweep header reports 732 standable positions; the trial grid contains 729 distinct coordinates due to 3 duplicate stand samples",
                    "entry does NOT complete on UP alone after mode16; continuing the SAME",
                    "push_direction used to fire is what carries Link through into level 8",
                    "(BurnLevel8BushController's ENTER phase always sends UP — flagged, not changed)",
                ],
            },
        },
        selected_trial=result,
        natural_entry=False,
    )
    print(f"saved {path}")
    print(f"saved {path.with_suffix('.provenance.json')}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
