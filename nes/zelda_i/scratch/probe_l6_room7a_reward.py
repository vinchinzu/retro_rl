"""L6 room 0x7a east-key reward-phase recon. Not a route claim.

    QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scratch/probe_l6_room7a_reward.py \
        --no-video --tag l6_7a_reward

Starts from ``Level6Entrance.state`` (0x79), walks the wall-first RIGHT into
0x7a (``Level6EntryRightController``), then drives the real fight controller
(``make_east_key_controller``) exactly like the spine hop does. Once the
clear signal fires, logs every 16 frames: frame, phase, link xy, room_item_id,
keys, room_all_dead, live-enemy count, and the FrameAction reason, plus the
live object table (any non-empty slot) so a dropped key sprite would show up.
Saves a PNG + ascii tile map at clear and at timeout/success so the reward
target can be checked against real room geometry.
"""

from __future__ import annotations

import argparse

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.engine import DungeonPhase
from zelda_i.dungeon.ids import object_name
from zelda_i.dungeon.tilemap import ascii_room, has_room_tile_map
from zelda_i.level6.dungeon import ROOM_7A_SPEC
from zelda_i.level6.overworld import EntryRightPhase, Level6EntryRightController
from zelda_i.level6.wizzrobe import make_east_key_controller
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot
from zelda_i.runner import add_common_args, make_assist, write_report

DEFAULT_STATE = "Level6Entrance"


def _dump(tag: str, obs, ram, snap, label: str) -> None:
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    png = RECORDINGS_DIR / f"{tag}_{label}.png"
    save_rgb_png(obs, png)
    print(f"  [{label}] png={png}")
    if has_room_tile_map(ram):
        print(ascii_room(ram))
    else:
        print(f"  [{label}] no room tile map available (screen 0x{snap.screen:02x})")


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_state=DEFAULT_STATE, default_tag="l6_7a_reward")
    parser.add_argument("--no-video", action="store_true", help="ignored; always rgb_array")
    parser.add_argument(
        "--lead-idle",
        type=int,
        default=0,
        help="Extra idle frames before driving controllers (RNG-phase probe only).",
    )
    args = parser.parse_args()
    configure_headless()
    assist = make_assist(args.infinite_life)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    try:
        obs, _ = reset_obs(env)
        obs, *_ = env.step(nes_idle_action())
        for _ in range(args.lead_idle):
            obs, *_ = env.step(nes_idle_action())
        ram0 = env.get_ram()
        snap0 = read_snapshot(ram0)
        print(
            f"start screen=0x{snap0.screen:02x} xy=({snap0.link_x},{snap0.link_y}) "
            f"mode={snap0.mode} keys={snap0.keys}"
        )

        right = Level6EntryRightController()
        fight = make_east_key_controller()

        dumped_clear = False
        frame = 0
        max_frames = right.max_frames + ROOM_7A_SPEC.max_frames + 200
        stage = "right"
        while frame < max_frames:
            frame += 1
            snap = read_snapshot(env.get_ram())
            if stage == "right":
                action = right.step(snap)
                if right.success:
                    stage = "fight"
                    print(f"f={frame} entered 0x7a xy=({snap.link_x},{snap.link_y})")
                elif right.phase is EntryRightPhase.FAILED:
                    print(f"f={frame} RIGHT controller FAILED notes={right.notes}")
                    break
            else:
                action = fight.step(snap)
                if not dumped_clear and fight.clear_signal_seen:
                    dumped_clear = True
                    _dump(args.tag, obs, env.get_ram(), snap, f"clear_f{frame}")
                if fight.phase in (DungeonPhase.COLLECT_REWARD, DungeonPhase.DONE) and (
                    frame % 16 == 0
                ):
                    live_objs = [
                        (o.slot, hex(o.type_id), o.x, o.y, o.hp, object_name(o.type_id))
                        for o in snap.objects
                        if o.type_id != 0
                    ]
                    print(
                        f"f={frame} phase={fight.phase.name} xy=({snap.link_x},{snap.link_y}) "
                        f"room_item_id=0x{snap.room_item_id:02x} keys={snap.keys} "
                        f"room_all_dead={snap.room_all_dead} live={fight.last_live_enemies} "
                        f"reason={action.reason} objs={live_objs}"
                    )
                if fight.success or fight.phase is DungeonPhase.FAILED:
                    break
            obs, *_ = env.step(action.action)
            if assist is not None:
                assist.apply_env(env, frame=frame)

        snap = read_snapshot(env.get_ram())
        _dump(args.tag, obs, env.get_ram(), snap, "final")
        payload = {
            "right_report": right.report(),
            "fight_report": fight.report(),
            "final_screen": hex(snap.screen),
            "final_xy": (int(snap.link_x), int(snap.link_y)),
            "final_keys": int(snap.keys),
            "frames_used": frame,
        }
        out = write_report("l6_7a_reward_probe", payload, tag=args.tag)
        print(out)
        print(
            f"fight.success={fight.success} fight.phase={fight.phase.name} "
            f"phase={fight.phase.name} frames={fight.frames} "
            f"final_xy={payload['final_xy']} keys={payload['final_keys']}"
        )
        print(f"fight notes={fight.notes}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
