"""Fast-iteration probe for Level9Stairs05Controller (rr-sz8.6), the hop that
fails power-on (times out at (98,149), mode=5) after the Spectacle Rock bomb
fix landed. Drives the real chain from Level8OWLeaveLive through
Level9PostL8OverworldController -> Level9SpectacleRockBombController ->
NaturalSilverArrowsController hops 0-8, then instruments hop 9
(Level9Stairs05Controller) frame by frame: block position, live wizzrobes,
Link position, and the controller's chosen action/reason.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_l9_05_stairs_diag.py
"""

from __future__ import annotations

from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.level9.dungeon import MEASURED_POST_L8_HANDOFF
from zelda_i.level9.natural_path import NaturalSilverArrowsController
from zelda_i.level9.overworld import (
    Level9PostL8OverworldController,
    Level9SpectacleRockBombController,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot

CLEAN_STATE = "L9Room05Entry"


def main() -> int:
    configure_headless()
    assist = UnlimitedHealthAssist(enabled=True)
    env = make_env(GAME, "Level8OWLeaveLive", GAME_DIR, render_mode="rgb_array")
    obs, _ = reset_obs(env)
    frame = 0

    ow = Level9PostL8OverworldController(handoff=MEASURED_POST_L8_HANDOFF)
    ow.bind_env(env)
    for _ in range(ow.max_frames):
        snap = read_snapshot(env.get_ram())
        act = ow.step(snap)
        obs, *_ = env.step(act.action)
        frame += 1
        assist.apply_env(env, frame=frame)
        if ow.failed or ow.success:
            break
    assert ow.success, ow.blocked_reason

    bomb = Level9SpectacleRockBombController(handoff=MEASURED_POST_L8_HANDOFF)
    bomb.bind_env(env)
    for _ in range(bomb.max_frames):
        snap = read_snapshot(env.get_ram())
        act = bomb.step(snap)
        obs, *_ = env.step(act.action)
        frame += 1
        assist.apply_env(env, frame=frame)
        if bomb.failed or bomb.success:
            break
    assert bomb.success, bomb.failure

    sa = NaturalSilverArrowsController(handoff=MEASURED_POST_L8_HANDOFF)
    sa.bind_env(env) if hasattr(sa, "bind_env") else None
    last_hop = -1
    for i in range(sa.max_frames):
        snap = read_snapshot(env.get_ram())
        if snap.hop_i if False else False:
            pass
        act = sa.step(snap)
        if sa.hop_i != last_hop:
            print(f"f{i}: entering hop_i={sa.hop_i} screen=0x{snap.screen:02x} "
                  f"xy=({snap.link_x},{snap.link_y}) keys={snap.keys} bombs={snap.bombs}")
            last_hop = sa.hop_i
        obs, *_ = env.step(act.action)
        frame += 1
        assist.apply_env(env, frame=frame)
        if sa.hop_i == 9 and sa._hops[9].frames == 0:
            # About to start hop 9 (stairs_05) fresh -- pin a savestate here.
            break
        if sa.failed or sa.success:
            break

    snap = read_snapshot(env.get_ram())
    print(f"pre-hop9 pin: screen=0x{snap.screen:02x} xy=({snap.link_x},{snap.link_y}) "
          f"keys={snap.keys} bombs={snap.bombs} hop_i={sa.hop_i} failed={sa.failed}")
    save_state(env, GAME_DIR, GAME, CLEAN_STATE)
    png = RECORDINGS_DIR / "probe_l9_05_stairs_entry.png"
    save_rgb_png(obs, png)
    print(f"screenshot: {png}")

    if sa.failed:
        print("ALREADY FAILED before hop 9:", sa.notes)
        env.close()
        return 1

    # Now drive hop 9 alone with heavy instrumentation.
    hop9 = sa._hops[9]
    print(f"hop9 spec_id={hop9.spec_id} origin=0x{hop9.origin:02x} dest_hyp=0x{hop9.dest_hyp:02x}")
    last_key = None
    for i in range(hop9.max_frames):
        snap = read_snapshot(env.get_ram())
        block = next((o for o in snap.objects if o.type_id == 0x68 or o.slot == 11), None)
        wizz = [o for o in snap.objects if o.type_id in (0x23, 0x24) and o.hp > 0]
        act = hop9.step(snap)
        key = (act.reason, snap.link_x, snap.link_y, block.y if block else None, len(wizz))
        if key != last_key:
            bstr = f"block_y={block.y}" if block else "block=None"
            wstr = ",".join(f"({w.x},{w.y},hp{w.hp})" for w in wizz)
            print(f"f{i}: xy=({snap.link_x},{snap.link_y}) {bstr} wizz=[{wstr}] "
                  f"act={act.reason}")
            last_key = key
        obs, *_ = env.step(act.action)
        frame += 1
        assist.apply_env(env, frame=frame)
        if hop9.failed or hop9.success:
            print(f"DONE f{i}: failed={hop9.failed} success={hop9.success} notes={hop9.notes}")
            break
    else:
        print("hop9 ran out of max_frames without success/fail flag set")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
