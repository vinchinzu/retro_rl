"""rr-8t4.1: capture the first-ever *naturally arrived* pond ``0x42`` pin.

``OW_L7Pond`` is a whistle=0 geometry-recon pin from ``PostSwordStart`` and
``AGENTS.md`` forbids starting the drain from it. This script instead
replays the already-proven 2/2 route from ``scratch/pond/
probe_recorder_warp_full_route.py`` (power-on-reachable
``POST_L6_EXIT_STATE`` -> greened ``POST_L6_TO_POND_HOPS`` prefix to ``0x24``
-> ``PauseSelectController`` selects the naturally-owned Recorder -> blow
facing DOWN until the whirlwind cycle lands on L4 island door ``0x45`` ->
join hop ``0x45 DOWN align_x=128`` -> ``0x55`` -> reuse the greened
``LEVEL7_POND_HOPS`` tail from ``0x55`` onward) and stops the instant Link
settles on pond ``0x42``, **before** ever blowing the Recorder there.

No RAM pokes anywhere in this recipe (writes=0). Saves the result as
``OW_L7PondNatural`` with a provenance sidecar, ``natural_entry: true``.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/pond/capture_pond_natural_pin.py
"""

from __future__ import annotations

from typing import Any

from retro_harness.env import make_env, reset_obs, save_state, state_path
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless
from zelda_i.dungeon.pause_select import B_SLOT_RECORDER, PauseSelectController
from zelda_i.dungeon.trace import compact_snapshot, write_state_provenance
from zelda_i.level7.entry import POST_L6_EXIT_STATE
from zelda_i.level7.overworld import LEVEL7_POND_HOPS, on_level7_pond_hyp
from zelda_i.level7.pond import POST_L6_TO_POND_HOPS, PostLevel6OverworldController
from zelda_i.overworld.graph import ScreenHop
from zelda_i.overworld.stitch import handoff_from_ram
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8
from zelda_i.runner import make_assist

BEAD = "rr-8t4.1"
PIN_NAME = "OW_L7PondNatural"
WALK_MAX = 20_000
FACE_SETTLE = 20
SETTLE_STABLE_FRAMES = 90
MAX_WAIT_PER_BLOW = 3000
MAX_BLOWS = 10
TAIL_MAX_FRAMES = 30_000
ISLAND_SCREEN = 0x45
JOIN_TARGET = 0x55
JOIN_ALIGN_X = 128
HOP_65_ALIGN_X = 128  # see probe_recorder_warp_full_route.py note on 0x55 dock x


def main() -> int:
    configure_headless()
    env = make_env(GAME, POST_L6_EXIT_STATE, GAME_DIR, render_mode="rgb_array")
    assist = make_assist(True)  # Survival infinite-life; disclosed, no progression writes
    total = [0]

    def step(action: Any):
        env.step(action)
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
        return read_snapshot(env.get_ram())

    def wait_settled(max_wait: int) -> tuple[Any, bool]:
        stable = 0
        last_xy = None
        for _ in range(max_wait):
            snap = read_snapshot(env.get_ram())
            xy = (int(snap.link_x), int(snap.link_y))
            if snap.mode == PLAY_MODE and not snap.transitioning and xy == last_xy:
                stable += 1
                if stable >= SETTLE_STABLE_FRAMES:
                    return snap, True
            else:
                stable = 0
            last_xy = xy
            snap = step(nes_idle_action())
        return read_snapshot(env.get_ram()), False

    reset_obs(env)
    snap = step(nes_idle_action())

    # --- Phase 1: walk the greened non-entrance prefix to 0x24 ---
    truncated_hops = POST_L6_TO_POND_HOPS[:4]
    handoff = handoff_from_ram(env.get_ram(), evidence="capture-pond-natural-pin", verified=True)
    ctl = PostLevel6OverworldController(handoff=handoff, hops=truncated_hops)
    ctl.bind_env(env)
    walked_ok = False
    for _ in range(WALK_MAX):
        snap = read_snapshot(env.get_ram())
        action = ctl.step(snap)
        snap = step(action.action)
        if int(snap.screen) == 0x24 and snap.mode == PLAY_MODE and not snap.transitioning:
            walked_ok = True
            break
        if ctl.failed:
            break
    print(f"walked_ok={walked_ok} landed=0x{snap.screen:02x} f={total[0]}")
    if not walked_ok:
        env.close()
        raise SystemExit("prefix walk to 0x24 regressed, aborting capture")

    # --- Phase 2: select the Recorder (already naturally owned) ---
    selector = PauseSelectController(want=B_SLOT_RECORDER, name="recorder")
    selector.bind_env(env)
    select_ok = False
    for _ in range(selector.max_frames + 20):
        snap = read_snapshot(env.get_ram())
        action = selector.drive(snap)
        if action is None:
            select_ok = True
            break
        if selector.failed:
            break
        snap = step(action.action)
    print(f"select_ok={select_ok} f={total[0]}")
    if not select_ok:
        env.close()
        raise SystemExit("recorder select regressed, aborting capture")

    # --- Phase 3: blow facing DOWN until landing on island 0x45 ---
    reached_island = False
    for blow_i in range(MAX_BLOWS):
        for _ in range(FACE_SETTLE):
            snap = step(nes_action("DOWN"))
        for _ in range(12):
            snap = step(nes_action("B"))
        post, settled = wait_settled(MAX_WAIT_PER_BLOW)
        print(
            f"blow {blow_i}: -> 0x{int(post.screen):02x} xy=({int(post.link_x)},{int(post.link_y)}) "
            f"settled={settled} f={total[0]}"
        )
        if not settled or int(post.level) != 0:
            break
        if int(post.screen) == ISLAND_SCREEN and post.mode == PLAY_MODE and not post.transitioning:
            reached_island = True
            break
    print(f"reached_island={reached_island}")
    if not reached_island:
        env.close()
        raise SystemExit("recorder warp to island 0x45 regressed, aborting capture")

    # --- Phase 4: join hop 0x45 -> 0x55, then the greened LEVEL7_POND_HOPS
    # tail from 0x55 onward -> pond 0x42. Stop the instant we arrive; never
    # blow the recorder at the pond. ---
    full_hops: tuple[ScreenHop, ...] = (
        ScreenHop(JOIN_TARGET, "DOWN", align_x=JOIN_ALIGN_X),
        ScreenHop(0x65, "DOWN", align_x=HOP_65_ALIGN_X),
    ) + LEVEL7_POND_HOPS[7:]
    handoff2 = handoff_from_ram(env.get_ram(), evidence="capture-pond-natural-pin-tail", verified=True)
    tail_ctl = PostLevel6OverworldController(handoff=handoff2, hops=full_hops, max_frames=TAIL_MAX_FRAMES)
    tail_ctl.bind_env(env)
    pond_ok = False
    snap = read_snapshot(env.get_ram())
    for _ in range(TAIL_MAX_FRAMES):
        snap = read_snapshot(env.get_ram())
        if on_level7_pond_hyp(snap):
            pond_ok = True
            break
        if tail_ctl.failed:
            break
        action = tail_ctl.step(snap)
        snap = step(action.action)
        if on_level7_pond_hyp(snap):
            pond_ok = True
            break
        if tail_ctl.failed:
            break
    print(
        f"pond_ok={pond_ok} tail_failed={tail_ctl.failed} screen=0x{snap.screen:02x} "
        f"xy=({snap.link_x},{snap.link_y}) mode={snap.mode} f={total[0]}"
    )
    if not pond_ok:
        env.close()
        raise SystemExit("tail hops to pond 0x42 regressed, aborting capture")

    # Settle a few idle frames so the saved state is not mid-frame-transition.
    for _ in range(SETTLE_STABLE_FRAMES):
        snap = step(nes_idle_action())

    ram = env.get_ram()
    whistle = int(read_u8(ram, ADDR_WHISTLE))
    snap = read_snapshot(ram)
    print(
        f"final: screen=0x{snap.screen:02x} xy=({snap.link_x},{snap.link_y}) "
        f"mode={snap.mode} whistle={whistle} frames={total[0]}"
    )
    if int(snap.screen) != 0x42 or whistle < 1:
        env.close()
        raise SystemExit("final state is not natural pond 0x42 with owned whistle, aborting")

    path = save_state(env, GAME_DIR, GAME, PIN_NAME)
    source_path = state_path(GAME_DIR, GAME, POST_L6_EXIT_STATE)
    write_state_provenance(
        path,
        source_state_path=source_path if source_path.exists() else None,
        request={
            "bead": BEAD,
            "phase": "level7_pond_natural_arrival",
            "track": "route_checkpoint",
            "route_eligible": False,
            "fixture_only": False,
            "natural_entry": True,
            "whistle_poke": False,
            "writes": 0,
            "route": {
                "notes": [
                    "power-on-reachable: POST_L6_EXIT_STATE (Level6ExitOverworld, "
                    "measured level6-exit fanfare leftover) -> POST_L6_TO_POND_HOPS[:4] "
                    "walk to 0x24 -> PauseSelectController selects the naturally-owned "
                    "Recorder (B-slot 5, no $0656 poke) -> 12xB facing DOWN, whirlwind "
                    "cycle lands on L4 island door 0x45 -> join hop 0x45 DOWN "
                    "align_x=128 -> 0x55 -> greened LEVEL7_POND_HOPS[7:] tail "
                    "(0x65 DOWN align_x=128, 0x64 LEFT align_y=141, 0x54 UP, "
                    "0x53 LEFT align_y=141, 0x52 LEFT align_y=189, pond 0x42 UP "
                    "align_x=112) -> pond 0x42 mode 5, not transitioning",
                    "recorder never blown again after the island 0x45 warp landing; "
                    "the pond drain blow itself happens downstream via "
                    "Level7PondDrainController, never inside this capture",
                    "supersedes OW_L7Pond (PostSwordStart geometry recon, whistle=0, "
                    "AGENTS.md forbids starting the drain from it)",
                ],
            },
        },
        selected_trial=compact_snapshot(snap) | {"whistle": whistle, "frames": total[0]},
        natural_entry=True,
    )
    print(f"saved {path}")
    print(f"saved {path.with_suffix('.provenance.json')}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
