"""Scratch probe: visual/RAM trace of the unassisted pre-L1 walk.

``reason_by_screen`` is a *count*; it names the rung that owned 4301 frames
on 0x7C and says nothing about whether Link moved while it did. This probe
runs the real committed walk (assist OFF, the spine's own controller) and
records ``(frame, screen, x, y, reason)`` every frame, so a stall can be read
as geometry rather than as a histogram. On the two C3 coast screens it also
saves a PNG plus compact Link/object RAM on entry, every hit, and every 250
frames. Object velocities come from the same six-sample tracker used by the
controller (two samples for projectiles); they are measurements, not a second
policy.

    QT_QPA_PLATFORM=offscreen uv run python \
      nes/zelda_i/scratch/probe_walk_trace.py --tag pre_l1_c3_trace1
    uv run python nes/zelda_i/scratch/probe_walk_trace.py \
      --arm rollout --headed --tag pre_l1_c3_rollout7d

No RAM is written after the power-on reset. The default arm is the committed
reactive walk. ``--arm rollout`` binds the opt-in ROM-truth evader on the
inner emulator so simulated fans do not blit or count as STATUS loads; the
honest cost is ``controller.report()["rollout"]``. ``--headed`` opens the
pygame watch (``[ ]`` speed, TAB turbo, ESC quit). Not a route claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from retro_harness.audit import AuditCapabilities, AuditedEnv
from retro_harness.env import make_env, reset_obs
from retro_harness.headed import (
    HEADED_ATTR,
    add_headed_flag,
    attach_headed,
    idle_headed,
)
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.combat import facing_to_direction, overworld_threat_objects
from zelda_i.dungeon.tracking import HazardClass, ObjectTracker, TrackedObject
from zelda_i.overworld.gathering import (
    SHOP_P7_WALK_MAX_FRAMES,
    SWORD_MAX,
    SwordCaveController,
    make_shop_p7_walk_controller,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot, read_snapshot
from zelda_i.route.chain import boot_to_ready, run_controller_stage

OUT = RECORDINGS_DIR / "scratch_walk_trace"
FOCUS_SCREENS = frozenset({0x7C, 0x7D})


def _tracked_row(obj: TrackedObject) -> dict:
    return {
        "slot": int(obj.slot),
        "type": int(obj.type_id),
        "type_hex": f"0x{int(obj.type_id):02X}",
        "hazard": obj.hazard.value,
        "state": int(obj.state),
        "hp": int(obj.hp),
        "x": int(obj.x),
        "y": int(obj.y),
        "vx": round(float(obj.vx), 3),
        "vy": round(float(obj.vy), 3),
    }


def _live_hazards(
    snap: ZeldaSnapshot, tracked: tuple[TrackedObject, ...]
) -> list[dict]:
    """Live bodies plus projectiles; no drops, corpses or dormant leevers."""
    body_slots = {int(obj.slot) for obj in overworld_threat_objects(snap)}
    return [
        _tracked_row(obj)
        for obj in tracked
        if obj.hazard is HazardClass.PROJECTILE
        or (int(obj.slot) in body_slots and int(obj.hp) < 200)
    ]


def _link_row(snap: ZeldaSnapshot) -> dict:
    return {
        "level": int(snap.level),
        "screen": int(snap.screen),
        "screen_hex": f"0x{int(snap.screen):02X}",
        "mode": int(snap.mode),
        "x": int(snap.link_x),
        "y": int(snap.link_y),
        "facing": int(snap.facing),
        "health": int(snap.health),
        "health_hex": f"0x{int(snap.health):02X}",
        "heart_partial": int(snap.heart_partial),
        "heart_containers": int(snap.heart_containers),
        "whole_hearts": int(snap.whole_hearts),
        "iframes": int(snap.link_iframes),
        "sword": int(snap.sword),
        "bombs": int(snap.bombs),
        "rupees": int(snap.rupees),
        "keys": int(snap.keys),
        "triforce": int(snap.triforce),
        "bow": int(snap.bow),
        "arrows": int(snap.arrows),
        "candle": int(snap.candle),
        "food": int(snap.food),
        "rod": int(snap.rod),
        "raft": int(snap.raft),
        "ladder": int(snap.ladder),
    }


def _png_path(tag: str, index: int, event: str, snap: ZeldaSnapshot) -> Path:
    return OUT / (
        f"{tag}_{index:03d}_{event}_s{int(snap.screen):02X}_"
        f"x{int(snap.link_x)}_y{int(snap.link_y)}.png"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="w1")
    parser.add_argument("--sample-frames", type=int, default=250)
    parser.add_argument("--arm", choices=("reactive", "rollout"), default="reactive")
    add_headed_flag(parser)
    args = parser.parse_args(argv)
    if args.sample_frames < 1:
        parser.error("--sample-frames must be positive")

    OUT.mkdir(parents=True, exist_ok=True)
    if not args.headed:
        configure_headless()
    raw = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")

    trace: list[tuple[int, int, int, int, str]] = []
    evidence: list[dict] = []
    walk = make_shop_p7_walk_controller()
    if args.arm == "rollout":
        # Bind the live emulator, not the AuditedEnv wrap: a rollout is a
        # restore, and counting it as a STATUS mid-run load is the wrong
        # ledger. The honest cost is ``walk.report()["rollout"]``.
        walk.attach_rollout(raw)
    tracker = ObjectTracker(shot_history=walk.shot_history)
    inner = walk.step
    frame_base = 0
    prior_iframes = 0
    prior_reason: str | None = None
    last_screen: int | None = None
    entered: set[int] = set()
    next_sample: dict[int, int] = {}
    env = raw

    def capture(
        snap: ZeldaSnapshot,
        tracked: tuple[TrackedObject, ...],
        reason: str,
        events: list[str],
    ) -> None:
        event = "+".join(events)
        path = _png_path(args.tag, len(evidence), event, snap)
        save_rgb_png(env.render(), path)
        evidence.append(
            {
                "event": events,
                "frame": int(frame_base + walk.frames),
                "walk_frame": int(walk.frames),
                "action_reason": reason,
                "preceding_action_reason": prior_reason,
                "link": _link_row(snap),
                "objects": _live_hazards(snap, tracked),
                "screenshot": str(path),
            }
        )
        print(
            f"{event} 0x{int(snap.screen):02X} ({int(snap.link_x)},{int(snap.link_y)}) "
            f"f={int(frame_base + walk.frames)} reason={reason} png={path.name}",
            flush=True,
        )

    def traced(snap):
        nonlocal prior_iframes, prior_reason, last_screen
        tracked = tracker.observe(snap)
        act = inner(snap)
        trace.append((
            int(walk.frames), int(snap.screen), int(snap.link_x), int(snap.link_y),
            str(act.reason),
        ))
        screen = int(snap.screen)
        if last_screen != screen:
            print(
                f"screen 0x{screen:02X} ({int(snap.link_x)},{int(snap.link_y)}) "
                f"f={int(frame_base + walk.frames)} reason={act.reason}",
                flush=True,
            )
            last_screen = screen
        focused = (
            screen in FOCUS_SCREENS
            and int(snap.mode) == PLAY_MODE
            and not snap.transitioning
        )
        events: list[str] = []
        if focused and screen not in entered:
            entered.add(screen)
            next_sample[screen] = int(walk.frames) + args.sample_frames
            events.append("entry")
        if focused and prior_iframes == 0 and int(snap.link_iframes) > 0:
            events.append("hit")
        if focused and int(walk.frames) >= next_sample.get(screen, 10**12):
            events.append("sample")
            next_sample[screen] = int(walk.frames) + args.sample_frames
        if events:
            capture(snap, tracked, str(act.reason), events)
        prior_iframes = int(snap.link_iframes)
        prior_reason = str(act.reason)
        return act

    walk.step = traced  # type: ignore[method-assign]

    def _hud(live) -> str:
        ram = live.get_ram()
        snap = read_snapshot(ram)
        headed = getattr(live, HEADED_ATTR, None)
        frame = int(getattr(headed, "frame", 0) or 0)
        try:
            face = facing_to_direction(int(snap.facing))
        except ValueError:
            face = "?"
        return (
            f"f{frame} {args.arm} 0x{int(snap.screen):02x} "
            f"({int(snap.link_x)},{int(snap.link_y)}) {face} "
            f"{int(snap.rupees)}R {int(snap.whole_hearts)}/"
            f"{int(snap.heart_containers)}H {prior_reason or '-'}"
        )

    pygame_mod = None
    try:
        obs, _ = reset_obs(raw)
        env = AuditedEnv(
            raw, capabilities=AuditCapabilities.all("zelda_i.pre_l1_c3_trace")
        )
        if args.headed:
            pygame_mod = attach_headed(
                env,
                title=f"Zelda I pre-L1 {args.arm}",
                hud=_hud,
            )
        obs, boot = boot_to_ready(env, first_playthrough=True, assist=None)
        sword = SwordCaveController()
        obs, sword_res = run_controller_stage(
            env, obs, name="sword_cave", controller=sword,
            max_frames=SWORD_MAX, assist=None, frame_base=boot,
        )
        if not sword.success:
            raise SystemExit("sword failed")
        frame_base = int(sword_res.end_frame)
        predecessor = _link_row(read_snapshot(env.get_ram()))
        obs, walk_res = run_controller_stage(
            env, obs, name="bomb_walk", controller=walk,
            max_frames=SHOP_P7_WALK_MAX_FRAMES, assist=None,
            frame_base=sword_res.end_frame,
        )
        snap = read_snapshot(env.get_ram())
        if (
            int(snap.screen) in FOCUS_SCREENS
            and int(snap.link_iframes) > 0
            and prior_iframes == 0
        ):
            # A fatal contact ends ``run_controller_stage`` before the controller
            # is called again, so the ordinary pre-action hook cannot see it.
            capture(snap, tracker.observe(snap), "terminal", ["hit", "death"])
        final_path = _png_path(args.tag, len(evidence), "final", snap)
        save_rgb_png(obs, final_path)
        audit = env.audit()
        path = OUT / f"{args.tag}.json"
        path.write_text(json.dumps({
            "tag": args.tag,
            "arm": args.arm,
            "assist": None,
            "allow_pokes": False,
            "set_state_count": int(audit.mid_run_loads or 0),
            "ram_writes": int(audit.ram_writes or 0),
            "predecessor": predecessor,
            "end_screen": f"0x{int(snap.screen):02X}",
            "mode": int(snap.mode),
            "rupees": int(snap.rupees),
            "frames": int(walk.frames),
            "notes": list(walk.notes),
            "evidence": evidence,
            "final": _link_row(snap),
            "final_screenshot": str(final_path),
            "controller": walk.report(),
            "trace": trace,
        }, indent=2) + "\n")
        print(json.dumps({
            "arm": args.arm,
            "screen": f"0x{int(snap.screen):02X}", "rupees": int(snap.rupees),
            "frames": int(walk.frames), "traced": len(trace),
            "evidence": len(evidence), "set_state": int(audit.mid_run_loads or 0),
            "ram_writes": int(audit.ram_writes or 0), "out": str(path),
        }), flush=True)
    finally:
        if pygame_mod is not None:
            try:
                idle_headed(env, pygame_mod)
            except KeyboardInterrupt:
                pass
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
