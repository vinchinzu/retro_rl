"""Measure the opening spine, then replay its committed path without restores.

The controller is development evidence. Default start is a core reset (NONE),
with blank SRAM prepared before the measured attempt. The controller still
searches by restoring states; its report must not be quoted as Clean.
The separate replay verifies every frame against full-RAM CRCs. A completed
no-assist replay with no loads, writes or deaths is Clean proof.

Run with a frozen source tree on PYTHONPATH when other agents are editing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from alttp.opening_route.full_tip import run_to_verified_tip
from alttp.opening_route.segment import get_segment
from alttp.paths import INTEGRATION_DIR
from alttp.ram import LINK_HP, LINK_MAX_HP, SRAM_PROGRESS, SRAM_PROGRESS_FLAGS, read_snapshot, snapshot_to_diag, wram_index
from alttp.scripts.replay_tape import _CommittedRecorder, _refuse_verified_tip_run
from alttp.spine.hops import hop_for, hops_through
from alttp.spine.tape import LiveRun, ram_crc, write_pin_provenance
from alttp.startup import build_blank_sram, build_boot_env, inject_sram_into_env
from retro_harness.env import write_state_bytes
from retro_harness.segment_runner import configure_headless


def story_glance(env: Any) -> dict[str, Any]:
    """Distinguish Sanctuary arrival from the completed Zelda/priest dialogue."""
    ram = env.get_ram()
    snap = read_snapshot(ram)
    progress = int(ram[wram_index(SRAM_PROGRESS)])
    flags = int(ram[wram_index(SRAM_PROGRESS_FLAGS)])
    return {
        "progress": progress, "progress_flags": flags,
        "map_indicator": int(ram[wram_index(0xF3C7)]),
        "follower": snap.follower, "max_hp": int(ram[wram_index(LINK_MAX_HP)]),
        "sanctuary_completed": (snap.in_sanctuary and snap.has_control
                                and progress == 2 and bool(flags & 4) and snap.follower == 0),
    }


def replay_goal_accepted(meta: dict[str, Any], final: dict[str, Any], story: dict[str, Any]) -> bool:
    """A synchronized failed controller tape is not a completed route."""
    if not meta.get("source_ok"):
        return False
    through = meta.get("through")
    if through == "sanctuary":
        return bool(story["sanctuary_completed"])
    if through == "sanctuary_heart":
        return bool(story.get("sanctuary_completed") and story.get("max_hp") == 32 and final.get("room_base_id") == 18)
    if through == "kakariko":
        standing = int(final["link_y"]) // 512 * 8 + int(final["link_x"]) // 512
        return bool(
            final["has_control"] and not final["indoors"] and not final["dark_world"]
            and int(final["screen_id"]) == 0x18 and standing in (0x18, 0x19, 0x20, 0x21)
            and final["has_fighter_sword"] and final["has_lamp"] and final["hp"] > 0
            and story["follower"] == 0 and story["progress"] == 2
            and story["progress_flags"] & 4 and story["max_hp"] == 32
        )
    if through in ("sanctuary_exit", "eastern_grounds", "sahasrahla_hut", "eastern_approach"):
        screen = 0x13 if through == "sanctuary_exit" else 0x1E
        target_standing = screen
        if through == "eastern_approach":
            from alttp.room_map import load_room_map

            point = load_room_map("screen_1e_eastern").point("eastern_door")
            if point is None or story.get("map_indicator") != 3:
                return False
            if max(abs(int(final[key]) - value) for key, value in
                   zip(("link_x", "link_y"), (point.x, point.y))) > 4:
                return False
            target_standing = point.y // 512 * 8 + point.x // 512
        if through == "sahasrahla_hut":
            from alttp.room_map import load_room_map

            door = load_room_map("room_05").door("south_to_0x1e")
            if door is None or door.landing_xy is None or story.get("map_indicator") != 3:
                return False
            if max(abs(int(final[key]) - value) for key, value in
                   zip(("link_x", "link_y"), door.landing_xy)) > 4:
                return False
        standing = int(final["link_y"]) // 512 * 8 + int(final["link_x"]) // 512
        return bool(
            final["has_control"] and not final["indoors"] and not final["dark_world"]
            and int(final["screen_id"]) == screen and standing == target_standing
            and final["has_fighter_sword"] and final["has_lamp"] and final["hp"] > 0
            and story["follower"] == 0 and story["progress"] == 2
            and story["progress_flags"] & 4 and story["max_hp"] == 32
        )
    if through == "zelda":
        return bool(final["has_control"] and final["in_zelda_cell"] and final["has_zelda_follower"])
    hop = hop_for(through)
    if hop is None or not hop.success.startswith("room_"):
        return False
    expected = int(hop.success.removeprefix("room_"), 16)
    return bool(final["has_control"] and final["indoors"] and final["room_base_id"] == expected
                and final["has_fighter_sword"])


class Survival:
    """Refill existing HP capacity at <= one heart; never resurrect Link."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.events: list[dict[str, Any]] = []

    def after_step(self, env: Any, frame: int) -> None:
        if not self.enabled:
            return
        ram = env.get_ram()
        snap = read_snapshot(ram)
        maximum = int(ram[wram_index(LINK_MAX_HP)])
        if (snap.game_mode not in (7, 9) or not 0 < snap.hp <= 8
                or not 8 < maximum <= 160 or maximum % 8):
            return
        env.data.memory.assign(0x7E0000 + LINK_HP, "|u1", maximum)
        env.data.update_ram()
        assert read_snapshot(env.get_ram()).hp == maximum
        self.events.append({
            "frame": frame, "room": snap.room_id, "from": snap.hp,
            "to": maximum, "hearts_restored": (maximum - snap.hp) / 8,
        })

    def report(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled, "active_from_frame": 1 if self.enabled else None,
            "write_count": len(self.events),
            "hearts_restored": sum(e["hearts_restored"] for e in self.events),
            "first_refill_frame": self.events[0]["frame"] if self.events else None,
            "progression_writes": 0, "capacity_writes": 0, "events": self.events,
        }


class Recorder(_CommittedRecorder):
    """Reuse the production committed-tape recorder; CRC includes HP assist."""

    def __init__(self, raw: Any, survival: Survival) -> None:
        super().__init__(raw)
        self.survival = survival
        self.live: Any = None
        self.steps_executed = 0

    def _step(self, action: Any, *args: Any, **kwargs: Any) -> Any:
        out = super()._step(action, *args, **kwargs)
        self.steps_executed += 1
        self.survival.after_step(self.live, len(self.buttons))
        self.crcs[-1] = ram_crc(self.raw.get_ram())
        return out

    def _set_state(self, state: Any) -> Any:
        out = super()._set_state(state)
        keep = len(self.buttons)
        self.survival.events[:] = [e for e in self.survival.events if e["frame"] <= keep]
        return out


class Stage:
    def __init__(self, segment: Any, rows: list[dict[str, Any]], rec: Recorder) -> None:
        self.segment, self.rows, self.rec = segment, rows, rec

    def __getattr__(self, name: str) -> Any:
        return getattr(self.segment, name)

    def play_checked(self, env: Any, **kwargs: Any) -> Any:
        start = len(self.rec.buttons)
        print(f"stage={self.segment.id} start={start}", flush=True)
        result = self.segment.play_checked(env, **kwargs)
        row = {"stage": self.segment.id, "start": start, "end": len(self.rec.buttons),
               "ok": result.ok, "blocker": result.blocker,
               "final": snapshot_to_diag(result.snapshot)}
        self.rows.append(row)
        print(f"stage={self.segment.id} ok={result.ok} end={row['end']} "
              f"room={result.snapshot.room_id:#x} blocker={result.blocker!r}", flush=True)
        return result


def play_verified_prefix(env: Any, tape: Path, through: str) -> tuple[str, ...]:
    """Replay a Clean power-on input prefix, then return the remaining rows.

    The tape supplies inputs, never a state. Validate the route relationship,
    every full-RAM CRC and the recorded leave before dispatching any new leg.
    """
    with np.load(tape, allow_pickle=False) as data:
        buttons, crcs = data["buttons"], data["frame_crc"]
        meta = json.loads(str(data["meta"]))
    prefix = tuple(s for h in hops_through(meta.get("through")) for s in h.stages)
    requested = tuple(s for h in hops_through(through) for s in h.stages)
    if (meta.get("start") != "NONE" or meta.get("survival") is not False
            or not meta.get("source_ok") or not prefix or len(prefix) >= len(requested)
            or requested[:len(prefix)] != prefix):
        raise ValueError("prefix must be a successful no-assist power-on tape for earlier spine rows")
    rom_hash = hashlib.sha256((INTEGRATION_DIR / "rom.sfc").read_bytes()).hexdigest()
    if meta.get("rom_sha256") != rom_hash:
        raise ValueError("prefix ROM differs from recording or has no recorded hash")
    if buttons.ndim != 2 or buttons.shape[1] != 12 or len(buttons) != len(crcs) or not len(buttons):
        raise ValueError("prefix requires one full-RAM CRC per 12-button frame")
    env.reset()
    for frame, (action, crc) in enumerate(zip(buttons, crcs), 1):
        env.step(action.astype(np.int8))
        snap = read_snapshot(env.get_ram())
        if ram_crc(env.get_ram()) != int(crc):
            raise RuntimeError(f"prefix desync frame {frame}; suffix not played")
        if snap.game_mode == 0x12 or (snap.game_mode in (7, 9) and snap.hp == 0):
            raise RuntimeError(f"prefix death frame {frame}; suffix not played")
    final, story = snapshot_to_diag(snap), story_glance(env)
    if (not all(final.get(k) == v for k, v in meta["source_final"].items())
            or not all(story.get(k) == v for k, v in meta.get("source_story", {}).items())
            or not replay_goal_accepted(meta, final, story)):
        raise RuntimeError("prefix leave rejected; suffix not played")
    print(f"prefix={meta['through']} verified_frames={len(buttons)}", flush=True)
    return requested[len(prefix):]


def controller(args: Any) -> dict[str, Any]:
    from alttp.rollout import restores

    raw = build_boot_env(state_name=args.start)
    survival = Survival(args.survival)
    rec = Recorder(raw, survival)
    live = LiveRun(rec.attach(), provider="alttp.power_on_audit")
    rec.live = live.env
    start_restores = restores()
    stages: list[dict[str, Any]] = []
    hops = hops_through(args.through)
    if not hops:
        raw.close()
        raise ValueError("audit requires a runnable hop")
    try:
        lookup = lambda sid: Stage(get_segment(sid), stages, rec)
        prefix_tape = getattr(args, "prefix_tape", None)
        if prefix_tape is not None:
            from alttp.scripts.run_spine import play_from_stages

            if args.survival or args.start != "NONE":
                raise ValueError("a verified prefix requires NONE and no Survival assists")
            remaining = play_verified_prefix(live.env, prefix_tape, args.through)
            prefix_frames = len(rec.buttons)
            result = play_from_stages(
                live.env, remaining, resume_stage=remaining[0], success=hops[-1].success,
                get_segment_fn=lookup, source="power_on_tape_prefix",
            )
        else:
            result = run_to_verified_tip(
                live.env, close=False,
                segment_ids=tuple(s for h in hops for s in h.stages),
                success=hops[-1].success, get_segment_fn=lookup,
            )
        live.note_rollout_restores(restores() - start_restores)
        audit = asdict(live.report())
        report = result.to_report()
        if result.ok and args.edge:
            from alttp.opening_route.room_engine import run_room_edge

            map_id, door = args.edge.split(":", 1)
            extension = run_room_edge(
                live.env, map_id, door, clear=False, source="power_on_assisted_audit",
            )
            report["extension"] = {
                "map": map_id, "door": door, "ok": extension.ok,
                "blocker": extension.blocker, "frames": extension.frames,
                "final": snapshot_to_diag(extension.snapshot),
            }
            report["spine_goal_ok"] = result.ok
            report["ok"] = extension.ok
            report["blocker"] = extension.blocker
            report["final"] = snapshot_to_diag(extension.snapshot)
            live.note_rollout_restores(restores() - start_restores)
            audit = asdict(live.report())
        # The production report labels a restore-search controller Clean.
        # Override that label here; only the separate replay can prove a path.
        report.update({
            "clean_chain": False, "development_only": True, "source": "controller_search",
            "intervention": "Survival" if args.survival else "search restores",
            "notes": ["Development controller search. Not Clean or continuous evidence."],
            "audit": audit, "actual_search_restores": rec.restores,
            "suppressed_duplicate_sram_setup": rec.swallowed,
            "steps_executed": rec.steps_executed, "committed_frames": len(rec.buttons),
            "search_frames_omitted": rec.omitted, "survival": survival.report(),
            "stage_splits": stages,
        })
        report["runner_reported_frames"] = report["frames"]
        report["frames"] = len(rec.buttons)
        if prefix_tape is not None:
            report["prefix_tape"] = str(prefix_tape)
            report["prefix_frames"] = prefix_frames
        for row in report["segments"]:
            row.update(source="controller_search", cleanChain=False, developmentOnly=True)
        report["story"] = story_glance(raw)
        meta = {"start": args.start, "through": args.through, "survival": args.survival,
                "source_ok": report["ok"], "source_final": report["final"], "stages": stages,
                "source_story": report["story"],
                "prefix_tape": str(prefix_tape) if prefix_tape is not None else None,
                "rom_sha256": hashlib.sha256((INTEGRATION_DIR / "rom.sfc").read_bytes()).hexdigest()}
        np.savez_compressed(
            args.out.with_suffix(".tape.npz"), buttons=np.stack(rec.buttons),
            frame_crc=np.asarray(rec.crcs, dtype=np.uint32), meta=json.dumps(meta),
        )
        save_end(raw, args.out, len(rec.buttons))
        return report
    finally:
        raw.close()


def replay(args: Any) -> dict[str, Any]:
    with np.load(args.tape, allow_pickle=False) as data:
        buttons, crcs = data["buttons"], data["frame_crc"]
        meta = json.loads(str(data["meta"]))
    if meta["start"] not in ("NONE", "YazeSlot000"):
        raise ValueError("replay start is not a boot condition")
    if meta.get("rom_sha256") and meta["rom_sha256"] != hashlib.sha256(
        (INTEGRATION_DIR / "rom.sfc").read_bytes()
    ).hexdigest():
        raise ValueError("replay ROM differs from recording")
    captures: dict[int, list[str]] = {}
    for spec in getattr(args, "capture", ()):
        frame, sep, name = spec.partition(":")
        if (not sep or not frame.isdigit() or not 1 <= int(frame) <= len(buttons)
                or not re.fullmatch(r"Dev[A-Za-z0-9_]+", name)):
            raise ValueError("capture requires FRAME:DevName inside the tape")
        captures.setdefault(int(frame), []).append(name)
    raw = build_boot_env(state_name=meta["start"])
    survival = Survival(meta["survival"])
    try:
        raw.reset()
        inject_sram_into_env(raw, build_blank_sram())
        live = LiveRun(raw, provider="alttp.power_on_forward_replay")
        first = None
        desynced = 0
        rooms: list[dict[str, Any]] = []
        previous = None
        hits: list[dict[str, Any]] = []
        health_gains: list[dict[str, Any]] = []
        death_frames: list[int] = []
        prev_hp = None
        for index, action in enumerate(buttons):
            live.env.step(action.astype(np.int8))
            snap = read_snapshot(raw.get_ram())
            if snap.game_mode in (7, 9):
                if prev_hp is not None and snap.hp < prev_hp:
                    hits.append({"frame": index + 1, "room": snap.room_id,
                                 "damage": prev_hp - snap.hp})
                if prev_hp is not None and snap.hp > prev_hp:
                    health_gains.append({"frame": index + 1, "room": snap.room_id,
                                         "hp_gained": snap.hp - prev_hp})
                if snap.hp == 0:
                    death_frames.append(index + 1)
                prev_hp = snap.hp
            survival.after_step(live.env, index + 1)
            snap = read_snapshot(raw.get_ram())
            if snap.game_mode in (7, 9):
                prev_hp = snap.hp
            location = (snap.indoors, snap.room_id if snap.indoors else snap.screen_id)
            if snap.has_control and location != previous:
                rooms.append({"frame": index + 1, "indoors": snap.indoors,
                              "location": location[1], "xy": [snap.link_x, snap.link_y],
                              "hp": snap.hp, "sword": snap.sword_level,
                              "lamp": snap.lamp_level, "keys": snap.num_keys,
                              "follower": snap.follower})
                previous = location
            if ram_crc(raw.get_ram()) != int(crcs[index]):
                desynced += 1
                if first is None:
                    first = index + 1
            for name in captures.get(index + 1, ()):
                if desynced:
                    raise RuntimeError("refusing to capture a desynchronized predecessor")
                path = INTEGRATION_DIR / f"{name}.state"
                write_state_bytes(path, raw.em.get_state())
                write_pin_provenance(path, source_run=str(args.out), frame=index + 1, kind="scan")
        final = snapshot_to_diag(read_snapshot(raw.get_ram()))
        story = story_glance(raw)
        audit = asdict(live.report())
        if audit["set_state"] or audit["writes"] != len(survival.events):
            raise RuntimeError("forward replay has an unexpected load or write")
        save_end(raw, args.out, len(buttons))
        # New glance fields must not invalidate older tapes. Every recorded
        # field still has to agree, and each route checks its own goal fields.
        synchronization_ok = (desynced == 0
                              and all(final.get(k) == v for k, v in meta["source_final"].items())
                              and all(story.get(k) == v for k, v in meta.get("source_story", {}).items()))
        route_goal_ok = replay_goal_accepted(meta, final, story)
        clean = (synchronization_ok and route_goal_ok and not death_frames
                 and not meta["survival"] and not audit["set_state"]
                 and not audit["writes"])
        return {
            "ok": synchronization_ok and route_goal_ok,
            "synchronization_ok": synchronization_ok,
            "start": meta["start"],
            "route_goal_ok": route_goal_ok, "through": meta.get("through"), "frames": len(buttons),
            "desynced": desynced, "first_desync_frame": first, "final": final,
            "audit": audit, "survival": survival.report(), "rooms": rooms, "story": story,
            "hits": hits, "death_frames": death_frames,
            "in_game_health_gains": health_gains,
            "captured_predecessors": captures,
            "clean_chain": clean, "source": "power_on_forward_replay",
            "intervention": "Survival" if meta["survival"] else "Clean",
            "development_only": not clean,
        }
    finally:
        raw.close()


def save_end(env: Any, output: Path, frame: int) -> None:
    Image.fromarray(np.asarray(env.render())).save(output.with_suffix(".png"))
    # Captures never get loaded by this runner. States stay in the integration.
    path = INTEGRATION_DIR / f"DevAudit_{output.stem}.state"
    write_state_bytes(path, env.em.get_state())
    write_pin_provenance(path, source_run=str(output), frame=frame, kind="scan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("controller", "replay"))
    parser.add_argument("--start", choices=("NONE", "YazeSlot000"), default="NONE")
    parser.add_argument("--through", default="room_81")
    parser.add_argument("--survival", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tape", type=Path)
    parser.add_argument("--prefix-tape", type=Path,
                        help="Controller only: replay a verified Clean power-on prefix before new rows")
    parser.add_argument("--edge", help="After the spine, try one mapped map_id:door edge")
    parser.add_argument("--capture", action="append", default=[], metavar="FRAME:DevName",
                        help="Replay only: save a CRC-checked development predecessor")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "replay" and args.tape is None:
        parser.error("replay requires --tape")
    if args.capture and args.mode != "replay":
        parser.error("--capture requires replay mode")
    if args.prefix_tape and args.mode != "controller":
        parser.error("--prefix-tape requires controller mode")
    if args.edge and (":" not in args.edge or not all(args.edge.split(":", 1))):
        parser.error("--edge requires map_id:door")
    _refuse_verified_tip_run(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    configure_headless()
    start = time.perf_counter()
    report = controller(args) if args.mode == "controller" else replay(args)
    report.update({
        "wall_seconds": time.perf_counter() - start,
        "start": args.start if args.mode == "controller" else report["start"],
        "start_setup": "reset, then blank SRAM, before frame 1",
        "contract": "snes/alttp/docs/ASSIST_CONTRACT.md",
        "rom_sha256": hashlib.sha256((INTEGRATION_DIR / "rom.sfc").read_bytes()).hexdigest(),
        "runtime_source": str(Path(__import__("alttp").__file__).parent),
    })
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report.get(k) for k in (
        "ok", "frames", "committed_frames", "blocker", "final", "audit",
        "survival", "desynced", "wall_seconds")}, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
