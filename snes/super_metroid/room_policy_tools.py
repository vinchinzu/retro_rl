"""Compile and verify reactive room policies from live hop anchors."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from super_metroid.dev.common import boot_from_state, make_dev_env
from super_metroid.human_tape.replay import check_hop_green, resolve_assist
from super_metroid.ram import GameplayPhase, parse_env_state
from super_metroid.reactive_policy import (
    PolicyVariant,
    ReactivePolicyRunner,
    ReactiveRoomPolicy,
    ReferenceSample,
    ReferenceTrajectory,
    default_variant_contract,
)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_button_frames(path: Path | str) -> tuple[list[list[int]], dict[str, Any]]:
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    frames = data.get("frames") or data.get("raw_buttons")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"no frames/raw_buttons in {source}")
    out: list[list[int]] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, list) or len(frame) != 12:
            raise ValueError(f"invalid SNES-12 frame {index} in {source}")
        values = [int(v) for v in frame]
        if any(v not in (0, 1) for v in values):
            raise ValueError(f"non-binary frame {index} in {source}")
        out.append(values)
    meta = dict(data.get("meta") or data.get("metadata") or {})
    return out, meta


def task_hop_frames(
    task_path: Path | str,
    hop_index: int,
    *,
    transition_tail: int = 180,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Extract a settled hop plus enough tail to observe the destination room."""
    from super_metroid.human_tape.hops import load_task_json, resolve_hop_slice

    task = Path(task_path)
    data = load_task_json(task)
    resolved = resolve_hop_slice(
        task,
        hop_index=int(hop_index),
        leave_extra=max(0, int(transition_tail)),
        task_data=data,
        settle=True,
    )
    hop = dict(resolved.get("hop") or {})
    start = int(resolved["replay_start"])
    end = int(resolved["end_index"]) + 1
    frames = [list(map(int, frame)) for frame in data["frames"][start:end]]
    meta = {
        "source_task": str(task),
        "hop_index": int(hop_index),
        "room": hop.get("room"),
        "name": hop.get("name"),
        "start_index": int(resolved["start_index"]),
        "replay_start": start,
        "end_index": end - 1,
        "entry_anchor": resolved.get("anchor_path"),
        "end_xy": hop.get("end_xy"),
        "items": hop.get("items"),
    }
    return frames, meta


def capture_reference_trajectory(
    entry_anchor: Path | str,
    frames: Sequence[Sequence[int]],
    *,
    trajectory_id: str,
    room_id: int,
    exit_room_id: int,
    assist: bool | Any = True,
    boot_settle: int = 0,
    source: Mapping[str, Any] | None = None,
    env: Any | None = None,
    max_span_frames: int = 8,
    exit_tail_frames: int = 180,
) -> tuple[ReferenceTrajectory, dict[str, Any]]:
    """Replay an expert hop into sparse, timed kinematic checkpoints.

    Equal actions are RLE-compressed, with a bounded span so feedback runs at
    least every ``max_span_frames`` during normal playback.
    """
    anchor = Path(entry_anchor)
    if not anchor.is_file():
        raise FileNotFoundError(anchor)
    owns_env = env is None
    if env is None:
        env = make_dev_env()
    assist_obj = resolve_assist(assist)
    source_frame_count = len(frames)
    replay_frames = [list(map(int, frame)) for frame in frames]
    replay_frames.extend(
        [[0] * 12 for _ in range(max(0, int(exit_tail_frames)))]
    )
    samples: list[ReferenceSample] = []
    start = None
    state = None
    left_at: int | None = None
    try:
        state = boot_from_state(env, anchor, settle_frames=boot_settle)
        start = state
        if int(state.room_id) != int(room_id):
            raise ValueError(
                f"anchor room 0x{int(state.room_id):04X} != 0x{int(room_id):04X}"
            )
        frame_index = 0
        span_limit = max(1, int(max_span_frames))
        while frame_index < len(replay_frames):
            raw = replay_frames[frame_index]
            state = parse_env_state(env, frame=frame_index, mode="nav")
            if int(state.room_id) != int(room_id):
                left_at = frame_index
                break
            span_start = state
            span_frames = 0
            while (
                frame_index < len(replay_frames)
                and span_frames < span_limit
                and replay_frames[frame_index] == list(raw)
            ):
                env.step(np.asarray(raw, dtype=np.int8))
                frame_index += 1
                span_frames += 1
                state = parse_env_state(env, frame=frame_index, mode="nav")
                if assist_obj is not None:
                    assist_obj.apply(env.data, state)
                if int(state.room_id) == int(exit_room_id):
                    left_at = frame_index
                    break
            samples.append(
                ReferenceSample.from_state(
                    span_start,
                    raw,
                    frames=span_frames,
                )
            )
            if int(state.room_id) == int(exit_room_id):
                break
        state = parse_env_state(env, mode="nav")
    finally:
        if owns_env:
            env.close()

    if not samples or start is None or state is None:
        raise RuntimeError("reference capture produced no samples")
    provenance = {
        "entryAnchor": str(anchor),
        "entryAnchorSha256": sha256_file(anchor),
        "inputFrames": source_frame_count,
        "exitTailFrames": max(0, int(exit_tail_frames)),
        "capturedSamples": len(samples),
        **dict(source or {}),
    }
    trajectory = ReferenceTrajectory(
        trajectory_id=trajectory_id,
        samples=tuple(samples),
        source=provenance,
    )
    report = {
        "ok": int(state.room_id) == int(exit_room_id),
        "left_at": left_at,
        "input_frames": source_frame_count,
        "replay_frames": left_at if left_at is not None else len(replay_frames),
        "samples": len(samples),
        "start_room": int(start.room_id),
        "start_xy": [int(start.samus_x), int(start.samus_y)],
        "items": int(start.collected_items),
        "hi_jump": bool(start.hi_jump),
        "room_id": int(state.room_id),
        "room": f"0x{int(state.room_id):04X}",
        "xy": [int(state.samus_x), int(state.samus_y)],
        "pose": int(state.pose),
        "phase": state.phase.value,
    }
    return trajectory, report


def merge_policy_variant(
    policy: ReactiveRoomPolicy | None,
    *,
    policy_id: str,
    route_id: str,
    room_id: int,
    from_room_id: int | None,
    exit_room_id: int,
    variant_id: str,
    trajectory: ReferenceTrajectory,
    required_items: int | None = None,
    forbidden_items: int | None = None,
) -> ReactiveRoomPolicy:
    """Add/replace one named trajectory without disturbing other variants."""
    default_required, default_forbidden = default_variant_contract(variant_id)
    req = default_required if required_items is None else int(required_items)
    forbid = default_forbidden if forbidden_items is None else int(forbidden_items)
    variants = list(policy.variants if policy is not None else ())
    matched = False
    for index, variant in enumerate(variants):
        if variant.variant_id != variant_id:
            continue
        trajectories = [
            value
            for value in variant.trajectories
            if value.trajectory_id != trajectory.trajectory_id
        ]
        trajectories.append(trajectory)
        variants[index] = replace(
            variant,
            trajectories=tuple(trajectories),
            required_items=req,
            forbidden_items=forbid,
        )
        matched = True
        break
    if not matched:
        variants.append(
            PolicyVariant(
                variant_id=variant_id,
                trajectories=(trajectory,),
                required_items=req,
                forbidden_items=forbid,
            )
        )
    previous_meta = dict(policy.meta) if policy is not None else {}
    verified_variants = dict(previous_meta.get("verifiedVariants") or {})
    verified_variants.pop(variant_id, None)
    previous_meta["verifiedVariants"] = verified_variants
    return ReactiveRoomPolicy(
        policy_id=policy_id,
        route_id=route_id,
        room_id=int(room_id),
        from_room_id=from_room_id,
        exit_room_id=int(exit_room_id),
        variants=tuple(variants),
        status="candidate",
        meta=previous_meta,
    )


def replay_reactive_policy(
    policy: ReactiveRoomPolicy,
    entry_anchor: Path | str,
    *,
    prelude_frames: Sequence[Sequence[int]] = (),
    max_frames: int = 10_000,
    assist: bool | Any = True,
    boot_settle: int = 0,
    use_adapter: bool = False,
    env: Any | None = None,
) -> dict[str, Any]:
    """Boot one live anchor and run feedback control until its destination."""
    owns_env = env is None
    if env is None:
        env = make_dev_env()
    assist_obj = resolve_assist(assist)
    adapter_frames: list[tuple[int, ...]] = []
    try:
        state = boot_from_state(env, Path(entry_anchor), settle_frames=boot_settle)
        if int(state.room_id) != policy.room_id:
            raise ValueError(
                f"anchor room 0x{int(state.room_id):04X} != policy room "
                f"0x{policy.room_id:04X}"
            )
        prelude_stepped = 0
        for raw in prelude_frames:
            if int(state.room_id) != policy.room_id:
                break
            env.step(np.asarray(raw, dtype=np.int8))
            prelude_stepped += 1
            state = parse_env_state(env, frame=prelude_stepped, mode="nav")
            if assist_obj is not None:
                assist_obj.apply(env.data, state)

        variant = policy.select_variant(int(state.collected_items))
        if variant is None:
            raise ValueError(
                f"no {policy.policy_id} variant for items 0x{int(state.collected_items):04X}"
            )
        runner = ReactivePolicyRunner(variant)
        target = runner.resume(state)
        if use_adapter and target.score > variant.rejoin_threshold:
            from super_metroid.room_adapter import search_live_adapter

            adapter_frames = list(search_live_adapter(env, runner).frames)

        stepped = 0
        started = time.perf_counter()
        for raw in adapter_frames:
            env.step(np.asarray(raw, dtype=np.int8))
            stepped += 1
            state = parse_env_state(env, frame=stepped, mode="nav")
            if assist_obj is not None:
                assist_obj.apply(env.data, state)

        if adapter_frames:
            runner.resume(parse_env_state(env, mode="nav"))
        for _ in range(max(0, int(max_frames) - stepped)):
            state = parse_env_state(env, frame=stepped, mode="nav")
            if int(state.room_id) == policy.exit_room_id:
                break
            if (
                int(state.room_id) != policy.room_id
                and state.phase is GameplayPhase.ORDINARY_GAMEPLAY
            ):
                break
            action = (
                runner.continue_action()
                if runner.has_held_action
                else runner.action(state)
            )
            env.step(action)
            stepped += 1
            state = parse_env_state(env, frame=stepped, mode="nav")
            if assist_obj is not None:
                assist_obj.apply(env.data, state)
        state = parse_env_state(env, frame=stepped, mode="nav")
        elapsed = max(time.perf_counter() - started, 1e-9)
        return {
            "ok": int(state.room_id) == policy.exit_room_id,
            "room_id": int(state.room_id),
            "room": f"0x{int(state.room_id):04X}",
            "xy": [int(state.samus_x), int(state.samus_y)],
            "pose": int(state.pose),
            "phase": state.phase.value,
            "frames": stepped,
            "prelude_frames": prelude_stepped,
            "autopilot_frames": stepped,
            "takeover_score": target.score,
            "wall_seconds": elapsed,
            "fps": stepped / elapsed,
            "variant": variant.variant_id,
            "trajectory": runner.trajectory.trajectory_id,
            "adapter_frames": len(adapter_frames),
            "runner": runner.status(),
        }
    finally:
        if owns_env:
            env.close()


def verify_takeover_sweep(
    policy: ReactiveRoomPolicy,
    entry_anchor: Path | str,
    seed_frames: Sequence[Sequence[int]],
    *,
    takeover_points: Sequence[int],
    perturb_frames: int = 0,
    max_frames: int = 10_000,
    assist: bool | Any = True,
    use_adapter: bool = True,
) -> dict[str, Any]:
    """Verify human→autopilot joins at several live frames.

    ``perturb_frames`` appends idle input after each expert prefix.  That makes
    position, velocity, acceleration/momentum, pose, and enemy timing differ
    from the captured checkpoint without writing synthetic WRAM values.
    """
    env = make_dev_env()
    runs: list[dict[str, Any]] = []
    try:
        for raw_point in takeover_points:
            point = max(0, min(int(raw_point), len(seed_frames)))
            prelude = [list(map(int, row)) for row in seed_frames[:point]]
            prelude.extend([[0] * 12 for _ in range(max(0, int(perturb_frames)))])
            result = replay_reactive_policy(
                policy,
                entry_anchor,
                prelude_frames=prelude,
                max_frames=max_frames,
                assist=assist,
                use_adapter=use_adapter,
                env=env,
            )
            result["takeover_point"] = point
            result["perturb_frames"] = max(0, int(perturb_frames))
            runs.append(result)
    finally:
        env.close()
    return {
        "ok": bool(runs) and all(bool(row.get("ok")) for row in runs),
        "policy_id": policy.policy_id,
        "anchor": str(entry_anchor),
        "runs": runs,
    }


def verify_reactive_policy(
    policy: ReactiveRoomPolicy,
    entry_anchor: Path | str,
    *,
    dual: bool = True,
    max_frames: int = 10_000,
    assist: bool | Any = True,
    use_adapter: bool = False,
) -> dict[str, Any]:
    """Dual-green verifier used before status/bank promotion."""
    env = make_dev_env()
    runs: list[dict[str, Any]] = []
    try:
        for _ in range(2 if dual else 1):
            runs.append(
                replay_reactive_policy(
                    policy,
                    entry_anchor,
                    max_frames=max_frames,
                    assist=assist,
                    use_adapter=use_adapter,
                    env=env,
                )
            )
    finally:
        env.close()
    check = check_hop_green(
        runs if dual else runs[0],
        policy.exit_room_id,
        dual=dual,
        start_room=policy.room_id,
    )
    return {
        "ok": bool(check.get("ok")),
        "green": bool(check.get("ok")),
        "dual": dual,
        "policy_id": policy.policy_id,
        "anchor": str(entry_anchor),
        "check": check,
        "runs": runs,
    }


def mark_verified(
    policy: ReactiveRoomPolicy,
    report: Mapping[str, Any],
) -> ReactiveRoomPolicy:
    if not report.get("green") or not report.get("dual"):
        raise ValueError("reactive policy promotion requires dual green")
    runs = list(report.get("runs") or [])
    frames = [int(row.get("frames", 0)) for row in runs]
    variants = {str(row.get("variant")) for row in runs if row.get("variant")}
    if len(variants) != 1:
        raise ValueError(f"verification must select one stable variant, got {variants}")
    variant_id = variants.pop()
    meta = dict(policy.meta)
    verification = {
        "dualGreen": True,
        "frames": frames,
        "anchor": report.get("anchor"),
        "fps": [float(row.get("fps", 0.0)) for row in runs],
    }
    verified_variants = dict(meta.get("verifiedVariants") or {})
    verified_variants[variant_id] = verification
    meta["verifiedVariants"] = verified_variants
    meta["verification"] = verification
    all_verified = all(
        variant.variant_id in verified_variants for variant in policy.variants
    )
    status = "verified_live_anchor" if all_verified else "candidate"
    return replace(policy, status=status, meta=meta)


def mark_takeovers_verified(
    policy: ReactiveRoomPolicy,
    report: Mapping[str, Any],
) -> ReactiveRoomPolicy:
    if not report.get("ok"):
        raise ValueError("cannot record a red takeover sweep")
    runs = list(report.get("runs") or [])
    variants = {str(row.get("variant")) for row in runs if row.get("variant")}
    if len(variants) != 1:
        raise ValueError(f"takeover sweep must use one stable variant, got {variants}")
    variant_id = variants.pop()
    compact = [
        {
            "takeoverFrame": int(row.get("takeover_point", 0)),
            "perturbFrames": int(row.get("perturb_frames", 0)),
            "autopilotFrames": int(row.get("autopilot_frames", 0)),
            "fps": float(row.get("fps", 0.0)),
            "adapterFrames": int(row.get("adapter_frames", 0)),
            "room": row.get("room"),
        }
        for row in runs
    ]
    meta = dict(policy.meta)
    takeover_verification = dict(meta.get("takeoverVerification") or {})
    takeover_verification[variant_id] = {
        "green": True,
        "anchor": report.get("anchor"),
        "runs": compact,
    }
    meta["takeoverVerification"] = takeover_verification
    return replace(policy, meta=meta)


__all__ = [
    "capture_reference_trajectory",
    "load_button_frames",
    "mark_takeovers_verified",
    "mark_verified",
    "merge_policy_variant",
    "replay_reactive_policy",
    "sha256_file",
    "task_hop_frames",
    "verify_reactive_policy",
    "verify_takeover_sweep",
]
