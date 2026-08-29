"""Showcase stitch + video render (not the A/B loop).

Sources: ``playthrough`` (one session, death-verified), ``legal_stitch``
(leaderboard PBs, may desync), ``optimizer`` (hillclimb recordings).

    uv run python -m smb.scripts.render_full_run --route warp
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from retro_harness.env import read_state_bytes
from smb.paths import (
    FULLGAME_RECORDINGS_DIR,
    GAME_DIR,
    GAME_V0,
    INTEGRATION_V0_DIR,
    LEGACY_SMB_ROOTS,
    OPTIMIZER_RUNS_DIR,
    SNES_EDITOR_SMB_ROOT,
)
from smb.ram import read_snapshot
from smb.routes import ExitRoute, ExitSegment, get_route

SourceKind = Literal["playthrough", "legal_stitch", "optimizer"]


@dataclass
class SegmentClip:
    """Resolved playable clip for one exit."""

    exit: ExitSegment
    state_path: Path
    """Save state loaded before this clip."""

    play_buttons: list[list[int]]
    """Buttons rendered to video (NES/SNES 9–12 element vectors)."""

    prefix_buttons: list[list[int]] = field(default_factory=list)
    """Buttons stepped without rendering (branch offset warm-up)."""

    source_kind: str = ""
    session_id: str = ""
    branch_id: int = 0
    frames: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_manifest_row(self) -> dict[str, Any]:
        row = {
            "exit_id": self.exit.exit_id,
            "segment_id": self.exit.segment_id,
            "label": self.exit.display(),
            "state_path": str(self.state_path),
            "frames": self.frames or len(self.play_buttons),
            "prefix_frames": len(self.prefix_buttons),
            "source_kind": self.source_kind,
            "session_id": self.session_id,
            "branch_id": self.branch_id,
        }
        row.update(self.meta)
        return row


@dataclass
class StitchPlan:
    """Resolved plan for a full route video."""

    route: ExitRoute
    source_kind: SourceKind
    clips: list[SegmentClip]
    missing: list[str] = field(default_factory=list)
    total_play_frames: int = 0
    notes: list[str] = field(default_factory=list)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "route_id": self.route.route_id,
            "display_name": self.route.display_name,
            "source_kind": self.source_kind,
            "total_play_frames": self.total_play_frames,
            "total_play_seconds": round(self.total_play_frames / 60.0, 3),
            "missing": self.missing,
            "notes": self.notes,
            "clips": [c.to_manifest_row() for c in self.clips],
        }


def resolve_state_path(
    state_file: str | None,
    *,
    state_name: str | None = None,
    session_dir: Path | None = None,
    recordings_dir: Path = FULLGAME_RECORDINGS_DIR,
    integration_dir: Path = INTEGRATION_V0_DIR,
) -> Path:
    """Resolve a branch/session state path across relocated trees.

    Practice branches often store absolute paths from an older
    ``speedrun/retro_rl/super_mario_bros`` checkout. Rewrite those onto the
    live snes_editor (or local symlink) tree, then fall back to session
    ``states/`` and the integration directory.
    """
    candidates: list[Path] = []

    if state_file:
        raw = Path(state_file)
        if raw.is_absolute():
            candidates.append(raw)
            for legacy in LEGACY_SMB_ROOTS:
                try:
                    rel = raw.relative_to(legacy)
                except ValueError:
                    continue
                candidates.append(SNES_EDITOR_SMB_ROOT / rel)
                candidates.append(GAME_DIR / rel)
        else:
            candidates.append(raw)
            if session_dir is not None:
                candidates.append(session_dir / raw)
                candidates.append(session_dir / "states" / raw.name)
            candidates.append(integration_dir / raw)
            candidates.append(integration_dir / raw.name)

        # Same basename under session states / integration.
        name = raw.name
        if session_dir is not None:
            candidates.append(session_dir / "states" / name)
        candidates.append(integration_dir / name)

    if state_name:
        candidates.append(integration_dir / f"{state_name}.state")
        if session_dir is not None:
            candidates.append(session_dir / "states" / f"{state_name}.state")

    seen: set[Path] = set()
    for cand in candidates:
        resolved = cand if cand.is_absolute() else (recordings_dir / cand)
        key = resolved.resolve() if resolved.exists() else resolved
        if key in seen:
            continue
        seen.add(key)
        if resolved.exists():
            return resolved.resolve()

    tried = ", ".join(str(c) for c in candidates[:8])
    raise FileNotFoundError(
        f"Could not resolve state_file={state_file!r} state_name={state_name!r}. "
        f"Tried: {tried}"
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _branch_path(session_dir: Path, branch_id: int) -> Path:
    return session_dir / "branches" / f"branch_{int(branch_id):03d}.json"


def _legal_rows_for_route(
    route: ExitRoute,
    leaderboard: dict[str, Any],
) -> list[tuple[ExitSegment, dict[str, Any] | None]]:
    by_id = {
        str(row["segment_id"]): row
        for row in leaderboard.get("legal_stitch", {}).get("segments", [])
    }
    # full_level_best is the same rows keyed by segment_id (fallback).
    full_best = leaderboard.get("full_level_best") or {}
    rows: list[tuple[ExitSegment, dict[str, Any] | None]] = []
    for exit_seg in route.exits:
        row = by_id.get(exit_seg.segment_id) or full_best.get(exit_seg.segment_id)
        rows.append((exit_seg, row))
    return rows


def resolve_legal_stitch_clip(
    exit_seg: ExitSegment,
    row: dict[str, Any],
    *,
    recordings_dir: Path = FULLGAME_RECORDINGS_DIR,
    integration_dir: Path = INTEGRATION_V0_DIR,
) -> SegmentClip:
    """Build a clip from one leaderboard legal-stitch row + branch JSON."""
    session_id = str(row["session_id"])
    branch_id = int(row["start_branch_id"])
    session_dir = recordings_dir / session_id
    branch = _load_json(_branch_path(session_dir, branch_id))

    started_at = int(branch.get("started_at_frame") or 0)
    start_frame = int(row["start_frame"])
    frames = int(row["frames"])
    offset = start_frame - started_at
    if offset < 0:
        raise ValueError(
            f"{exit_seg.segment_id}: start_frame {start_frame} before "
            f"branch start {started_at}"
        )

    raw = branch.get("raw_buttons") or []
    if not raw:
        raise ValueError(f"{exit_seg.segment_id}: branch {branch_id} has no raw_buttons")

    end = offset + frames
    if end > len(raw):
        raise ValueError(
            f"{exit_seg.segment_id}: need buttons[{offset}:{end}] but branch "
            f"only has {len(raw)} frames"
        )

    state_path = resolve_state_path(
        str(branch.get("state_file") or row.get("start_state_name") or ""),
        state_name=str(
            branch.get("state_name") or row.get("start_state_name") or ""
        )
        or None,
        session_dir=session_dir,
        recordings_dir=recordings_dir,
        integration_dir=integration_dir,
    )

    play = _nes9(raw, offset, end)
    prefix = _nes9(raw, 0, offset)

    return SegmentClip(
        exit=exit_seg,
        state_path=state_path,
        play_buttons=play,
        prefix_buttons=prefix,
        source_kind="legal_stitch",
        session_id=session_id,
        branch_id=branch_id,
        frames=len(play),
        meta={
            "start_source": row.get("start_source"),
            "start_state_name": row.get("start_state_name")
            or branch.get("state_name"),
            "leaderboard_frames": frames,
            "leaderboard_seconds": row.get("seconds"),
            "branch_offset": offset,
        },
    )


def _find_optimizer_recording(segment_id: str, runs_dir: Path) -> Path | None:
    level_dir = runs_dir / segment_id
    if not level_dir.exists():
        return None

    preferred = (
        "hillclimb_best_final.json",
        "hillclimb_raw_best.json",
        "recording_000.json",
    )
    for name in preferred:
        path = level_dir / name
        if not path.exists():
            continue
        try:
            data = _load_json(path)
        except json.JSONDecodeError:
            continue
        if name.startswith("recording") or data.get("completed", False):
            if data.get("raw_buttons") or data.get("actions"):
                return path
    # Fall back to any completed-looking file.
    for path in sorted(level_dir.glob("*.json")):
        if path.name.endswith("_trace.json"):
            continue
        try:
            data = _load_json(path)
        except json.JSONDecodeError:
            continue
        if data.get("completed") and (data.get("raw_buttons") or data.get("actions")):
            return path
    return None


def resolve_optimizer_clip(
    exit_seg: ExitSegment,
    *,
    runs_dir: Path = OPTIMIZER_RUNS_DIR,
    integration_dir: Path = INTEGRATION_V0_DIR,
    start_state: str | None = None,
) -> SegmentClip:
    """Build a clip from an optimizer / hillclimb recording + level start state."""
    rec_path = _find_optimizer_recording(exit_seg.segment_id, runs_dir)
    if rec_path is None:
        raise FileNotFoundError(
            f"No optimizer recording for {exit_seg.segment_id} under {runs_dir}"
        )
    data = _load_json(rec_path)
    if data.get("raw_buttons"):
        play = _nes9(data["raw_buttons"])
    else:
        # Action indices — expand via SMB action table at render time is heavier;
        # store as single-int "buttons" and expand later.
        from retro_harness.platformer.actions import action_index_to_buttons
        from smb.platformer_levels import SMB_ACTIONS

        play = [
            list(action_index_to_buttons(int(a), SMB_ACTIONS))
            for a in data["actions"]
        ]

    state_name = start_state or _default_level_state(exit_seg)
    state_path = resolve_state_path(
        f"{state_name}.state",
        state_name=state_name,
        integration_dir=integration_dir,
    )
    return SegmentClip(
        exit=exit_seg,
        state_path=state_path,
        play_buttons=play,
        prefix_buttons=[],
        source_kind="optimizer",
        session_id="",
        branch_id=0,
        frames=len(play),
        meta={
            "recording": str(rec_path),
            "completed": data.get("completed"),
            "start_state_name": state_name,
        },
    )


def _default_level_state(exit_seg: ExitSegment) -> str:
    if exit_seg.world and exit_seg.level:
        return f"Level{exit_seg.world}_{exit_seg.level}"
    # smb_1_1 → Level1_1
    parts = exit_seg.segment_id.split("_")
    if len(parts) >= 3 and parts[0] == "smb":
        return f"Level{parts[1]}_{parts[2]}"
    raise ValueError(f"Cannot infer start state for {exit_seg.segment_id}")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _branch_covering(
    summary: dict[str, Any], frame: int
) -> dict[str, Any] | None:
    branches = summary.get("branches") or []
    for inclusive in (False, True):
        for branch in branches:
            start = int(branch.get("started_at_frame") or 0)
            end = int(branch.get("ended_at_frame") or 10**9)
            if start <= frame < end or (inclusive and start <= frame <= end):
                return branch
    return None


def _nes9(raw: list[Any], lo: int = 0, hi: int | None = None) -> list[list[int]]:
    return [list(map(int, frame)) for frame in raw[lo:hi]]


def _pad_action(raw: list[int], action_size: int) -> np.ndarray:
    buttons = list(raw[:action_size])
    if len(buttons) < action_size:
        buttons += [0] * (action_size - len(buttons))
    return np.array(buttons, dtype=np.int8)


def _open_clip_env(
    clip: SegmentClip,
    *,
    game_dir: Path = GAME_DIR,
    game_name: str = GAME_V0,
) -> Any:
    from retro_harness.env import make_env

    env = make_env(game_name, None, game_dir, render_mode="rgb_array")
    env.reset()
    env.em.set_state(read_state_bytes(clip.state_path))
    return env


def _warm_prefix(env: Any, clip: SegmentClip) -> int:
    action_size = int(env.action_space.shape[0])
    for raw in clip.prefix_buttons:
        env.step(_pad_action(raw, action_size))
    return action_size


def verify_clip_deathless(
    clip: SegmentClip,
    *,
    game_dir: Path = GAME_DIR,
    game_name: str = GAME_V0,
) -> dict[str, Any]:
    """Replay a clip in the emulator; report deaths / dying status."""
    env = _open_clip_env(clip, game_dir=game_dir, game_name=game_name)
    try:
        action_size = _warm_prefix(env, clip)
        snap = read_snapshot(env.get_ram())
        last_lives = int(snap.lives)
        was_dying = snap.dying
        max_x = int(snap.player_x)
        deaths = 0
        dying = 0
        first_death_frame: int | None = None

        for i, raw in enumerate(clip.play_buttons):
            env.step(_pad_action(raw, action_size))
            snap = read_snapshot(env.get_ram())
            max_x = max(max_x, int(snap.player_x))
            if int(snap.lives) < last_lives:
                deaths += 1
                if first_death_frame is None:
                    first_death_frame = i
            if snap.dying and not was_dying:
                dying += 1
            last_lives = int(snap.lives)
            was_dying = snap.dying

        return {
            "ok": deaths == 0 and dying == 0,
            "deaths": deaths,
            "dying": dying,
            "max_x": max_x,
            "first_death_frame": first_death_frame,
            "final": {
                "world": snap.world,
                "level": snap.level,
                "player_x": snap.player_x,
                "lives": snap.lives,
                "game_mode": snap.oper_mode,
                "player_status": snap.player_state,
            },
        }
    finally:
        env.close()


def resolve_session_window_clip(
    exit_seg: ExitSegment,
    *,
    session_dir: Path,
    start_frame: int,
    end_frame: int,
    integration_dir: Path = INTEGRATION_V0_DIR,
    source_kind: str = "playthrough",
) -> SegmentClip:
    """Build a clip from a session frame window + owning branch state."""
    summary = _load_json(session_dir / "summary.json")
    bmeta = _branch_covering(summary, start_frame)
    if bmeta is None:
        bmeta = _branch_covering(summary, max(0, end_frame - 1))
    if bmeta is None:
        raise FileNotFoundError(
            f"No branch covers frames {start_frame}-{end_frame} in {session_dir.name}"
        )

    branch_id = int(bmeta["branch_id"])
    branch = _load_json(_branch_path(session_dir, branch_id))
    branch_start = int(branch.get("started_at_frame") or bmeta.get("started_at_frame") or 0)
    offset = start_frame - branch_start
    frames = end_frame - start_frame
    raw = branch.get("raw_buttons") or []
    if offset < 0 or offset >= len(raw):
        raise ValueError(
            f"{exit_seg.segment_id}: offset {offset} out of range for branch "
            f"{branch_id} ({len(raw)} frames, start={branch_start})"
        )
    end = min(offset + frames, len(raw))
    play = _nes9(raw, offset, end)
    prefix = _nes9(raw, 0, offset)

    state_path = resolve_state_path(
        str(branch.get("state_file") or ""),
        state_name=str(branch.get("state_name") or "") or None,
        session_dir=session_dir,
        integration_dir=integration_dir,
    )
    return SegmentClip(
        exit=exit_seg,
        state_path=state_path,
        play_buttons=play,
        prefix_buttons=prefix,
        source_kind=source_kind,
        session_id=session_dir.name,
        branch_id=branch_id,
        frames=len(play),
        meta={
            "start_frame": start_frame,
            "end_frame": start_frame + len(play),
            "branch_offset": offset,
            "start_state_name": branch.get("state_name"),
        },
    )


def _final_attempt_window(
    events: list[dict[str, Any]],
    summary: dict[str, Any],
    segment_id: str,
    *,
    completed_frame: int | None = None,
) -> tuple[int, int] | None:
    """Return (start_frame, end_frame) for the last successful split of segment."""
    splits = [
        e
        for e in events
        if e.get("event") == "split"
        and e.get("segment_id") == segment_id
        and e.get("reason") in ("segment_change", "game_complete", None)
    ]
    if completed_frame is not None:
        splits = [e for e in splits if int(e.get("frame") or 0) <= completed_frame + 5]
    if not splits:
        return None
    split_e = splits[-1]
    end_frame = int(split_e["frame"])
    bmeta = _branch_covering(summary, end_frame - 1) or _branch_covering(
        summary, end_frame
    )
    if bmeta is None:
        return None
    branch_start = int(bmeta.get("started_at_frame") or 0)
    starts = [
        e
        for e in events
        if e.get("event") == "split_start"
        and e.get("segment_id") == segment_id
        and branch_start <= int(e.get("frame") or 0) <= end_frame
    ]
    start_frame = int(starts[-1]["frame"]) if starts else branch_start
    start_frame = max(start_frame, branch_start)
    if end_frame <= start_frame:
        return None
    return start_frame, end_frame


def build_session_playthrough_plan(
    route: ExitRoute,
    session_id: str,
    *,
    recordings_dir: Path = FULLGAME_RECORDINGS_DIR,
    integration_dir: Path = INTEGRATION_V0_DIR,
    require_verified: bool = True,
    skip_missing: bool = False,
) -> StitchPlan:
    """Build a death-verified plan from one practice session's successful path.

    For each route exit, takes the final completing split in that session and
    the owning branch's start state, then optionally re-simulates to reject
    desyncing / death-filled windows (the failure mode of naive legal stitch).
    """
    session_dir = recordings_dir / session_id
    if not session_dir.exists():
        raise FileNotFoundError(f"Session not found: {session_dir}")

    events = _load_jsonl(session_dir / "events.jsonl")
    summary = _load_json(session_dir / "summary.json")
    completed_frame = summary.get("completed_frame")
    if completed_frame is not None:
        completed_frame = int(completed_frame)

    clips: list[SegmentClip] = []
    missing: list[str] = []
    notes: list[str] = [
        f"session playthrough from {session_id}",
        "each clip is the final successful attempt for that exit in-session",
    ]

    for exit_seg in route.exits:
        window = _final_attempt_window(
            events,
            summary,
            exit_seg.segment_id,
            completed_frame=completed_frame,
        )
        if window is None:
            missing.append(exit_seg.exit_id)
            msg = f"skip {exit_seg.exit_id}: no completing split in session"
            notes.append(msg)
            if not skip_missing:
                raise FileNotFoundError(msg)
            continue
        start_frame, end_frame = window
        try:
            clip = resolve_session_window_clip(
                exit_seg,
                session_dir=session_dir,
                start_frame=start_frame,
                end_frame=end_frame,
                integration_dir=integration_dir,
                source_kind="playthrough",
            )
        except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
            missing.append(exit_seg.exit_id)
            msg = f"skip {exit_seg.exit_id}: {exc}"
            notes.append(msg)
            if not skip_missing:
                raise
            continue

        if require_verified:
            report = verify_clip_deathless(clip)
            clip.meta["verify"] = report
            if not report["ok"]:
                missing.append(exit_seg.exit_id)
                msg = (
                    f"skip {exit_seg.exit_id}: verify failed "
                    f"deaths={report['deaths']} dying={report['dying']} "
                    f"max_x={report['max_x']}"
                )
                notes.append(msg)
                if not skip_missing:
                    raise RuntimeError(msg)
                continue
            notes.append(
                f"{exit_seg.exit_id}: verified clean "
                f"{clip.frames}f max_x={report['max_x']}"
            )

        clips.append(clip)

    total = sum(len(c.play_buttons) for c in clips)
    return StitchPlan(
        route=route,
        source_kind="playthrough",
        clips=clips,
        missing=missing,
        total_play_frames=total,
        notes=notes,
    )


def select_playthrough_session(
    route: ExitRoute,
    *,
    recordings_dir: Path = FULLGAME_RECORDINGS_DIR,
    integration_dir: Path = INTEGRATION_V0_DIR,
    prefer_sessions: tuple[str, ...] = (),
) -> str:
    """Pick a completed full-game session that verifies clean for every exit.

    Prefers longer total verified play frames. ``prefer_sessions`` are tried
    first (still must fully verify).
    """
    leaderboard_path = recordings_dir / "leaderboard.json"
    candidates: list[str] = list(prefer_sessions)
    if leaderboard_path.exists():
        lb = _load_json(leaderboard_path)
        for key in ("full_runs", "completed_sessions"):
            for row in lb.get(key) or []:
                sid = str(row.get("session_id") or "")
                if sid and sid not in candidates:
                    candidates.append(sid)
    # Known completed any% practice sessions (world-8 coverage / length).
    for sid in (
        "20260429_214207",
        "20260429_172649",
        "20260429_165717",
        "20260429_174136",
    ):
        if sid not in candidates:
            candidates.append(sid)

    best_id: str | None = None
    best_score = -1
    errors: list[str] = []

    for sid in candidates:
        session_dir = recordings_dir / sid
        if not (session_dir / "summary.json").exists():
            continue
        try:
            plan = build_session_playthrough_plan(
                route,
                sid,
                recordings_dir=recordings_dir,
                integration_dir=integration_dir,
                require_verified=True,
                skip_missing=False,
            )
        except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
            errors.append(f"{sid}: {exc}")
            continue
        if plan.missing or len(plan.clips) != len(route.exits):
            errors.append(
                f"{sid}: incomplete clips={len(plan.clips)} missing={plan.missing}"
            )
            continue
        score = plan.total_play_frames
        # Prefer longer world-8 coverage when totals are close.
        for clip in plan.clips:
            if clip.exit.segment_id in {"smb_8_1", "smb_8_2"}:
                score += clip.frames // 2
        if score > best_score:
            best_score = score
            best_id = sid

    if best_id is None:
        detail = "; ".join(errors[:6]) if errors else "no candidates"
        raise RuntimeError(
            "No fully verified playthrough session found for route "
            f"{route.route_id}. {detail}"
        )
    return best_id


def build_stitch_plan(
    route: ExitRoute | str,
    *,
    source: SourceKind = "playthrough",
    recordings_dir: Path = FULLGAME_RECORDINGS_DIR,
    runs_dir: Path = OPTIMIZER_RUNS_DIR,
    integration_dir: Path = INTEGRATION_V0_DIR,
    skip_missing: bool = True,
    session_id: str | None = None,
    require_verified: bool = True,
) -> StitchPlan:
    """Resolve every exit on ``route`` into a playable clip list."""
    if isinstance(route, str):
        route = get_route(route)

    clips: list[SegmentClip] = []
    missing: list[str] = []
    notes: list[str] = []

    if source == "playthrough":
        sid = session_id or select_playthrough_session(
            route,
            recordings_dir=recordings_dir,
            integration_dir=integration_dir,
        )
        return build_session_playthrough_plan(
            route,
            sid,
            recordings_dir=recordings_dir,
            integration_dir=integration_dir,
            require_verified=require_verified,
            skip_missing=skip_missing,
        )

    if source == "legal_stitch":
        leaderboard_path = recordings_dir / "leaderboard.json"
        if not leaderboard_path.exists():
            raise FileNotFoundError(
                f"Missing {leaderboard_path}. Run the snes_editor fullgame "
                "leaderboard analyzer first, or pass --source optimizer."
            )
        leaderboard = _load_json(leaderboard_path)
        for exit_seg, row in _legal_rows_for_route(route, leaderboard):
            if row is None:
                missing.append(exit_seg.exit_id)
                if skip_missing:
                    notes.append(f"skip {exit_seg.exit_id}: no legal_stitch row")
                    continue
                raise FileNotFoundError(
                    f"No legal_stitch row for {exit_seg.segment_id}"
                )
            try:
                clips.append(
                    resolve_legal_stitch_clip(
                        exit_seg,
                        row,
                        recordings_dir=recordings_dir,
                        integration_dir=integration_dir,
                    )
                )
            except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
                missing.append(exit_seg.exit_id)
                if skip_missing:
                    notes.append(f"skip {exit_seg.exit_id}: {exc}")
                    continue
                raise
    elif source == "optimizer":
        for exit_seg in route.exits:
            try:
                clips.append(
                    resolve_optimizer_clip(
                        exit_seg,
                        runs_dir=runs_dir,
                        integration_dir=integration_dir,
                    )
                )
            except FileNotFoundError as exc:
                missing.append(exit_seg.exit_id)
                if skip_missing:
                    notes.append(f"skip {exit_seg.exit_id}: {exc}")
                    continue
                raise
    else:
        raise ValueError(f"Unknown source {source!r}")

    total = sum(len(c.play_buttons) for c in clips)
    return StitchPlan(
        route=route,
        source_kind=source,
        clips=clips,
        missing=missing,
        total_play_frames=total,
        notes=notes,
    )


def render_stitch_plan(
    plan: StitchPlan,
    output: Path | str,
    *,
    scale: int = 3,
    title_card_frames: int | None = None,
    game_dir: Path = GAME_DIR,
    game_name: str = GAME_V0,
    fps: int = 60,
    abort_on_death: bool = True,
) -> Path:
    """Replay resolved clips into a single MP4 via ``rta_panel.VideoWriter``."""
    from retro_harness.segment_runner import configure_headless
    from smb.rta_panel import VideoWriter, env_audio_rate, write_video

    configure_headless()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not plan.clips:
        raise RuntimeError("Stitch plan has no clips to render")
    hold = (
        (24 if plan.source_kind == "playthrough" else 90)
        if title_card_frames is None
        else title_card_frames
    )

    probe = _open_clip_env(plan.clips[0], game_dir=game_dir, game_name=game_name)
    try:
        obs = probe.render()
        if obs is None:
            obs, *_ = probe.step(
                np.zeros(probe.action_space.shape[0], dtype=np.int8)
            )
        h, w = int(obs.shape[0]), int(obs.shape[1])
        audio_rate = env_audio_rate(probe)
    finally:
        probe.close()

    video = VideoWriter(
        output_path,
        width=w,
        height=h,
        scale=scale,
        fps=fps,
        audio_rate=audio_rate,
        hud=True,
        route_label=plan.route.display_name,
        splits_panel=plan.route.route_id == "smb_warp_any_percent",
    )
    print(
        f"Recording {plan.route.display_name} source={plan.source_kind} "
        f"clips={len(plan.clips)} frames={plan.total_play_frames} → {output_path} "
        f"{video.out_w}x{video.out_h}"
    )
    if plan.missing:
        print(f"Missing exits (skipped): {', '.join(plan.missing)}")
    for note in plan.notes:
        print(f"  note: {note}")

    cumulative = 0
    deaths_seen = 0
    try:
        for i, clip in enumerate(plan.clips):
            label = clip.exit.display()
            if hold > 0:
                video.write_intro([label], hold_frames=hold)
            env = _open_clip_env(clip, game_dir=game_dir, game_name=game_name)
            try:
                action_size = _warm_prefix(env, clip)
                snap = read_snapshot(env.get_ram())
                last_lives = int(snap.lives)
                was_dying = snap.dying
                print(
                    f"  [{i}] {label}: {len(clip.play_buttons)}f "
                    f"(+{len(clip.prefix_buttons)} prefix) "
                    f"from {clip.state_path.name}"
                    + (f" session={clip.session_id}" if clip.session_id else "")
                )
                for frame_idx, raw in enumerate(clip.play_buttons):
                    action = _pad_action(raw, action_size)
                    obs, *_ = env.step(action)
                    snap = read_snapshot(env.get_ram())
                    died_now = int(snap.lives) < last_lives or (
                        snap.dying and not was_dying
                    )
                    last_lives = int(snap.lives)
                    was_dying = snap.dying
                    write_video(
                        video, obs, env=env, action=action, label=label, snap=snap
                    )
                    cumulative += 1
                    if died_now:
                        deaths_seen += 1
                        if abort_on_death:
                            raise RuntimeError(
                                f"Death during render at {label} frame {frame_idx} "
                                f"(session={clip.session_id}). "
                                "Plan should have been verify-filtered."
                            )
            finally:
                env.close()
    finally:
        try:
            video.close()
        except RuntimeError:
            if deaths_seen == 0:
                raise

    if not output_path.exists():
        raise RuntimeError(f"Video not written: {output_path}")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(
        f"\nDone! {output_path} ({size_mb:.1f} MB, "
        f"{cumulative}f / {cumulative / fps:.1f}s, deaths_seen={deaths_seen})"
    )

    manifest_path = output_path.with_suffix(".json")
    manifest = plan.to_manifest()
    manifest["video"] = str(output_path.resolve())
    manifest["rendered_frames"] = cumulative
    manifest["scale"] = scale
    manifest["fps"] = fps
    manifest["deaths_seen"] = deaths_seen
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest: {manifest_path}")
    return output_path


def write_plan_manifest(plan: StitchPlan, path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan.to_manifest(), indent=2) + "\n", encoding="utf-8")
    return out
