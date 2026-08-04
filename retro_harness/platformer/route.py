"""Route definition and evaluation for multi-segment speedruns.

A route is an ordered list of segments, each referencing a registered
LevelConfig.  evaluate_route() runs each segment independently using
the Evaluator (same pattern as super_metroid_rl/scripts/eval_full_route.py),
giving reliable per-segment results without emulator state leakage.

Usage:
    from retro_harness.platformer.route import RouteConfig, evaluate_route, get_route

    route = get_route("smb_any_percent")
    results = evaluate_route(route)
"""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from retro_harness.platformer.evaluator import Evaluator, EvalResult
from retro_harness.platformer.level_config import get_level_config, LevelConfig


@dataclass
class RouteSegment:
    """One segment of a speedrun route."""

    config_id: str          # registered LevelConfig ID
    label: str = ""         # human-readable label (e.g. "1-1", "8-4 seg3")
    recording: str = ""     # explicit path to recording JSON (relative to runs_dir or absolute)
    neuro_checkpoint: str = ""  # path to neuro_best.json (relative to runs_dir); plays live after recording


@dataclass
class RouteConfig:
    """Ordered list of segments forming a complete speedrun route."""

    route_id: str
    display_name: str
    segments: list[RouteSegment] = field(default_factory=list)


# -- Route registry ----------------------------------------------------------

ROUTE_REGISTRY: dict[str, RouteConfig] = {}


def register_route(route: RouteConfig, *aliases: str) -> None:
    """Register a route config."""
    ROUTE_REGISTRY[route.route_id] = route
    for alias in aliases:
        ROUTE_REGISTRY[alias] = route


def get_route(route_id: str) -> RouteConfig:
    """Look up a route by ID or alias."""
    key = route_id.lower() if route_id.lower() in ROUTE_REGISTRY else route_id
    if key not in ROUTE_REGISTRY:
        available = sorted(set(r.route_id for r in ROUTE_REGISTRY.values()))
        raise KeyError(f"Unknown route '{route_id}'. Available: {available}")
    return ROUTE_REGISTRY[key]


def list_routes() -> list[RouteConfig]:
    """Return deduplicated list of all registered routes."""
    seen: set[str] = set()
    result: list[RouteConfig] = []
    for route in ROUTE_REGISTRY.values():
        if route.route_id not in seen:
            seen.add(route.route_id)
            result.append(route)
    return result


# -- Recording discovery -----------------------------------------------------

def find_best_recording(config: LevelConfig) -> Path | None:
    """Find the best available recording for a level config.

    Priority: hillclimb (if completed) > recording_000.
    Hillclimb results that didn't complete are skipped in favor of
    the original recording which may have raw buttons for faithful replay.
    """
    runs = config.runs_dir
    if not runs.exists():
        return None

    hill = runs / "hillclimb_best_final.json"
    if hill.exists():
        try:
            data = json.loads(hill.read_text())
            if data.get("completed", False):
                return hill
        except (json.JSONDecodeError, KeyError):
            pass

    # Check all recording_*.json for a completed one (prefer highest number)
    for rec_path in sorted(runs.glob("recording_*.json"), reverse=True):
        if "_raw" in rec_path.stem:
            continue
        try:
            data = json.loads(rec_path.read_text())
            if data.get("completed", False):
                return rec_path
        except (json.JSONDecodeError, KeyError):
            pass

    rec = runs / "recording_000.json"
    if rec.exists():
        return rec

    return None


def _load_practice_seeds(
    practice_dir: Path,
    min_frames: int = 60,
) -> list[tuple[list[int] | list[list[int]], bool]]:
    """Load practice attempts as ``(frames, is_raw)`` seed pairs.

    Faithful companion ``*_raw.json`` inputs take precedence. Older attempts
    that only contain action indices remain supported as a fallback.
    """
    from retro_harness.platformer.bk2_extract import load_raw_buttons

    seed_files = sorted(practice_dir.glob("attempt_*.json"))
    seed_files = [f for f in seed_files if "_raw" not in f.stem]

    seeds: list[tuple[list[int] | list[list[int]], bool]] = []
    for f in seed_files:
        try:
            raw = load_raw_buttons(f)
            if raw is not None:
                if len(raw) >= min_frames:
                    seeds.append((raw, True))
                continue
            data = json.loads(f.read_text())
            actions = data["actions"]
            if len(actions) >= min_frames:
                seeds.append((actions, False))
        except (KeyError, OSError, TypeError, json.JSONDecodeError):
            pass
    return seeds


def load_recording_data(path: Path) -> tuple[list, bool]:
    """Load actions from a recording JSON.

    Returns (actions, is_raw).
    Prefers raw buttons (companion _raw.json or embedded) for faithful replay.
    Falls back to action indices if no raw data available.
    """
    from retro_harness.platformer.bk2_extract import load_raw_buttons

    raw = load_raw_buttons(path)
    if raw is not None:
        return raw, True
    data = json.loads(path.read_text())
    return data["actions"], False


# -- Route evaluation --------------------------------------------------------

@dataclass
class SegmentResult:
    """Result of evaluating one route segment."""

    segment: RouteSegment
    config: LevelConfig
    recording_path: Path | None
    eval_result: EvalResult | None
    error: str = ""


@dataclass
class RouteResult:
    """Result of evaluating a full route."""

    route: RouteConfig
    segments: list[SegmentResult] = field(default_factory=list)

    @property
    def total_frames(self) -> int:
        return sum(
            s.eval_result.total_frames
            for s in self.segments
            if s.eval_result and s.eval_result.completed
        )

    @property
    def completed_count(self) -> int:
        return sum(1 for s in self.segments if s.eval_result and s.eval_result.completed)

    @property
    def total_count(self) -> int:
        return len(self.segments)

    @property
    def all_completed(self) -> bool:
        return self.completed_count == self.total_count


def evaluate_route(
    route: RouteConfig,
    *,
    verbose: bool = True,
    pad_frames: int = 100,
) -> RouteResult:
    """Evaluate a full route by running each segment independently.

    Each segment gets its own Evaluator with its own saved state,
    so there's no emulator state leakage between segments.

    Args:
        route: The route to evaluate.
        verbose: Print per-segment results.
        pad_frames: Extra no-input frames appended to catch delayed transitions.
    """
    result = RouteResult(route=route)

    if verbose:
        print(f"Route: {route.display_name} ({len(route.segments)} segments)\n")
        print(f"{'#':>2s}  {'Label':<20s}  {'Config':<20s}  {'Recording':<35s}  "
              f"{'Frames':>6s}  {'Progress':>8s}  {'Fitness':>10s}  {'Status'}")
        print("-" * 130)

    cumulative = 0

    for i, seg in enumerate(route.segments):
        try:
            config = get_level_config(seg.config_id)
        except KeyError as e:
            sr = SegmentResult(seg, None, None, None, error=str(e))  # type: ignore[arg-type]
            result.segments.append(sr)
            if verbose:
                print(f"  {i:2d}  {seg.label:<20s}  {seg.config_id:<20s}  {'':35s}  CONFIG ERROR: {e}")
            continue

        # Find recording
        if seg.recording:
            rec_path = Path(seg.recording)
            if not rec_path.is_absolute():
                rec_path = config.runs_dir / seg.recording
        else:
            rec_path = find_best_recording(config)

        if rec_path is None or not rec_path.exists():
            sr = SegmentResult(seg, config, rec_path, None, error="no recording")
            result.segments.append(sr)
            if verbose:
                print(f"  {i:2d}  {seg.label:<20s}  {seg.config_id:<20s}  {'MISSING':35s}")
            continue

        try:
            actions, is_raw = load_recording_data(rec_path)

            # Pad with no-input frames for delayed transitions
            if is_raw:
                btn_len = len(actions[0]) if actions else 12
                actions = actions + [[0] * btn_len] * pad_frames
            else:
                actions = actions + [0] * pad_frames

            ev = Evaluator(config)
            er = ev.evaluate(actions, early_terminate=False)
            ev.close()

            sr = SegmentResult(seg, config, rec_path, er)
            result.segments.append(sr)

            if er.completed:
                cumulative += er.total_frames

            if verbose:
                status = "COMPLETED" if er.completed else ("DIED" if er.died else "incomplete")
                icon = {"COMPLETED": "+", "DIED": "X", "incomplete": "-"}[status]
                rec_name = rec_path.name
                if rec_path.parent.name != config.level_id:
                    rec_name = f"{rec_path.parent.name}/{rec_name}"
                print(f"  {i:2d}  {seg.label:<20s}  {seg.config_id:<20s}  {rec_name:<35s}  "
                      f"{er.total_frames:>6d}f  {er.max_progress:>8.1f}  "
                      f"{er.fitness:>10.0f}  {icon} {status}")

        except Exception as e:
            sr = SegmentResult(seg, config, rec_path, None, error=str(e))
            result.segments.append(sr)
            if verbose:
                print(f"  {i:2d}  {seg.label:<20s}  {seg.config_id:<20s}  {rec_path.name:<35s}  ERROR: {e}")

    if verbose:
        print(f"\n{'='*80}")
        print(f"Results: {result.completed_count}/{result.total_count} segments completed")
        print(f"Total completion frames: {result.total_frames} ({result.total_frames / 60:.1f}s)")

        failed = [i for i, s in enumerate(result.segments)
                  if s.eval_result and not s.eval_result.completed]
        missing = [i for i, s in enumerate(result.segments)
                   if s.eval_result is None]
        if failed:
            labels = [result.segments[i].segment.label for i in failed]
            print(f"Failed/incomplete: {labels}")
        if missing:
            labels = [result.segments[i].segment.label for i in missing]
            print(f"Missing: {labels}")

    return result


# -- True chain-live (single emulator, no state reloading) -------------------


@dataclass
class ChainLiveSegmentResult:
    """Result of replaying one segment in a chain-live run."""

    segment: RouteSegment
    config: LevelConfig
    recording_path: Path | None
    status: str  # "COMPLETED", "DIED", "TRANSITION_FAILED", "NO_RECORDING", "ERROR"
    frames: int = 0
    error: str = ""


@dataclass
class ChainLiveResult:
    """Result of a full chain-live run."""

    route: RouteConfig
    segments: list[ChainLiveSegmentResult] = field(default_factory=list)
    total_frames: int = 0

    @property
    def completed_count(self) -> int:
        return sum(1 for s in self.segments if s.status == "COMPLETED")

    @property
    def all_completed(self) -> bool:
        return (
            len(self.segments) == len(self.route.segments)
            and all(s.status == "COMPLETED" for s in self.segments)
        )


def _run_neuro_live(
    env,
    config: LevelConfig,
    checkpoint_path: Path,
    schema,
    main_ids: set,
    initial_lives: int | None,
    max_progress: float,
    write_video_frame,
    seg_label: str,
    verbose: bool,
    max_frames: int = 3000,
) -> tuple[bool, bool, int, float]:
    """Run a neural network live on the emulator until completion or death.

    Returns (completed, died, frames_played, max_progress).
    """
    from retro_harness.platformer.neuro import (
        NeuralNet,
        outputs_to_buttons,
        read_smb_inputs,
        read_smb_inputs_legacy,
    )

    ckpt = json.loads(checkpoint_path.read_text())
    net = NeuralNet(
        n_inputs=ckpt["n_inputs"],
        n_hidden=ckpt["n_hidden"],
        n_outputs=ckpt["n_outputs"],
        weights=np.array(ckpt["weights"], dtype=np.float32),
        arch=ckpt.get("arch", "mlp"),
        hidden_layers=tuple(ckpt.get("hidden_layers", (ckpt["n_hidden"],))),
        n_conv=int(ckpt.get("n_conv", 8)),
        use_recurrent=bool(ckpt.get("use_recurrent", False)),
    )
    # Old checkpoints are 189-dim; new builder is 210-dim.
    read_fn = read_smb_inputs_legacy if int(ckpt["n_inputs"]) <= 189 else read_smb_inputs
    if verbose:
        print(f"       neuro: {ckpt['n_inputs']}in {ckpt['n_hidden']}h {ckpt['n_outputs']}out")

    action_size = env.action_space.shape[0]
    completed = False
    died = False
    net.reset_state()

    for frame_idx in range(max_frames):
        ram = env.get_ram()
        inputs = read_fn(ram)
        outputs = net.forward(inputs)
        buttons = outputs_to_buttons(outputs)

        if len(buttons) < action_size:
            buttons = buttons + [0] * (action_size - len(buttons))
        elif len(buttons) > action_size:
            buttons = buttons[:action_size]

        obs, *_ = env.step(np.array(buttons, dtype=np.int8))

        ram = env.get_ram()
        values = schema.read(ram)
        config.apply_computed(values)

        level_id = values.get("level_id", 0)
        lives = values.get("lives", 0)
        in_sub = level_id != 0 and level_id not in main_ids

        if not in_sub:
            px = float(values.get("player_x", values.get("camera_x", 0)))
            if px > max_progress:
                max_progress = px

        write_video_frame(obs, label=seg_label, status_text="NEURO", status_color=(255, 200, 0))

        # Completion check
        if config.completion_signal == "ram_flag":
            flag_val = values.get(config.completion_ram_key, None)
            if (flag_val is not None
                    and flag_val == config.completion_ram_value
                    and max_progress >= config.completion_min_progress):
                completed = True
                break
        elif config.completion_signal == "level_id_change":
            if level_id not in main_ids and level_id != 0:
                is_real = (
                    max_progress >= config.completion_min_progress
                    and (not config.completion_level_ids
                         or level_id in config.completion_level_ids)
                    and level_id not in config.completion_exclude_ids
                )
                if is_real:
                    completed = True
                    break

        # Death check
        if initial_lives is not None and lives < initial_lives:
            died = True
            break

    return completed, died, frame_idx + 1, max_progress


def chain_live(
    route: RouteConfig,
    *,
    save_states: bool = False,
    verbose: bool = True,
    pad_frames: int = 100,
    transition_max_frames: int = 600,
    video_path: str | None = None,
    video_scale: int = 3,
) -> ChainLiveResult:
    """Play all route segments on a SINGLE emulator instance with no state reloads.

    Unlike evaluate_route() which tests each segment independently, this runs
    the entire game end-to-end in one emulator session. Recordings are replayed
    sequentially on the live emulator, with automatic transition handling between
    segments.

    Args:
        route: The route to play.
        save_states: Save chained states to custom_integrations at each segment boundary.
        verbose: Print per-segment status.
        pad_frames: Extra no-input frames after each recording to catch delayed transitions.
        transition_max_frames: Max frames to wait for level transition between segments.
        video_path: If set, pipe frames to ffmpeg to produce an MP4.
        video_scale: Pixel scaling for video output.
    """
    from retro_harness.platformer.actions import action_index_to_buttons, DEFAULT_PLATFORMER_ACTIONS
    from retro_harness.env import make_env

    result = ChainLiveResult(route=route)

    if not route.segments:
        return result

    # Boot the emulator from the first segment's start state
    first_config = get_level_config(route.segments[0].config_id)
    env = make_env(
        first_config.game_name,
        first_config.start_state,
        first_config.game_dir,
        render_mode="rgb_array",
    )
    env.reset()
    action_size = env.action_space.shape[0]
    no_input = np.zeros(action_size, dtype=np.int8)

    # Video setup
    ffmpeg_proc = None
    if video_path:
        from retro_harness.platformer.record_video import draw_text

        obs = env.render()
        h, w = obs.shape[0], obs.shape[1]
        out_h, out_w = h * video_scale, w * video_scale
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{out_w}x{out_h}",
            "-pix_fmt", "rgb24", "-r", "60",
            "-i", "-",
            "-c:v", "libx264", "-preset", "slow", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            video_path,
        ]
        ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def write_video_frame(obs_frame, label="", status_text="", status_color=(255, 255, 255)):
        if ffmpeg_proc is None:
            return
        from retro_harness.platformer.record_video import draw_text
        frame = np.repeat(np.repeat(obs_frame, video_scale, axis=0), video_scale, axis=1).copy()
        secs = result.total_frames / 60.0
        draw_text(frame, f"F:{result.total_frames}", 4, 4)
        draw_text(frame, f"T:{secs:.1f}S", 4, 12, (200, 200, 200))
        if label:
            draw_text(frame, label, 4, 20, (100, 255, 100))
        if status_text:
            draw_text(frame, status_text, frame.shape[1] - len(status_text) * 5 - 4, 4, status_color)
        try:
            ffmpeg_proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            pass

    if verbose:
        print(f"Chain-live: {route.display_name} ({len(route.segments)} segments)")
        print(f"{'='*80}\n")

    cumulative = 0

    for seg_idx, seg in enumerate(route.segments):
        try:
            config = get_level_config(seg.config_id)
        except KeyError as e:
            sr = ChainLiveSegmentResult(seg, None, None, "ERROR", error=str(e))  # type: ignore[arg-type]
            result.segments.append(sr)
            if verbose:
                print(f"  [{seg_idx}] {seg.label}: CONFIG ERROR: {e}")
            break

        # Find recording
        if seg.recording:
            rec_path = Path(seg.recording)
            if not rec_path.is_absolute():
                rec_path = config.runs_dir / seg.recording
        else:
            rec_path = find_best_recording(config)

        if rec_path is None or not rec_path.exists():
            sr = ChainLiveSegmentResult(seg, config, rec_path, "NO_RECORDING")
            result.segments.append(sr)
            if verbose:
                print(f"  [{seg_idx}] {seg.label}: NO RECORDING")
            break

        # Read current RAM state before replay
        schema = config.ram_schema
        ram = env.get_ram()
        pre_values = schema.read(ram)
        config.apply_computed(pre_values)
        initial_lives = pre_values.get("lives")

        if verbose:
            level_id = pre_values.get("level_id", 0)
            print(f"  [{seg_idx}] {seg.label}: replaying {rec_path.name}")
            print(f"       state: level_id=0x{level_id:04X} lives={initial_lives} "
                  f"x={pre_values.get('player_x', '?')}")

        # Save chained state to disk, then reload the env from it.
        # Recordings are optimized (by chain_optimize / Evaluator) from saved
        # states, so the env must be initialized from the same state file to
        # ensure byte-identical emulator + wrapper state. Simply calling
        # set_state() on a running env doesn't reset gymnasium-level wrapper
        # state (initial_state, data lookup caches, etc.), causing desync.
        state_data = env.em.get_state()
        states_dir = config.game_dir / "custom_integrations" / config.game_name
        safe_label = (seg.label or seg.config_id).replace(" ", "_").replace("/", "_")
        safe_label = safe_label.replace("(", "").replace(")", "").replace("→", "to")
        state_name = f"Chained_{safe_label}"

        # Always save the state (needed for env reload); delete later if !save_states
        out_path = states_dir / f"{state_name}.state"
        with gzip.open(out_path, "wb") as f:
            f.write(state_data)
        if verbose and save_states:
            print(f"       saved state: {out_path.name}")

        # Reload env from the saved state for segments after the first.
        # This ensures the env is initialized identically to how the Evaluator
        # would load it, preventing desync from accumulated wrapper state.
        if seg_idx > 0:
            env.close()
            env = make_env(config.game_name, state_name, config.game_dir,
                           render_mode="rgb_array")
            env.reset()
            action_size = env.action_space.shape[0]
            no_input = np.zeros(action_size, dtype=np.int8)

        if not save_states:
            out_path.unlink(missing_ok=True)

        # Load recording
        actions, is_raw = load_recording_data(rec_path)

        # Pad with no-input frames
        if is_raw:
            btn_len = len(actions[0]) if actions else action_size
            actions = list(actions) + [[0] * btn_len] * pad_frames
        else:
            actions = list(actions) + [0] * pad_frames

        action_table = config.action_table or DEFAULT_PLATFORMER_ACTIONS

        # Build completion detection state (mirrors Evaluator logic)
        main_ids = {config.target_level_id} | set(config.level_id_aliases)
        max_progress = 0.0
        seg_frames = 0
        completed = False
        died = False

        for frame_idx, action in enumerate(actions):
            if is_raw:
                buttons = list(action)
            else:
                buttons = action_index_to_buttons(action, action_table)
            if len(buttons) < action_size:
                buttons = buttons + [0] * (action_size - len(buttons))
            elif len(buttons) > action_size:
                buttons = buttons[:action_size]

            obs, *_ = env.step(np.array(buttons, dtype=np.int8))
            seg_frames += 1

            ram = env.get_ram()
            values = schema.read(ram)
            config.apply_computed(values)

            level_id = values.get("level_id", 0)
            lives = values.get("lives", 0)
            in_sub = level_id != 0 and level_id not in main_ids

            # Track progress (only in main level)
            if not in_sub:
                px = float(values.get("player_x", values.get("camera_x", 0)))
                if px > max_progress:
                    max_progress = px

            # Write video frame
            write_video_frame(obs, label=seg.label)

            # Completion check (same logic as Evaluator)
            if config.completion_signal == "level_id_change":
                if level_id not in main_ids and level_id != 0:
                    is_real = (
                        max_progress >= config.completion_min_progress
                        and (not config.completion_level_ids
                             or level_id in config.completion_level_ids)
                        and level_id not in config.completion_exclude_ids
                    )
                    if is_real:
                        completed = True
                        break
            elif config.completion_signal == "ram_flag":
                flag_val = values.get(config.completion_ram_key, None)
                if (flag_val is not None
                        and flag_val == config.completion_ram_value
                        and max_progress >= config.completion_min_progress):
                    completed = True
                    break

            # Death check
            if initial_lives is not None and lives < initial_lives:
                died = True
                break

        cumulative += seg_frames
        result.total_frames = cumulative

        if completed:
            sr = ChainLiveSegmentResult(seg, config, rec_path, "COMPLETED", frames=seg_frames)
            result.segments.append(sr)
            if verbose:
                print(f"       COMPLETED in {seg_frames} frames ({seg_frames/60:.1f}s) "
                      f"progress={max_progress:.0f}")

            # Write a few "DONE" video frames
            for _ in range(30):
                obs, *_ = env.step(no_input)
                write_video_frame(obs, label=seg.label, status_text="DONE", status_color=(0, 255, 0))

            # Transition: step no-input frames until the NEXT segment's level appears
            if seg_idx + 1 < len(route.segments):
                next_seg = route.segments[seg_idx + 1]
                try:
                    next_config = get_level_config(next_seg.config_id)
                except KeyError:
                    # Can't look up next config; just step some frames
                    for _ in range(120):
                        obs, *_ = env.step(no_input)
                        cumulative += 1
                        write_video_frame(obs, label="TRANSITION")
                    result.total_frames = cumulative
                    continue

                next_main_ids = {next_config.target_level_id} | set(next_config.level_id_aliases)
                next_schema = next_config.ram_schema
                transitioned = False

                if verbose:
                    print(f"       waiting for transition to {next_seg.label} "
                          f"(target_id=0x{next_config.target_level_id:04X})...")

                for t in range(transition_max_frames):
                    obs, *_ = env.step(no_input)
                    cumulative += 1
                    write_video_frame(obs, label="TRANSITION")

                    ram = env.get_ram()
                    t_values = next_schema.read(ram)
                    next_config.apply_computed(t_values)
                    t_level_id = t_values.get("level_id", 0)

                    if t_level_id in next_main_ids:
                        # Level ID matched — but we may still be in a transition
                        # animation (flagpole, score tally, pipe, etc.) where
                        # the level_id and player_x can be valid but gameplay
                        # hasn't started. Wait for player position to RESET
                        # (go to 0) then RE-APPEAR (back to > 0). This catches
                        # the actual level load cycle.
                        settle = 0
                        saw_reset = False
                        for settle in range(600):
                            obs, *_ = env.step(no_input)
                            cumulative += 1
                            write_video_frame(obs, label=next_seg.label)
                            ram = env.get_ram()
                            sv = next_schema.read(ram)
                            next_config.apply_computed(sv)
                            px = sv.get("player_x", sv.get("camera_x", 0))
                            py = sv.get("player_y", 0)

                            if not saw_reset:
                                # Phase 1: wait for position to reset to 0
                                # (level load / transition clears player position)
                                if px == 0 and py == 0:
                                    saw_reset = True
                            else:
                                # Phase 2: wait for player to spawn (position > 0)
                                if px > 0 or py > 0:
                                    # Player spawned — settle a few more frames
                                    for _ in range(10):
                                        obs, *_ = env.step(no_input)
                                        cumulative += 1
                                        write_video_frame(obs, label=next_seg.label)
                                    break
                        transitioned = True
                        if verbose:
                            print(f"       transition OK after {t + settle + 11} frames "
                                  f"(spawn wait={settle}, reset={'yes' if saw_reset else 'no'})")
                        break

                result.total_frames = cumulative

                if not transitioned:
                    # Read final state for diagnostics
                    ram = env.get_ram()
                    t_values = next_schema.read(ram)
                    next_config.apply_computed(t_values)
                    t_level_id = t_values.get("level_id", 0)
                    if verbose:
                        print(f"       TRANSITION FAILED after {transition_max_frames} frames "
                              f"(level_id=0x{t_level_id:04X}, expected 0x{next_config.target_level_id:04X})")
                    # Mark current segment as transition failed
                    sr.status = "TRANSITION_FAILED"
                    break

        elif died:
            sr = ChainLiveSegmentResult(seg, config, rec_path, "DIED", frames=seg_frames)
            result.segments.append(sr)
            if verbose:
                print(f"       DIED at frame {seg_frames} (progress={max_progress:.0f})")
            # Write death frames to video
            for _ in range(60):
                obs, *_ = env.step(no_input)
                write_video_frame(obs, label=seg.label, status_text="DEAD", status_color=(255, 0, 0))
            break

        else:
            # Didn't complete or die - ran out of recording frames.
            # If a neuro checkpoint is specified, switch to neural network live play.
            neuro_path = None
            if seg.neuro_checkpoint:
                np_candidate = config.runs_dir / seg.neuro_checkpoint
                if np_candidate.exists():
                    neuro_path = np_candidate

            if neuro_path is not None:
                if verbose:
                    print(f"       recording ended at frame {seg_frames}, switching to neuro live...")
                neuro_completed, neuro_died, neuro_frames, max_progress = _run_neuro_live(
                    env, config, neuro_path, schema, main_ids, initial_lives,
                    max_progress, write_video_frame, seg.label, verbose,
                )
                seg_frames += neuro_frames
                if neuro_completed:
                    completed = True
                elif neuro_died:
                    died = True

                cumulative += neuro_frames
                result.total_frames = cumulative

                if completed:
                    sr = ChainLiveSegmentResult(seg, config, rec_path, "COMPLETED", frames=seg_frames)
                    result.segments.append(sr)
                    if verbose:
                        print(f"       COMPLETED (neuro) in {seg_frames} frames ({seg_frames/60:.1f}s) "
                              f"progress={max_progress:.0f}")
                    for _ in range(30):
                        obs, *_ = env.step(no_input)
                        write_video_frame(obs, label=seg.label, status_text="DONE", status_color=(0, 255, 0))
                    # No transition handling needed - 8-4 is the last segment
                elif died:
                    sr = ChainLiveSegmentResult(seg, config, rec_path, "DIED", frames=seg_frames)
                    result.segments.append(sr)
                    if verbose:
                        print(f"       DIED (neuro) at frame {seg_frames} (progress={max_progress:.0f})")
                    for _ in range(60):
                        obs, *_ = env.step(no_input)
                        write_video_frame(obs, label=seg.label, status_text="DEAD", status_color=(255, 0, 0))
                    break
                else:
                    sr = ChainLiveSegmentResult(seg, config, rec_path, "INCOMPLETE", frames=seg_frames)
                    result.segments.append(sr)
                    if verbose:
                        print(f"       INCOMPLETE (neuro exhausted) at frame {seg_frames}")
                    break
            else:
                sr = ChainLiveSegmentResult(seg, config, rec_path, "INCOMPLETE", frames=seg_frames)
                result.segments.append(sr)
                if verbose:
                    ram = env.get_ram()
                    values = schema.read(ram)
                    config.apply_computed(values)
                    level_id = values.get("level_id", 0)
                    print(f"       INCOMPLETE after {seg_frames} frames "
                          f"(level_id=0x{level_id:04X}, progress={max_progress:.0f})")
                break

    # Finalize video
    if ffmpeg_proc is not None:
        try:
            ffmpeg_proc.stdin.close()
        except BrokenPipeError:
            pass
        ffmpeg_proc.wait()
        if video_path:
            vp = Path(video_path)
            if vp.exists():
                size_mb = vp.stat().st_size / (1024 * 1024)
                if verbose:
                    print(f"\nVideo: {video_path} ({size_mb:.1f} MB)")

    env.close()

    # Summary
    if verbose:
        print(f"\n{'='*80}")
        print(f"Chain-live results: {result.completed_count}/{len(route.segments)} segments completed")
        print(f"Total frames: {result.total_frames} ({result.total_frames/60:.1f}s)")
        print()
        icons = {"COMPLETED": "+", "DIED": "X", "TRANSITION_FAILED": "!", "NO_RECORDING": "?",
                 "INCOMPLETE": "-", "ERROR": "E"}
        for sr in result.segments:
            icon = icons.get(sr.status, "?")
            print(f"  {icon} {sr.segment.label:<20s}  {sr.status:<22s}  {sr.frames}f ({sr.frames/60:.1f}s)")

        # Show next step hints
        failed = [s for s in result.segments if s.status != "COMPLETED"]
        if failed:
            f = failed[0]
            print(f"\nNext step: fix {f.segment.label} ({f.status})")
            if f.status == "TRANSITION_FAILED":
                print(f"  The recording completed but the game transitioned to the wrong level.")
                print(f"  Re-record this segment from its chained state.")
            elif f.status == "DIED":
                print(f"  Re-record from chained state with a safer route.")
            elif f.status == "NO_RECORDING":
                print(f"  Record this segment:")
                if f.config:
                    print(f"    uv run python -m retro_harness.platformer -l {f.config.level_id} play")
            elif f.status == "INCOMPLETE":
                print(f"  Recording didn't reach completion. Re-record or extend.")

    return result


def _find_alignment(
    config: LevelConfig,
    state_name: str,
    actions: list,
    is_raw: bool,
    *,
    max_pad: int = 120,
    step: int = 1,
    pad_action: int = 0,
    action_table: list | None = None,
) -> tuple[int, float] | None:
    """Try prepending pad_action frames (0..max_pad) to find alignment that completes.

    pad_action=0 gives NOOP padding (standing still).
    pad_action=2 (or whatever "run right" is) gives running-right padding.

    Returns (best_pad, fitness) or None if no padding completes.
    """
    best_pad = None
    best_fitness = -1e9

    for pad in range(0, max_pad + 1, step):
        if is_raw:
            # Convert pad_action index to raw button frame
            btn_len = len(actions[0]) if actions else 9
            if pad_action == 0:
                pad_frame = [0] * btn_len
            elif action_table and pad_action < len(action_table):
                pad_frame = list(action_table[pad_action])[:btn_len]
                if len(pad_frame) < btn_len:
                    pad_frame += [0] * (btn_len - len(pad_frame))
            else:
                pad_frame = [0] * btn_len
            padded = [pad_frame] * pad + list(actions)
        else:
            padded = [pad_action] * pad + list(actions)

        ev = Evaluator(config, start_state=state_name)
        r = ev.evaluate(padded, early_terminate=False)
        ev.close()

        if r.completed and r.fitness > best_fitness:
            best_fitness = r.fitness
            best_pad = pad

    return (best_pad, best_fitness) if best_pad is not None else None


def chain_optimize(
    route: RouteConfig,
    *,
    iterations: int = 2000,
    verbose: bool = True,
) -> ChainLiveResult:
    """Iteratively fix each segment from chained states until the full route works.

    Strategy per broken segment:
    1. NOOP-pad scan: try prepending 0-120 NOOP frames to the recording.
       Chained states often differ from standalone states only in player position;
       NOOP frames let the player reach the expected starting position.
    2. If padding alone completes the segment, hill-climb with that padding
       for speed optimization.
    3. If no padding works, hill-climb the raw seed from the chained state.

    Repeats until all segments chain or a segment can't be fixed.
    """
    from retro_harness.platformer.actions import (
        DEFAULT_PLATFORMER_ACTIONS,
        action_index_to_buttons,
        buttons_to_action_index,
    )
    from retro_harness.platformer.hillclimb import hillclimb

    max_rounds = len(route.segments) * 2  # safety limit

    for round_num in range(max_rounds):
        print(f"\n{'#'*80}")
        print(f"# Chain-optimize round {round_num + 1}")
        print(f"{'#'*80}\n")

        # Run chain-live with state saving
        result = chain_live(route, save_states=True, verbose=verbose)

        if result.all_completed:
            print(f"\nAll {len(route.segments)} segments completed!")
            print(f"Total: {result.total_frames}f ({result.total_frames/60:.1f}s)")
            return result

        # Find first non-completed segment
        failed = [s for s in result.segments if s.status != "COMPLETED"]
        if not failed:
            return result

        f = failed[0]
        seg = f.segment
        config = f.config

        if config is None:
            print(f"\nCannot fix {seg.label}: no config")
            return result

        if f.status == "NO_RECORDING":
            print(f"\nCannot fix {seg.label}: no recording to use as seed")
            print(f"  Record it: uv run python -m retro_harness.platformer -l {config.level_id} play")
            return result

        if f.status == "TRANSITION_FAILED":
            print(f"\nCannot auto-fix {seg.label}: transition failed (wrong exit?)")
            print(f"  This likely needs a manual re-record with different routing.")
            return result

        # Find the chained state for this segment
        safe_label = (seg.label or seg.config_id).replace(" ", "_").replace("/", "_")
        safe_label = safe_label.replace("(", "").replace(")", "").replace("→", "to")
        state_name = f"Chained_{safe_label}"
        states_dir = config.game_dir / "custom_integrations" / config.game_name
        state_path = states_dir / f"{state_name}.state"

        if not state_path.exists():
            print(f"\nNo chained state for {seg.label}: {state_path}")
            return result

        print(f"\nFixing {seg.label} from chained state: {state_name}")
        print(f"  Status was: {f.status} (frames={f.frames})")

        # Load seed recording
        rec_path = f.recording_path
        if rec_path is None:
            print(f"  No recording path for {seg.label}")
            return result

        actions_data, is_raw = load_recording_data(rec_path)
        action_table = config.action_table or DEFAULT_PLATFORMER_ACTIONS

        # Build list of recordings to try for alignment
        alt_recordings: list[tuple[str, list, bool]] = [("primary", actions_data, is_raw)]
        rec_000 = config.runs_dir / "recording_000.json"
        if rec_000.exists() and rec_000 != rec_path:
            alt_data, alt_raw = load_recording_data(rec_000)
            alt_recordings.append(("recording_000", alt_data, alt_raw))
        # Also try previous chained hill-climb result (incremental improvement)
        chained_hc = config.runs_dir / "chained" / "hillclimb_best_final.json"
        if chained_hc.exists() and chained_hc != rec_path:
            chained_data, chained_raw = load_recording_data(chained_hc)
            alt_recordings.append(("chained_hc", chained_data, chained_raw))

        # Phase 1: Alignment scan (fast — each eval takes ~0.5s, scan up to 120)
        # Try NOOP padding first, then running-right padding if NOOP fails.
        best_pad_result = None
        best_pad_source = None
        best_pad_action = 0  # 0=NOOP, run_right_idx for running

        # Find the "run right" action index (RIGHT+B)
        run_right_idx = 1  # fallback: RIGHT/walk
        for idx, btns in enumerate(action_table):
            active = {i for i, b in enumerate(btns) if b}
            if 7 in active and 0 in active and 8 not in active:  # RIGHT+B, no jump
                run_right_idx = idx
                break

        for source_name, acts, raw in alt_recordings:
            # Try NOOP alignment
            print(f"  Scanning NOOP alignment for {source_name} (0-120 frames)...")
            pad_result = _find_alignment(
                config, state_name, acts, raw, max_pad=120, step=1,
                pad_action=0, action_table=action_table,
            )
            if pad_result is not None:
                pad, fitness = pad_result
                print(f"    Found: pad={pad} frames, fitness={fitness:.0f}")
                if best_pad_result is None or fitness > best_pad_result[1]:
                    best_pad_result = pad_result
                    best_pad_source = (source_name, acts, raw)
                    best_pad_action = 0
            else:
                # Try running-right alignment (player runs forward to match position)
                print(f"    NOOP failed. Trying run-right alignment (0-120 frames)...")
                pad_result = _find_alignment(
                    config, state_name, acts, raw, max_pad=120, step=1,
                    pad_action=run_right_idx, action_table=action_table,
                )
                if pad_result is not None:
                    pad, fitness = pad_result
                    print(f"    Found: run-right pad={pad} frames, fitness={fitness:.0f}")
                    if best_pad_result is None or fitness > best_pad_result[1]:
                        best_pad_result = pad_result
                        best_pad_source = (source_name, acts, raw)
                        best_pad_action = run_right_idx
                else:
                    print(f"    No alignment found for {source_name}")

        if best_pad_result is not None:
            pad, fitness = best_pad_result
            source_name, acts, raw = best_pad_source

            # Build padded action-index sequence
            if raw:
                padded_indices = (
                    [best_pad_action] * pad
                    + [buttons_to_action_index(frame, action_table=action_table) for frame in acts]
                )
            else:
                padded_indices = [best_pad_action] * pad + list(acts)

            pad_type = "NOOP" if best_pad_action == 0 else "run-right"
            print(f"  Using {source_name} with {pad} {pad_type} pad ({len(padded_indices)} total frames)")

            # Phase 2: Hill-climb the padded sequence for speed
            if iterations > 0:
                print(f"  Hill-climbing for {iterations} iterations...")
                evaluator = Evaluator(config, start_state=state_name)
                best_actions, best_result = hillclimb(
                    actions=padded_indices,
                    evaluator=evaluator,
                    max_iterations=iterations,
                    output_dir=config.runs_dir / "chained",
                    verbose=verbose,
                )
                evaluator.close()
            else:
                best_actions = padded_indices
                ev = Evaluator(config, start_state=state_name)
                best_result = ev.evaluate(padded_indices, early_terminate=False)
                ev.close()
            best_actions_raw = False

        else:
            # No alignment works — check for practice recordings to use GA
            practice_dir = config.runs_dir / "practice"
            practice_seeds = _load_practice_seeds(practice_dir) if practice_dir.exists() else []

            if len(practice_seeds) >= 2:
                # Preserve faithful practice input whenever paired raw files
                # exist. Legacy indexed seeds can join the raw population via
                # their exact action-table representation.
                from retro_harness.platformer.genetic import run_ga, run_ga_raw

                print(f"  No alignment found. Using GA with {len(practice_seeds)} practice seeds")
                evaluator = Evaluator(config, start_state=state_name)
                best_actions_raw = any(is_raw_seed for _, is_raw_seed in practice_seeds)
                if best_actions_raw:
                    raw_seeds = [
                        frames
                        if is_raw_seed
                        else [
                            action_index_to_buttons(action, action_table=action_table)
                            for action in frames
                        ]
                        for frames, is_raw_seed in practice_seeds
                    ]
                    ga_best = run_ga_raw(
                        seeds=raw_seeds,
                        evaluator=evaluator,
                        output_dir=config.runs_dir / "chained",
                        verbose=verbose,
                    )
                else:
                    indexed_seeds = [frames for frames, _ in practice_seeds]
                    ga_best = run_ga(
                        seed_actions=indexed_seeds,
                        evaluator=evaluator,
                        output_dir=config.runs_dir / "chained",
                        verbose=verbose,
                    )
                best_actions = ga_best.actions
                best_result = ga_best.result if ga_best.result else evaluator.evaluate(best_actions)

                # If GA found completion, hill-climb for speed
                if best_result.completed and iterations > 0:
                    print(f"  GA completed! Hill-climbing for speed ({iterations} iters)...")
                    if best_actions_raw:
                        from retro_harness.platformer.hillclimb_raw import hillclimb_raw

                        best_actions, best_result = hillclimb_raw(
                            raw_buttons=best_actions,
                            evaluator=evaluator,
                            max_iterations=iterations,
                            output_dir=config.runs_dir / "chained",
                            verbose=verbose,
                        )
                    else:
                        best_actions, best_result = hillclimb(
                            actions=best_actions,
                            evaluator=evaluator,
                            max_iterations=iterations,
                            output_dir=config.runs_dir / "chained",
                            verbose=verbose,
                        )
                evaluator.close()
            else:
                # Fallback: hill climb from best available seed.
                # Prefer previous chained result (incremental) over raw recording.
                best_seed_name = "primary"
                best_seed_acts = actions_data
                best_seed_raw = is_raw
                best_seed_fitness = -1e9

                for sname, sacts, sraw in alt_recordings:
                    ev = Evaluator(config, start_state=state_name)
                    r = ev.evaluate(sacts, early_terminate=False)
                    ev.close()
                    if r.fitness > best_seed_fitness:
                        best_seed_fitness = r.fitness
                        best_seed_name = sname
                        best_seed_acts = sacts
                        best_seed_raw = sraw

                if best_seed_raw:
                    seed_indices = [
                        buttons_to_action_index(frame, action_table=action_table)
                        for frame in best_seed_acts
                    ]
                else:
                    seed_indices = list(best_seed_acts)

                # Extend short recordings with "run right" frames
                min_seed_len = 3000
                if len(seed_indices) < min_seed_len:
                    extension = min_seed_len - len(seed_indices)
                    seed_indices = seed_indices + [run_right_idx] * extension
                    print(f"  No alignment found. Using {best_seed_name} (fitness={best_seed_fitness:.0f}), "
                          f"extended to {len(seed_indices)}f (+{extension} run-right)")
                else:
                    print(f"  No alignment found. Using {best_seed_name} (fitness={best_seed_fitness:.0f})")
                print(f"  Hill-climbing from seed ({iterations} iters)...")

                evaluator = Evaluator(config, start_state=state_name)
                best_actions, best_result = hillclimb(
                    actions=seed_indices,
                    evaluator=evaluator,
                    max_iterations=iterations,
                    output_dir=config.runs_dir / "chained",
                    verbose=verbose,
                )
                evaluator.close()
                best_actions_raw = False

        if not best_result.completed:
            # Save partial progress to chained/ for incremental improvement
            partial_path = config.runs_dir / "chained" / "hillclimb_best_final.json"
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            partial_data = {
                "raw_buttons" if best_actions_raw else "actions": best_actions,
                "num_frames": len(best_actions),
                "fitness": best_result.fitness,
                "completed": False,
                "total_frames": best_result.total_frames,
                "max_progress": best_result.max_progress,
                "level": config.level_id,
                "source": "chain_optimize_partial",
                "chained_state": state_name,
            }
            partial_path.write_text(json.dumps(partial_data, indent=2))
            print(f"\n  Could NOT complete {seg.label} (progress={best_result.max_progress:.0f})")
            print(f"  Saved partial result to {partial_path}")
            print(f"  Re-run chain-optimize to continue from this point, or re-record:")
            print(f"    uv run python -m retro_harness.platformer -l {config.level_id} play --state {state_name}")
            return result

        print(f"\n  Completed {seg.label}!")
        print(f"  Frames: {best_result.total_frames} ({best_result.total_frames/60:.1f}s)")

        # Save as the primary hillclimb result so chain-live picks it up
        optimized_path = config.runs_dir / "hillclimb_best_final.json"
        data = {
            "raw_buttons" if best_actions_raw else "actions": best_actions,
            "num_frames": len(best_actions),
            "fitness": best_result.fitness,
            "completed": best_result.completed,
            "total_frames": best_result.total_frames,
            "max_x": best_result.max_x,
            "max_progress": best_result.max_progress,
            "bonus_frames": best_result.bonus_frames,
            "level": config.level_id,
            "source": "chain_optimize",
            "chained_state": state_name,
        }
        optimized_path.parent.mkdir(parents=True, exist_ok=True)
        optimized_path.write_text(json.dumps(data, indent=2))
        print(f"  Saved: {optimized_path}")

    print(f"\nExhausted {max_rounds} rounds without completing the full chain.")
    return result


# -- Route video rendering ---------------------------------------------------

def record_route_video(
    route: RouteConfig,
    output: str,
    *,
    scale: int = 3,
    completion_hold: int = 60,
) -> None:
    """Render the full route as a single MP4.

    Each segment is played in its own env from its saved state.
    Segments are stitched together with a brief label overlay between them.
    """
    from retro_harness.platformer.actions import action_index_to_buttons, DEFAULT_PLATFORMER_ACTIONS
    from retro_harness.platformer.record_video import draw_text
    from retro_harness.env import make_env

    # First pass: figure out video dimensions from first segment
    seg0 = route.segments[0]
    cfg0 = get_level_config(seg0.config_id)
    env0 = make_env(cfg0.game_name, cfg0.start_state, cfg0.game_dir, render_mode="rgb_array")
    obs0, _ = env0.reset()
    h, w = obs0.shape[0], obs0.shape[1]
    env0.close()

    out_h, out_w = h * scale, w * scale
    cumulative_frames = 0

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}",
        "-pix_fmt", "rgb24", "-r", "60",
        "-i", "-",
        "-c:v", "libx264", "-preset", "slow", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        output,
    ]

    print(f"Recording route: {route.display_name} ({len(route.segments)} segments)")
    print(f"Output: {output} at {out_w}x{out_h}")

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def write_frame(f):
        try:
            proc.stdin.write(f.tobytes())
        except BrokenPipeError:
            pass

    def write_title_card(text: str, frames: int = 90):
        """Write a black frame with centered text for N frames."""
        frame = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        cx = out_w // 2 - len(text) * 5 // 2
        cy = out_h // 2 - 3
        draw_text(frame, text, cx, cy, (255, 255, 255), shadow=False)
        for _ in range(frames):
            write_frame(frame)

    for i, seg in enumerate(route.segments):
        try:
            config = get_level_config(seg.config_id)
        except KeyError:
            print(f"  [{i}] {seg.label}: CONFIG ERROR, skipping")
            continue

        # Find recording
        if seg.recording:
            rec_path = Path(seg.recording)
            if not rec_path.is_absolute():
                rec_path = config.runs_dir / seg.recording
        else:
            rec_path = find_best_recording(config)

        if rec_path is None or not rec_path.exists():
            print(f"  [{i}] {seg.label}: NO RECORDING, skipping")
            continue

        actions, is_raw = load_recording_data(rec_path)
        action_table = config.action_table or DEFAULT_PLATFORMER_ACTIONS

        # Title card between segments
        write_title_card(seg.label or config.display_name)

        env = make_env(config.game_name, config.start_state, config.game_dir, render_mode="rgb_array")
        obs, _ = env.reset()
        action_size = env.action_space.shape[0]

        schema = config.ram_schema
        ram = env.get_ram()
        initial_values = schema.read(ram)
        config.apply_computed(initial_values)
        initial_lives = initial_values.get("lives")
        _main_ids = {config.target_level_id} | set(config.level_id_aliases)
        max_progress = 0.0
        seg_frames = 0

        print(f"  [{i}] {seg.label}: {len(actions)} frames from {rec_path.name}")

        for frame_idx, action in enumerate(actions):
            if is_raw:
                buttons = list(action)
            else:
                buttons = action_index_to_buttons(action, action_table)
            if len(buttons) < action_size:
                buttons = buttons + [0] * (action_size - len(buttons))
            elif len(buttons) > action_size:
                buttons = buttons[:action_size]

            obs, *_ = env.step(np.array(buttons, dtype=np.int8))

            ram = env.get_ram()
            values = schema.read(ram)
            config.apply_computed(values)
            level_id = values.get("level_id", 0)
            lives = values.get("lives", 0)
            in_sub = level_id != 0 and level_id not in _main_ids

            if not in_sub:
                px = float(values.get("player_x", values.get("camera_x", 0)))
                if px > max_progress:
                    max_progress = px

            # Check completion
            completed = False
            if config.completion_signal == "level_id_change":
                if level_id not in _main_ids and level_id != 0:
                    if (max_progress >= config.completion_min_progress
                            and (not config.completion_level_ids or level_id in config.completion_level_ids)
                            and level_id not in config.completion_exclude_ids):
                        completed = True
            elif config.completion_signal == "ram_flag":
                flag_val = values.get(config.completion_ram_key, None)
                if (flag_val is not None
                        and flag_val == config.completion_ram_value
                        and max_progress >= config.completion_min_progress):
                    completed = True

            died = initial_lives is not None and lives < initial_lives

            # Scale and annotate frame
            frame = np.repeat(np.repeat(obs, scale, axis=0), scale, axis=1).copy()
            secs = (cumulative_frames + frame_idx) / 60.0
            draw_text(frame, f"F:{cumulative_frames + frame_idx}", 4, 4)
            draw_text(frame, f"T:{secs:.1f}S", 4, 12, (200, 200, 200))
            draw_text(frame, seg.label or config.display_name, 4, 20, (100, 255, 100))

            if completed:
                draw_text(frame, "DONE", out_w - 30, 4, (0, 255, 0))
            elif died:
                draw_text(frame, "DEAD", out_w - 30, 4, (255, 0, 0))

            write_frame(frame)
            seg_frames = frame_idx + 1

            if completed:
                for _ in range(completion_hold):
                    obs, *_ = env.step(np.zeros(action_size, dtype=np.int8))
                    frame = np.repeat(np.repeat(obs, scale, axis=0), scale, axis=1).copy()
                    draw_text(frame, "DONE", out_w - 30, 4, (0, 255, 0))
                    write_frame(frame)
                break

            if died:
                for _ in range(60):
                    obs, *_ = env.step(np.zeros(action_size, dtype=np.int8))
                    frame = np.repeat(np.repeat(obs, scale, axis=0), scale, axis=1).copy()
                    draw_text(frame, "DEAD", out_w - 30, 4, (255, 0, 0))
                    write_frame(frame)
                break

        cumulative_frames += seg_frames
        env.close()

    try:
        proc.stdin.close()
    except BrokenPipeError:
        pass
    proc.wait()

    out_path = Path(output)
    if out_path.exists():
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"\nDone! {output} ({size_mb:.1f} MB, {cumulative_frames}f / {cumulative_frames/60:.1f}s)")
