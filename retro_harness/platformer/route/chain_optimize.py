"""Iterative chain-optimize: hill-climb segments from chained states."""

from __future__ import annotations

import json
from pathlib import Path

from retro_harness.platformer.evaluator import Evaluator
from retro_harness.platformer.level_config import LevelConfig, get_level_config
from retro_harness.platformer.route.chain_live import (
    ChainLiveResult,
    ChainLiveSegmentResult,
)
from retro_harness.platformer.route.models import (
    RouteConfig,
    find_best_recording,
    load_recording_data,
)


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


