"""Manual play / record command."""

from __future__ import annotations

import argparse
import json
import time

from retro_harness.platformer.actions import (
    action_index_to_buttons,
    buttons_to_action_index,
)
from retro_harness.platformer.bk2_extract import save_actions
from retro_harness.platformer.cli.helpers import _get_action_table, _resolve_config


def _require_playable_state(config, start_state: str) -> None:
    """Fail with a clear message when a custom start state file is missing."""
    if not start_state or start_state in ("NONE", "none"):
        return
    from pathlib import Path

    from retro_harness.env import add_custom_integrations, state_path

    custom = Path(state_path(config.game_dir, config.game_name, start_state))
    if custom.is_file():
        return
    # Fall back to stable-retro / registered integrations.
    try:
        import stable_retro as retro

        add_custom_integrations(config.game_dir)
        resolved = retro.data.get_file_path(
            config.game_name,
            f"{start_state}.state",
            inttype=retro.data.Integrations.ALL,
        )
        if resolved:
            return
    except Exception:
        pass
    raise SystemExit(
        f"Missing start state {start_state!r} for {config.level_id}.\n"
        f"  Expected custom file: {custom}\n"
        f"  Capture with: uv run python -m <game> capture-state "
        f"--from <anchor> --name {start_state}\n"
        f"  (SMW Iggy: capture-state --from NONE — package YI4 does not open the castle path.)"
    )


def cmd_play(args: argparse.Namespace) -> None:
    """Play a level manually while recording inputs as action indices."""
    import numpy as np

    config = _resolve_config(args)
    action_table = _get_action_table(config)
    start_state = getattr(args, "state", None) or config.start_state
    _require_playable_state(config, start_state)

    from retro_harness.env import make_env, read_custom_state_bytes

    env = make_env(
        game=config.game_name,
        state=start_state,
        game_dir=config.game_dir,
        render_mode="rgb_array",
    )
    # Custom *.state: re-apply after PlaySession reset (drop free frame).
    resync_bytes = read_custom_state_bytes(
        config.game_dir, config.game_name, start_state
    )

    from retro_harness.play_session import PlaySession
    from retro_harness.platformer.progress import make_progress_tracker

    schema = config.ram_schema
    tracker = make_progress_tracker(config)
    tracker.reset()
    recorded_actions: list[int] = []
    recorded_raw: list[list[int]] = []
    recorded_raw_pre_sanitize: list[list[int]] = []
    best_progress = 0.0
    tracker_seeded = False

    # Workaround: stable-retro ignores SNES Select for SM weapon toggle.
    # Track toggle state and force via RAM on rising edge of Select.
    _select_state = [0, False]  # [current_item, was_pressed_last_frame]

    def on_step(obs, reward, done, info):
        nonlocal best_progress, tracker_seeded
        # Record the action index and raw buttons for this frame
        raw = list(last_raw_action)
        raw_pre = list(last_raw_action_pre_sanitize)
        idx = buttons_to_action_index(raw, action_table=action_table)
        recorded_actions.append(idx)
        recorded_raw.append(raw)
        recorded_raw_pre_sanitize.append(raw_pre)

        # Workaround: force weapon toggle via RAM on Select press edge
        select_pressed = bool(raw[2])  # SNES_SELECT
        if select_pressed and not _select_state[1]:
            try:
                _select_state[0] ^= 1
                env.unwrapped.data.set_value(
                    "selected_item", _select_state[0]
                )
            except Exception:
                pass
        _select_state[1] = select_pressed

        ram = env.get_ram()
        values = schema.read(ram)
        config.apply_computed(values)
        if not tracker_seeded:
            tracker.update(values)
            tracker_seeded = True
        progress = tracker.update(values)
        if progress > best_progress:
            best_progress = progress

    _ROOM_NAMES = {
        0x91F8: 'Landing Site', 0x92FD: 'Parlor', 0x9879: 'Flyway',
        0x9804: 'Bomb Torizo', 0x96BA: 'Climb', 0x975C: 'Pit Room',
        0x9AD9: 'BB Elevator', 0x9E9F: 'Morph Ball Room',
        0x9F11: 'Construction Zone', 0x9E52: 'First Missile Room',
        0xA011: 'BB E-Tank', 0x962A: 'Terminator Room',
        0x99BD: 'Green Pirates Shaft', 0x9BC8: 'Lower Mushrooms',
        0x9938: 'Green Brinstar Elev', 0x9F64: 'GB Main Shaft',
        0x9FBA: 'GB Fireflea', 0x9FC5: 'GB Missile Refill',
    }

    def on_hud(info):
        ram = env.get_ram()
        values = schema.read(ram)
        config.apply_computed(values)
        level_id = values.get("level_id", 0)
        health = values.get("health")
        lives = values.get("lives", "?")
        stat = f"hp={health}" if health is not None else f"lives={lives}"
        lines = [
            f"REC {len(recorded_actions)} frames | progress={best_progress:.0f}",
            f"{stat} | {_ROOM_NAMES.get(level_id, f'0x{level_id:04X}')}",
        ]
        # Show weapon status if SM (has selected_item in data.json)
        try:
            sel = env.unwrapped.data.lookup_value("selected_item")
            missiles = env.unwrapped.data.lookup_value("missiles")
            weapon = "MISSILES" if sel else "BEAM"
            lines.append(f"weapon={weapon} | missiles={missiles}")
        except Exception:
            pass
        return lines

    # Intercept raw actions before they go to the env
    last_raw_action = [0] * 12
    last_raw_action_pre_sanitize = [0] * 12
    _orig_gather = PlaySession._gather_action

    def patched_gather(self, pg, keyboard_action, controller_action, sanitize_action):
        nonlocal last_raw_action, last_raw_action_pre_sanitize
        action = _orig_gather(self, pg, keyboard_action, controller_action, sanitize_action)
        post = getattr(self, "_last_action_post_sanitize", action)
        pre = getattr(self, "_last_action_pre_sanitize", post)
        last_raw_action = list(post) if hasattr(post, "__iter__") else [0] * 12
        last_raw_action_pre_sanitize = list(pre) if hasattr(pre, "__iter__") else [0] * 12
        return action

    PlaySession._gather_action = patched_gather

    # Checkpoint recording state: map slot -> frame count at save time
    _checkpoint_frames: dict[int, int] = {}

    def on_key_down(key):
        import pygame as pg
        nonlocal best_progress, tracker_seeded

        _SLOT_KEYS = {pg.K_F1: 1, pg.K_F2: 2, pg.K_F3: 3, pg.K_F4: 4}

        if key in _SLOT_KEYS:
            slot = _SLOT_KEYS[key]
            mods = pg.key.get_mods()
            if mods & pg.KMOD_SHIFT:
                # Load checkpoint → truncate recording to that frame
                frame = session.load_checkpoint(slot)
                if frame is not None and slot in _checkpoint_frames:
                    rec_frame = _checkpoint_frames[slot]
                    del recorded_actions[rec_frame:]
                    del recorded_raw[rec_frame:]
                    del recorded_raw_pre_sanitize[rec_frame:]
                    # Reset progress tracker from truncated recording
                    tracker.reset()
                    tracker_seeded = False
                    best_progress = 0.0
                    print(f"  Recording truncated to {rec_frame} frames")
            else:
                # Save checkpoint → record frame position
                _checkpoint_frames[slot] = len(recorded_actions)
                session.save_checkpoint(slot)
            return True

        if key == pg.K_r:
            # Restart: reset env and clear recording
            recorded_actions.clear()
            recorded_raw.clear()
            recorded_raw_pre_sanitize.clear()
            tracker.reset()
            tracker_seeded = False
            best_progress = 0.0
            print("[RESTART] recording cleared")
            return False  # let PlaySession handle the env.reset()

        return False

    session = PlaySession(
        env,
        game_dir=str(config.game_dir),
        game=config.game_name,
        scale=args.scale,
        title=f"RECORD: {config.display_name}",
        resync_state_bytes=resync_bytes,
    )
    session.on_step = on_step
    session.on_hud = on_hud
    session.on_key_down = on_key_down

    # Recording-aware trigger hooks (L2=load, R2=save checkpoint 1)
    def trigger_save(slot: int) -> None:
        _checkpoint_frames[slot] = len(recorded_actions)
        session.save_checkpoint(slot)

    def trigger_load(slot: int) -> None:
        nonlocal best_progress, tracker_seeded
        frame = session.load_checkpoint(slot)
        if frame is not None and slot in _checkpoint_frames:
            rec_frame = _checkpoint_frames[slot]
            del recorded_actions[rec_frame:]
            del recorded_raw[rec_frame:]
            del recorded_raw_pre_sanitize[rec_frame:]
            tracker.reset()
            tracker_seeded = False
            best_progress = 0.0
            print(f"  Recording truncated to {rec_frame} frames")

    session.on_trigger_save = trigger_save
    session.on_trigger_load = trigger_load

    print(f"Recording: {config.display_name}")
    print(f"State: {start_state}")
    print(f"Action table: {len(action_table)} actions")
    print(f"\nControls:")
    print(f"  Arrow keys = D-pad")
    print(f"  Z = B    X = A    A = Y    S = X")
    print(f"  F1-F4 = save checkpoint    Shift+F1-F4 = load checkpoint")
    print(f"  F5 = export state to disk  R = restart & clear recording")
    print(f"  TAB = turbo    ESC = stop & save")
    print(f"  Controller: L2 = load checkpoint 1    R2 = save checkpoint 1\n")

    try:
        session.run()
    finally:
        # Restore original method
        PlaySession._gather_action = _orig_gather

    if not recorded_actions:
        print("No frames recorded.")
        return

    # Save recording
    output_dir = config.runs_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-increment filename
    existing = sorted(output_dir.glob("recording_*.json"))
    next_idx = 0
    for p in existing:
        try:
            n = int(p.stem.split("_")[1])
            next_idx = max(next_idx, n + 1)
        except (IndexError, ValueError):
            pass

    output_path = output_dir / f"recording_{next_idx:03d}.json"
    metadata = {
        "level": config.level_id,
        "source": "manual_play",
        "best_progress": best_progress,
        "total_frames": len(recorded_actions),
        "button_order": ["B", "Y", "Select", "Start", "Up", "Down", "Left", "Right", "A", "X", "L", "R"],
        "raw_buttons_note": "raw_buttons are post-sanitize env inputs (used for replay).",
        "raw_buttons_pre_sanitize_note": "raw_buttons_pre_sanitize are captured before directional conflict sanitization.",
    }
    save_actions(recorded_actions, output_path, metadata=metadata)

    # Also save raw 12-button arrays for faithful replay
    raw_path = output_path.with_name(output_path.stem + "_raw.json")
    import json as _json
    with open(raw_path, "w") as _f:
        _json.dump({
            "raw_buttons": recorded_raw,
            "raw_buttons_pre_sanitize": recorded_raw_pre_sanitize,
            "actions": recorded_actions,
            "metadata": metadata,
        }, _f)
    print(f"Raw buttons (post + pre-sanitize): {raw_path}")

    print(f"\nRecorded {len(recorded_actions)} frames")
    print(f"Best progress: {best_progress:.0f}")
    print(f"Saved to: {output_path}")
    print(f"\nNext steps:")
    print(f"  Verify:    uv run python -m retro_harness.platformer -l {config.level_id} verify --actions {output_path}")
    print(f"  Hillclimb: uv run python -m retro_harness.platformer -l {config.level_id} hillclimb --seed {output_path}")
    print(f"  (uses raw_buttons automatically; pass --force-index only for table-index GA seeds)")


