"""Command implementations for Spring D1 town recon CLI."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from harvest.paths import GAME_DIR, TASKS_DIR, ensure_monorepo_on_path

ensure_monorepo_on_path()

from retro_harness import TaskStatus
from retro_harness import (
    controller_action,
    describe_input_mapping,
    format_input_mapping,
    init_controller as _init_controller,
    keyboard_action,
    sanitize_action,
    SNES_BUTTON_NAMES,
)

from harvest.core.scene import classify_scene_from_ram
from harvest.runtime.recording_trace import recording_trace_entry, summarize_recording
from harvest.runtime.retro_setup import make_harvest_env
from harvest.tasks.nav import make_action

from harvest.scripts.town_day1_recon_lib import (
    D1_TOWN_BITS,
    DEFAULT_ENTRY_STATE,
    DEFAULT_RECORD_NAME,
    GATE_PIXEL,
    TARGET_MASK,
    TOWN_TILEMAP,
    TRUCK_PIXEL,
    VERIFIED_ROUTES,
    STILL_TO_RECORD,
    configure_headless,
    decode_mask_bits,
    end_state_paths,
    is_town_gate_entry,
    read_town_snapshot,
    run_power_on,
    save_state,
    unset_headless_for_interactive,
    world,
)


def cmd_checklist(_args: argparse.Namespace) -> int:
    print("=" * 64)
    print(" Spring D1 town handoff recon checklist")
    print(" Natural entry: power-on → town 0x04 @(712,424) 07:00")
    print(f" Completion mask: 0x{TARGET_MASK:02X} at d1_town_event_mask (WRAM 0x11F74)")
    print("=" * 64)
    print("\nConversations:")
    for bit, (person, note) in D1_TOWN_BITS.items():
        print(f"  0x{bit:02X}  {person:<18}  {note}")
    print("\nVerified routes:")
    for line in VERIFIED_ROUTES:
        print(f"  - {line}")
    print("\nStill to record:")
    for line in STILL_TO_RECORD:
        print(f"  - {line}")
    print("\nSuggested order from the gate:")
    print("  1. Flower shop owner (0x08) + Nina back room (0x04)")
    print("  2. Church Maria (0x20)")
    print("  3. Lower road Ann (0x01) + Eve (0x02)")
    print("  4. Animal shop livestock dealer (0x10)")
    print("  5. Truck ~(728,424) leave response once mask==0x3F")
    print("  6. Path → farm → house → sleep → assert D2")
    print("\nCommands:")
    print("  uv run python -m harvest.scripts.town_day1_recon capture-entry")
    print("  uv run python -m harvest.scripts.town_day1_recon record")
    print("  uv run python -m harvest.scripts.town_day1_recon record --power-on")
    print(
        "  HEADLESS=1 uv run python -m harvest.scripts.town_day1_recon replay "
        f"--task {DEFAULT_RECORD_NAME}"
    )
    print("=" * 64)
    return 0


def cmd_capture_entry(args: argparse.Namespace) -> int:
    configure_headless()
    env = make_harvest_env(None, render_mode="rgb_array")
    t0 = time.monotonic()
    try:
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        power_on, frames = run_power_on(env)
        if "failure" in power_on:
            report = {
                "success": False,
                "power_on": power_on,
                "frames": frames,
                "wall_seconds": round(time.monotonic() - t0, 2),
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(report, indent=2))
            return 2

        snap = read_town_snapshot(env.get_ram(), frame=frames)
        gate_ok = is_town_gate_entry(snap)
        state_path = save_state(env, args.name)
        # Also keep a tasks-local copy for discovery without hunting integrations.
        tasks_copy = TASKS_DIR / f"{args.name}.state"
        TASKS_DIR.mkdir(parents=True, exist_ok=True)
        with gzip.open(tasks_copy, "wb") as handle:
            handle.write(env.em.get_state())

        report = {
            "success": bool(gate_ok),
            "entry_state": args.name,
            "state_path": str(state_path),
            "tasks_copy": str(tasks_copy),
            "power_on": power_on,
            "snapshot": snap.as_dict(),
            "gate_ok": gate_ok,
            "expected_gate": {"tilemap": TOWN_TILEMAP, "pixel": list(GATE_PIXEL), "hour_min": 7},
            "frames": frames,
            "wall_seconds": round(time.monotonic() - t0, 2),
            "clean_run": {
                "initial_state_loads": 0,
                "mid_run_state_loads": 0,
                "ram_writes": 0,
            },
            "next": [
                f"uv run python -m harvest.scripts.town_day1_recon record --state {args.name}",
                f"HEADLESS=1 uv run python -m harvest.scripts.town_day1_recon replay "
                f"--task {DEFAULT_RECORD_NAME} --state {args.name}",
            ],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            f"[RECON] entry state={args.name} gate_ok={gate_ok} "
            f"mask={snap.mask_hex} pos=({snap.x},{snap.y}) report={args.out}",
            flush=True,
        )
        return 0 if gate_ok else 1
    finally:
        env.close()


def _print_record_banner(name: str, start_label: str, joystick) -> None:
    print("\n" + "=" * 64)
    print(f" RECORD D1 town recon: {name}")
    print(f" start: {start_label}")
    print("=" * 64)
    print(" Goal: set mask 0x3F (six talks), truck leave, farm sleep → D2")
    print(" Still open priority: flower-shop owner counter bit 0x08")
    print("")
    print(" Live HUD shows mask bits as they set.")
    print(" Controls:")
    if joystick:
        print(f"  Controller: {joystick.get_name()}")
        print(f"    Mapping: {format_input_mapping(describe_input_mapping(joystick=joystick))}")
        print("    D-Pad/Stick: move | A: talk | B: run/cancel")
    print("  Keyboard: arrows move | C=A | Z=B | X=Y | V=menu")
    print("  [ ] speed | TAB hold = FF | F5 save | ESC cancel")
    print(" Checklist bits:")
    for bit, (person, note) in D1_TOWN_BITS.items():
        print(f"    0x{bit:02X} {person}: {note}")
    print(f" Truck ~ {TRUCK_PIXEL}; gate entry was {GATE_PIXEL}")
    print("=" * 64 + "\n")


def cmd_record(args: argparse.Namespace) -> int:
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    unset_headless_for_interactive()

    import pygame

    name = args.name
    start_state: str | None
    power_on_report: dict[str, object] | None = None
    boot_frames = 0

    if args.power_on:
        start_state = None
        start_label = "power-on (clean natural entry)"
    else:
        start_state = args.state
        state_path = GAME_DIR / f"{start_state}.state"
        if not state_path.is_file():
            print(f"[RECON] missing entry state: {state_path}", file=sys.stderr)
            print(
                "[RECON] run: uv run python -m harvest.scripts.town_day1_recon capture-entry",
                file=sys.stderr,
            )
            print("[RECON] or pass --power-on for a clean boot + record session", file=sys.stderr)
            return 2
        start_label = f"state={start_state}"

    try:
        env = make_harvest_env(start_state, render_mode="rgb_array")
    except Exception as exc:
        print(f"[RECON] env failed: {exc}", file=sys.stderr)
        return 2

    pygame.init()
    try:
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        if args.power_on:
            power_on_report, boot_frames = run_power_on(env)
            if power_on_report and "failure" in power_on_report:
                print(f"[RECON] power-on failed: {power_on_report['failure']}", file=sys.stderr)
                return 2
            # Optionally pin the entry state so later replays can skip boot.
            if args.save_entry:
                entry_path = save_state(env, args.save_entry)
                print(f"[RECON] also saved entry state: {entry_path}", flush=True)
            obs = env.step(make_action())[0]

        h, w = int(obs.shape[0]), int(obs.shape[1])
        scale = max(1, int(args.scale))
        screen = pygame.display.set_mode((w * scale, h * scale))
        pygame.display.set_caption(f"D1 town recon [{name}] F5=save ESC=cancel")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("monospace", 13)
        joystick = _init_controller(pygame)
        _print_record_banner(name, start_label, joystick)

        frames: list[list[int]] = []
        trace: list[dict] = []
        running = True
        save = False
        last_mask = -1
        speed_levels = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
        speed_idx = 2
        speed = speed_levels[speed_idx]
        start_snap = read_town_snapshot(env.get_ram(), frame=0)
        print(
            f"[RECON] start mask={start_snap.mask_hex} "
            f"pos=({start_snap.x},{start_snap.y}) "
            f"tm=0x{start_snap.tilemap:02X} "
            f"S{start_snap.season}D{start_snap.day} "
            f"{start_snap.hour:02d}:{start_snap.minute:02d}",
            flush=True,
        )

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        print("[RECON] cancelled (no save)", flush=True)
                        running = False
                    elif event.key in {pygame.K_F5, pygame.K_F1}:
                        save = True
                        running = False
                    elif event.key == pygame.K_LEFTBRACKET:
                        speed_idx = max(0, speed_idx - 1)
                        speed = speed_levels[speed_idx]
                        print(f"[RECON] speed {speed}x", flush=True)
                    elif event.key == pygame.K_RIGHTBRACKET:
                        speed_idx = min(len(speed_levels) - 1, speed_idx + 1)
                        speed = speed_levels[speed_idx]
                        print(f"[RECON] speed {speed}x", flush=True)

            keys = pygame.key.get_pressed()
            action = np.zeros(12, dtype=np.int32)
            keyboard_action(keys, action, pygame)
            controller_action(joystick, action)
            sanitize_action(action)
            fast_forward = bool(keys[pygame.K_TAB])

            frames.append(action.tolist())
            obs, _reward, _term, _trunc, _info = env.step(action)
            ram = env.get_ram()
            snap = read_town_snapshot(ram, frame=len(frames) - 1)
            row = recording_trace_entry(
                ram,
                frame=len(frames) - 1,
                action=action,
            )
            row["d1_town_event_mask"] = snap.mask
            row["d1_bits_set"] = snap.bits_set
            trace.append(row)

            if snap.mask != last_mask:
                gained = decode_mask_bits(snap.mask & ~max(last_mask, 0)) if last_mask >= 0 else snap.bits_set
                if last_mask >= 0 and gained:
                    print(
                        f"[RECON] mask {last_mask:#04x} → {snap.mask_hex} "
                        f"+{gained} missing={snap.bits_missing}",
                        flush=True,
                    )
                last_mask = snap.mask

            if not fast_forward:
                surf = pygame.surfarray.make_surface(np.asarray(obs).swapaxes(0, 1))
                scaled = pygame.transform.scale(surf, (w * scale, h * scale))
                screen.blit(scaled, (0, 0))
                missing = ",".join(snap.bits_missing) if snap.bits_missing else "none"
                lines = [
                    f"[REC] f={len(frames)} {speed}x mask={snap.mask_hex}/0x3F "
                    f"{'OK' if snap.mask == TARGET_MASK else '…'}",
                    f"tm=0x{snap.tilemap:02X} ({snap.x},{snap.y}) "
                    f"S{snap.season}D{snap.day} {snap.hour:02d}:{snap.minute:02d}",
                    f"set: {','.join(snap.bits_set) or '—'}  miss: {missing}",
                    "F5=save  ESC=cancel  [ ]=speed  TAB=FF",
                ]
                y = 4
                for line in lines:
                    text = font.render(line, True, (255, 40, 40) if snap.mask != TARGET_MASK else (40, 220, 80))
                    screen.blit(text, (6, y))
                    y += 16
                pressed = [SNES_BUTTON_NAMES[i] for i in range(12) if action[i]]
                if pressed:
                    btn = font.render(" ".join(pressed), True, (255, 255, 0))
                    screen.blit(btn, (6, h * scale - 22))
                pygame.display.flip()
                clock.tick(int(60 * speed))
            elif len(frames) % 60 == 0:
                pygame.display.set_caption(
                    f"D1 town recon [{name}] FF f={len(frames)} mask={snap.mask_hex}"
                )
                pygame.display.flip()

        if not save or not frames:
            return 1 if not save else 0

        end_snap = read_town_snapshot(env.get_ram(), frame=len(frames))
        scene = classify_scene_from_ram(env.get_ram())
        metadata = summarize_recording(frames=frames, trace=trace)
        metadata.update(
            {
                "recon": "town_day1",
                "target_mask": TARGET_MASK,
                "start_mask": start_snap.mask,
                "end_mask": end_snap.mask,
                "start_snapshot": start_snap.as_dict(),
                "end_snapshot": end_snap.as_dict(),
                "end_scene": scene.to_dict(),
                "boot_frames": boot_frames,
                "power_on": power_on_report,
                "natural_entry": "power_on" if args.power_on else f"state:{start_state}",
                "truck_hint_pixel": list(TRUCK_PIXEL),
                "bit_labels": {
                    f"0x{bit:02X}": {"person": person, "note": note}
                    for bit, (person, note) in D1_TOWN_BITS.items()
                },
            }
        )
        task_data = {
            "name": name,
            "frames": frames,
            "trace": trace,
            "start_state": start_state,
            "metadata": metadata,
            "recorded_at": datetime.now().isoformat(),
            "frame_count": len(frames),
        }
        TASKS_DIR.mkdir(parents=True, exist_ok=True)
        task_json = TASKS_DIR / f"{name}.json"
        task_json.write_text(json.dumps(task_data, indent=2) + "\n", encoding="utf-8")
        end_bytes = env.em.get_state()
        for path in end_state_paths(task_json):
            path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(path, "wb") as handle:
                handle.write(end_bytes)

        print(f"[RECON] saved {task_json} ({len(frames)} frames)", flush=True)
        print(
            f"[RECON] end mask={end_snap.mask_hex} complete={end_snap.mask == TARGET_MASK} "
            f"day={end_snap.day} pos=({end_snap.x},{end_snap.y}) tm=0x{end_snap.tilemap:02X}",
            flush=True,
        )
        print(
            f"[RECON] next: HEADLESS=1 uv run python -m harvest.scripts.town_day1_recon "
            f"replay --task {name}"
            + (f" --state {start_state}" if start_state else " --power-on"),
            flush=True,
        )
        return 0
    finally:
        env.close()
        pygame.quit()


def _load_task(task: str) -> tuple[Path, dict]:
    path = Path(task)
    if path.suffix != ".json":
        path = TASKS_DIR / f"{task}.json"
    if not path.is_file():
        raise FileNotFoundError(f"task not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return path, json.load(handle)


def cmd_replay(args: argparse.Namespace) -> int:
    configure_headless()
    try:
        task_path, data = _load_task(args.task)
    except FileNotFoundError as exc:
        print(f"[RECON] {exc}", file=sys.stderr)
        return 2

    frames: list = data.get("frames") or []
    if not frames:
        print(f"[RECON] empty frames in {task_path}", file=sys.stderr)
        return 2

    meta = data.get("metadata") or {}
    use_power_on = bool(args.power_on)
    start_state = args.state or data.get("start_state")
    if use_power_on:
        start_state = None
    elif not start_state:
        start_state = DEFAULT_ENTRY_STATE

    if not use_power_on:
        state_path = GAME_DIR / f"{start_state}.state"
        if not state_path.is_file():
            print(f"[RECON] missing start state: {state_path}", file=sys.stderr)
            print("[RECON] capture-entry first, or pass --power-on", file=sys.stderr)
            return 2

    env = make_harvest_env(start_state, render_mode="rgb_array")
    t0 = time.monotonic()
    mask_events: list[dict[str, object]] = []
    tilemap_events: list[dict[str, object]] = []
    try:
        env.reset()
        power_on_report = None
        boot_frames = 0
        if use_power_on:
            power_on_report, boot_frames = run_power_on(env)
            if power_on_report and "failure" in power_on_report:
                report = {
                    "success": False,
                    "reason": power_on_report.get("failure"),
                    "power_on": power_on_report,
                    "task": str(task_path),
                }
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                return 2

        start_snap = read_town_snapshot(env.get_ram(), frame=0)
        prev_mask = start_snap.mask
        prev_tm = start_snap.tilemap

        for idx, action_list in enumerate(frames):
            action = np.asarray(action_list, dtype=np.int32)
            env.step(action)
            snap = read_town_snapshot(env.get_ram(), frame=idx)
            if snap.mask != prev_mask:
                mask_events.append(
                    {
                        "frame": idx,
                        "from": prev_mask,
                        "to": snap.mask,
                        "from_hex": f"0x{prev_mask:02X}",
                        "to_hex": snap.mask_hex,
                        "gained": decode_mask_bits(snap.mask & ~prev_mask),
                        "pos": [snap.x, snap.y],
                        "tilemap": snap.tilemap,
                    }
                )
                prev_mask = snap.mask
            if snap.tilemap != prev_tm:
                tilemap_events.append(
                    {
                        "frame": idx,
                        "from": prev_tm,
                        "to": snap.tilemap,
                        "from_hex": f"0x{prev_tm:02X}",
                        "to_hex": f"0x{snap.tilemap:02X}",
                        "pos": [snap.x, snap.y],
                        "mask": snap.mask,
                    }
                )
                prev_tm = snap.tilemap

        end_snap = read_town_snapshot(env.get_ram(), frame=len(frames))
        scene = classify_scene_from_ram(env.get_ram())
        mask_ok = end_snap.mask == TARGET_MASK
        day2_ok = end_snap.day >= 2 and end_snap.season == 0
        if args.require_day2:
            success = mask_ok and day2_ok
            reason = "ok" if success else (
                "mask incomplete" if not mask_ok else "did not reach day 2"
            )
        else:
            success = mask_ok if args.require_mask else True
            reason = "ok" if success else f"mask {end_snap.mask_hex} != 0x{TARGET_MASK:02X}"

        report = {
            "success": success,
            "reason": reason,
            "task": str(task_path),
            "start_state": start_state,
            "power_on": power_on_report,
            "boot_frames": boot_frames,
            "frame_count": len(frames),
            "wall_seconds": round(time.monotonic() - t0, 2),
            "start": start_snap.as_dict(),
            "end": end_snap.as_dict(),
            "end_scene": scene.to_dict(),
            "mask_events": mask_events,
            "tilemap_events": tilemap_events,
            "assertions": {
                "target_mask": TARGET_MASK,
                "mask_ok": mask_ok,
                "day2_ok": day2_ok,
                "require_mask": bool(args.require_mask),
                "require_day2": bool(args.require_day2),
            },
            "recording_metadata": {
                k: meta.get(k)
                for k in (
                    "recon",
                    "natural_entry",
                    "start_mask",
                    "end_mask",
                    "frame_count",
                    "duration_seconds",
                )
                if k in meta
            },
            "clean_run": {
                "initial_state_loads": 0 if use_power_on else 1,
                "mid_run_state_loads": 0,
                "ram_writes": 0,
            },
            "next_automation": [
                "Extract stand tiles / facing from mask_events and trace",
                "Promote flower-owner counter as recorded segment skill",
                "Wire truck leave + corrected town_to_farm into day1 phases",
            ],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            f"[RECON] replay success={success} mask={end_snap.mask_hex} "
            f"day={end_snap.day} events={len(mask_events)} report={args.out}",
            flush=True,
        )
        return 0 if success else 1
    finally:
        env.close()


def cmd_status(args: argparse.Namespace) -> int:
    """Load a state (or power-on) and print the current mask/position."""
    configure_headless()
    use_power_on = bool(args.power_on)
    state = None if use_power_on else args.state
    if not use_power_on:
        path = GAME_DIR / f"{state}.state"
        if not path.is_file():
            print(f"[RECON] missing state: {path}", file=sys.stderr)
            return 2
    env = make_harvest_env(state, render_mode="rgb_array")
    try:
        env.reset()
        power_on = None
        if use_power_on:
            power_on, _ = run_power_on(env)
            if power_on and "failure" in power_on:
                print(json.dumps({"success": False, "power_on": power_on}, indent=2))
                return 2
        snap = read_town_snapshot(env.get_ram())
        scene = classify_scene_from_ram(env.get_ram())
        report = {
            "success": True,
            "state": state,
            "power_on": power_on,
            "snapshot": snap.as_dict(),
            "scene": scene.to_dict(),
            "gate_ok": is_town_gate_entry(snap),
        }
        print(json.dumps(report, indent=2))
        return 0
    finally:
        env.close()


def cmd_auto(args: argparse.Namespace) -> int:
    """Run the precomputed D1 town handoff (talks → truck → farm → sleep)."""
    from harvest.tasks.town_day1_handoff import TARGET_MASK, TownDay1HandoffTask, read_mask

    configure_headless()
    use_power_on = bool(args.power_on)
    state = None if use_power_on else args.state
    if not use_power_on:
        path = GAME_DIR / f"{state}.state"
        if not path.is_file():
            print(f"[RECON] missing state: {path}", file=sys.stderr)
            print("[RECON] run capture-entry first, or pass --power-on", file=sys.stderr)
            return 2

    env = make_harvest_env(state, render_mode="rgb_array")
    t0 = time.monotonic()
    mask_events: list[dict[str, object]] = []
    try:
        env.reset()
        power_on = None
        boot_frames = 0
        if use_power_on:
            power_on, boot_frames = run_power_on(env)
            if power_on and "failure" in power_on:
                report = {"success": False, "reason": power_on.get("failure"), "power_on": power_on}
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                return 2
            if args.save_entry:
                entry_path = save_state(env, args.save_entry)
                print(f"[RECON] saved entry state {entry_path}", flush=True)

        start_snap = read_town_snapshot(env.get_ram(), frame=0)
        prev_mask = start_snap.mask
        peak_mask = start_snap.mask
        task = TownDay1HandoffTask(
            include_sleep=bool(args.sleep),
            require_full_mask=bool(args.require_full_mask),
            timeout=int(args.timeout),
        )
        task.reset(world(env, 0))
        frames = 0
        last_log = 0
        final_status = "running"
        final_reason = ""
        while frames < task.timeout:
            w = world(env, frames)
            result = task.step(w)
            snap = read_town_snapshot(w.ram, frame=frames)
            peak_mask = max(peak_mask, snap.mask)
            if snap.mask != prev_mask:
                gained = decode_mask_bits(snap.mask & ~prev_mask)
                mask_events.append(
                    {
                        "frame": frames,
                        "from": prev_mask,
                        "to": snap.mask,
                        "gained": gained,
                        "pos": [snap.x, snap.y],
                        "tilemap": snap.tilemap,
                        "phase": task.summary(w).get("phase"),
                    }
                )
                print(
                    f"[RECON] mask 0x{prev_mask:02X}→{snap.mask_hex} +{gained} "
                    f"tm=0x{snap.tilemap:02X} ({snap.x},{snap.y}) f={frames}",
                    flush=True,
                )
                prev_mask = snap.mask

            if frames - last_log >= 600 or result.status != TaskStatus.RUNNING:
                summary = task.summary(w)
                print(
                    f"[RECON] f={frames} phase={summary.get('phase')} "
                    f"mask={summary.get('mask_hex')} "
                    f"tm=0x{summary.get('tilemap'):02X} "
                    f"day={summary.get('day')} "
                    f"pos=({summary.get('x')},{summary.get('y')}) "
                    f"status={result.status.value} {result.reason or ''}",
                    flush=True,
                )
                last_log = frames

            if result.status == TaskStatus.SUCCESS:
                final_status = "success"
                final_reason = result.reason or "ok"
                break
            if result.status in (TaskStatus.FAILURE, TaskStatus.BLOCKED):
                final_status = result.status.value
                final_reason = result.reason or result.status.value
                break
            action = result.action.action if result.action is not None else make_action()
            env.step(action)
            frames += 1
        else:
            final_status = "timeout"
            final_reason = "frame budget exhausted"

        end_snap = read_town_snapshot(env.get_ram(), frame=frames)
        scene = classify_scene_from_ram(env.get_ram())
        summary = task.summary(world(env, frames))
        mask = int(read_mask(env.get_ram()))
        target = TARGET_MASK if args.require_full_mask else 0x03  # Ann|Eve baseline
        # Mask clears on D2 after truck handoff sleep; accept peak mask too.
        mask_ok = (mask & target) == target or (peak_mask & target) == target
        day2_ok = end_snap.day >= 2 and end_snap.season == 0
        # Rest recording already sleeps; --sleep may be true but include_sleep
        # was turned off inside the task when the rest file is present.
        need_day2 = bool(args.require_full_mask)
        # Gate B: house_size=0 (power-on) must end with grass+can in carry.
        require_shed = bool(summary.get("require_starter_tools"))
        shed_ok = (
            (not require_shed)
            or (
                bool(summary.get("has_watering_can"))
                and bool(summary.get("has_grass_seeds"))
            )
        )
        success = (
            final_status == "success"
            and mask_ok
            and (day2_ok if need_day2 else True)
            and shed_ok
        )

        if args.save_end_state and success:
            end_path = save_state(env, args.save_end_state)
            print(f"[RECON] saved end state {end_path}", flush=True)

        report = {
            "success": success,
            "final_status": final_status,
            "reason": final_reason,
            "state": state,
            "power_on": power_on,
            "boot_frames": boot_frames,
            "frames": frames,
            "wall_seconds": round(time.monotonic() - t0, 2),
            "start": start_snap.as_dict(),
            "end": end_snap.as_dict(),
            "end_scene": scene.to_dict(),
            "summary": summary,
            "mask_events": mask_events,
            "assertions": {
                "target_mask": TARGET_MASK,
                "mask_ok": mask_ok,
                "peak_mask": peak_mask,
                "peak_mask_hex": f"0x{peak_mask:02X}",
                "end_mask": mask,
                "day2_ok": day2_ok,
                "include_sleep": bool(args.sleep),
                "require_starter_tools": require_shed,
                "shed_ok": shed_ok,
                "house_size_at_start": summary.get("house_size_at_start"),
                "has_watering_can": summary.get("has_watering_can"),
                "has_grass_seeds": summary.get("has_grass_seeds"),
            },
            "clean_run": {
                "initial_state_loads": 0 if use_power_on else 1,
                "mid_run_state_loads": 0,
                "ram_writes": 0,
            },
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            f"[RECON] auto success={success} status={final_status} "
            f"mask=0x{mask:02X} day={end_snap.day} frames={frames} "
            f"report={args.out}",
            flush=True,
        )
        return 0 if success else 1
    finally:
        env.close()
