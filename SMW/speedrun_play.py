#!/usr/bin/env python3
"""Fresh-game Super Mario World speedrun practice recorder.

This is the human entrypoint for booting SMW from the ROM/title screen and
recording a full-session input trace. It intentionally keeps route logic light:
the early SMW route states and split detectors are still being built, so this
captures raw input, RAM traces, save/load events, and branch recordings that can
later seed segment extraction and optimization.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from platformer_common.actions import buttons_to_action_index
from platformer_common.levels.super_mario_world import SMW_SPEED_ACTIONS
from retro_harness.recordings import append_jsonl

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
GAME = "SuperMarioWorld-Snes-v0"
STATE_DIR = ROOT / "custom_integrations" / GAME
GAME_ROM = STATE_DIR / "rom.sfc"
DEFAULT_RECORDINGS_DIR = ROOT / "recordings" / "speedrun"

BUTTON_ORDER = [
    "B",
    "Y",
    "Select",
    "Start",
    "Up",
    "Down",
    "Left",
    "Right",
    "A",
    "X",
    "L",
    "R",
]

ADDR = {
    "true_frame": 0x0013,
    "effective_frame": 0x0014,
    "powerup": 0x0019,
    "camera_x": 0x001A,
    "camera_y": 0x001C,
    "player_animation": 0x0071,
    "player_in_air": 0x0072,
    "player_direction": 0x0076,
    "player_blocked_dir": 0x0077,
    "player_x_speed": 0x007A,
    "player_y_speed": 0x007C,
    "player_x": 0x00D1,
    "player_y": 0x00D3,
    "game_mode": 0x0100,
    "lives": 0x0DBE,
    "coins": 0x0DBF,
    "item_box": 0x0DC2,
    "level_timer_frames": 0x0F30,
    "level_timer_hundreds": 0x0F31,
    "level_timer_tens": 0x0F32,
    "level_timer_ones": 0x0F33,
    "translevel": 0x13BF,
    "current_submap": 0x13C3,
    "midway_flag": 0x13CE,
    "p_meter": 0x13E4,
    "on_ground": 0x13EF,
    "active_boss": 0x13FC,
    "camera_scrolling": 0x13FD,
}


def utc_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def normalize_buttons(buttons: Any, size: int = 12) -> list[int]:
    if not isinstance(buttons, list):
        try:
            buttons = list(buttons)
        except TypeError:
            buttons = []
    out = [int(bool(value)) for value in buttons[:size]]
    if len(out) < size:
        out.extend([0] * (size - len(out)))
    return out


def u16(ram: Any, addr: int) -> int:
    return int(ram[addr]) | (int(ram[addr + 1]) << 8)


def s8(value: int) -> int:
    value &= 0xFF
    return value - 0x100 if value & 0x80 else value


def s16(value: int) -> int:
    value &= 0xFFFF
    return value - 0x10000 if value & 0x8000 else value


def read_smw_ram_values(ram: Any) -> dict[str, int]:
    return {
        "true_frame": int(ram[ADDR["true_frame"]]),
        "effective_frame": int(ram[ADDR["effective_frame"]]),
        "game_mode": int(ram[ADDR["game_mode"]]),
        "translevel": int(ram[ADDR["translevel"]]),
        "current_submap": int(ram[ADDR["current_submap"]]),
        "camera_x": u16(ram, ADDR["camera_x"]),
        "camera_y": u16(ram, ADDR["camera_y"]),
        "player_x": u16(ram, ADDR["player_x"]),
        "player_y": u16(ram, ADDR["player_y"]),
        "player_x_speed": s16(u16(ram, ADDR["player_x_speed"])),
        "player_y_speed": s16(u16(ram, ADDR["player_y_speed"])),
        "player_animation": int(ram[ADDR["player_animation"]]),
        "player_in_air": int(ram[ADDR["player_in_air"]]),
        "player_direction": int(ram[ADDR["player_direction"]]),
        "player_blocked_dir": int(ram[ADDR["player_blocked_dir"]]),
        "lives": s8(int(ram[ADDR["lives"]])),
        "coins": int(ram[ADDR["coins"]]),
        "powerup": int(ram[ADDR["powerup"]]),
        "item_box": int(ram[ADDR["item_box"]]),
        "level_timer_frames": int(ram[ADDR["level_timer_frames"]]),
        "level_timer_hundreds": int(ram[ADDR["level_timer_hundreds"]]),
        "level_timer_tens": int(ram[ADDR["level_timer_tens"]]),
        "level_timer_ones": int(ram[ADDR["level_timer_ones"]]),
        "midway_flag": int(ram[ADDR["midway_flag"]]),
        "p_meter": int(ram[ADDR["p_meter"]]),
        "on_ground": int(ram[ADDR["on_ground"]]),
        "active_boss": int(ram[ADDR["active_boss"]]),
        "camera_scrolling": int(ram[ADDR["camera_scrolling"]]),
    }


def read_smw_ram(env: Any) -> dict[str, int]:
    return read_smw_ram_values(env.get_ram())


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_rom_sha() -> str:
    return (STATE_DIR / "rom.sha").read_text(encoding="utf-8").strip()


def rom_candidates() -> list[Path]:
    return [
        ROOT / "roms" / "smw.sfc",
        REPO_ROOT / "roms" / "smw.sfc",
        REPO_ROOT / "roms" / "Super Mario World.sfc",
        REPO_ROOT / "roms" / "Super Mario World (USA).sfc",
        REPO_ROOT / "roms" / "Super Mario World.smc",
        REPO_ROOT / "roms" / "Super Mario World (USA).smc",
        ROOT / "Super Mario World.sfc",
        ROOT / "Super Mario World.smc",
    ]


def ensure_rom_installed() -> None:
    expected = expected_rom_sha()
    if GAME_ROM.exists():
        actual = sha1(GAME_ROM)
        if actual != expected:
            raise SystemExit(
                "SMW ROM hash mismatch:\n"
                f"  path:     {GAME_ROM}\n"
                f"  expected: {expected}\n"
                f"  actual:   {actual}"
            )
        return

    for candidate in rom_candidates():
        if not candidate.exists():
            continue
        actual = sha1(candidate)
        if actual != expected:
            raise SystemExit(
                "SMW ROM hash mismatch:\n"
                f"  path:     {candidate}\n"
                f"  expected: {expected}\n"
                f"  actual:   {actual}"
            )
        GAME_ROM.parent.mkdir(parents=True, exist_ok=True)
        if GAME_ROM.is_symlink():
            GAME_ROM.unlink()
        try:
            rel = os.path.relpath(candidate.resolve(), GAME_ROM.parent.resolve())
            GAME_ROM.symlink_to(rel)
        except OSError:
            shutil.copy2(candidate, GAME_ROM)
        return

    raise SystemExit(
        "SMW ROM not found.\n"
        f"Place a legally owned USA ROM at {ROOT / 'roms' / 'smw.sfc'} "
        f"or {REPO_ROOT / 'roms' / 'smw.sfc'}."
    )


def write_state_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        f.write(data)


def read_state_bytes(path: Path) -> bytes:
    with gzip.open(path, "rb") as f:
        return f.read()


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        rel = os.path.relpath(src.resolve(), dst.parent.resolve())
        dst.symlink_to(rel)
    except OSError:
        shutil.copy2(src, dst)


@contextmanager
def fresh_game_dir(enabled: bool) -> Iterator[Path]:
    """Yield a game_dir with custom metadata but no battery-backed SRAM."""
    if not enabled:
        yield ROOT
        return

    with tempfile.TemporaryDirectory(prefix="smw_fresh_") as tmp:
        temp_root = Path(tmp)
        temp_state_dir = temp_root / "custom_integrations" / GAME
        temp_state_dir.mkdir(parents=True, exist_ok=True)
        for name in ("data.json", "metadata.json", "scenario.json", "rom.sha", "rom.sfc"):
            _link_or_copy(STATE_DIR / name, temp_state_dir / name)
        for state_path in STATE_DIR.glob("*.state"):
            _link_or_copy(state_path, temp_state_dir / state_path.name)
        yield temp_root


@dataclass
class Branch:
    branch_id: int
    reason: str
    started_at_frame: int
    state_name: str = ""
    state_file: str = ""
    start_ram: dict[str, int] = field(default_factory=dict)
    actions: list[int] = field(default_factory=list)
    raw_buttons: list[list[int]] = field(default_factory=list)
    raw_buttons_pre_sanitize: list[list[int]] = field(default_factory=list)


class SpeedrunRecorder:
    def __init__(self, session_dir: Path, *, session_id: str, trace_every: int) -> None:
        self.session_dir = session_dir
        self.session_id = session_id
        self.trace_every = max(1, trace_every)
        self.frames_path = session_dir / "frames.jsonl"
        self.events_path = session_dir / "events.jsonl"
        self.branches_dir = session_dir / "branches"
        self.states_dir = session_dir / "states"
        self.global_frame = 0
        self.current_branch: Branch | None = None
        self.next_branch_id = 1
        self.branch_summaries: list[dict[str, Any]] = []

        self.branches_dir.mkdir(parents=True, exist_ok=True)
        self.states_dir.mkdir(parents=True, exist_ok=True)

    def log_event(self, event: str, **data: Any) -> dict[str, Any]:
        entry = {
            "frame": self.global_frame,
            "event": event,
            "time": datetime.now(timezone.utc).isoformat(),
            **data,
        }
        append_jsonl(self.events_path, entry)
        return entry

    def start_branch(
        self,
        reason: str,
        *,
        state_name: str = "",
        state_file: str = "",
        ram: dict[str, int] | None = None,
    ) -> None:
        self.finish_branch(f"branch_end:{reason}")
        branch = Branch(
            branch_id=self.next_branch_id,
            reason=reason,
            started_at_frame=self.global_frame,
            state_name=state_name,
            state_file=state_file,
            start_ram=dict(ram or {}),
        )
        self.next_branch_id += 1
        self.current_branch = branch
        self.log_event(
            "branch_start",
            branch_id=branch.branch_id,
            reason=reason,
            state_name=state_name,
            state_file=state_file,
            ram=dict(ram or {}),
        )

    def record_frame(
        self,
        *,
        action_idx: int,
        raw: list[int],
        raw_pre: list[int],
        ram: dict[str, int],
    ) -> None:
        if self.current_branch is None:
            self.start_branch("implicit")
        assert self.current_branch is not None
        frame = self.global_frame
        self.current_branch.actions.append(action_idx)
        self.current_branch.raw_buttons.append(list(raw))
        self.current_branch.raw_buttons_pre_sanitize.append(list(raw_pre))
        if frame % self.trace_every == 0:
            append_jsonl(
                self.frames_path,
                {
                    "frame": frame,
                    "branch_id": self.current_branch.branch_id,
                    "branch_frame": len(self.current_branch.actions) - 1,
                    "action": action_idx,
                    "raw_buttons": raw,
                    "raw_buttons_pre_sanitize": raw_pre,
                    "ram": ram,
                },
            )
        self.global_frame += 1

    def finish_branch(self, reason: str) -> None:
        branch = self.current_branch
        if branch is None:
            return
        if not branch.actions:
            self.current_branch = None
            return

        path = self.branches_dir / f"branch_{branch.branch_id:03d}.json"
        metadata = {
            "session_id": self.session_id,
            "branch_id": branch.branch_id,
            "reason": branch.reason,
            "ended_by": reason,
            "started_at_frame": branch.started_at_frame,
            "total_frames": len(branch.actions),
            "state_name": branch.state_name,
            "state_file": branch.state_file,
            "start_ram": branch.start_ram,
            "button_order": BUTTON_ORDER,
            "raw_buttons_note": "raw_buttons are post-sanitize env inputs; use these for faithful replay.",
        }
        payload = {
            "actions": branch.actions,
            "raw_buttons": branch.raw_buttons,
            "raw_buttons_pre_sanitize": branch.raw_buttons_pre_sanitize,
            "metadata": metadata,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        summary = {
            "branch_id": branch.branch_id,
            "path": str(path),
            "frames": len(branch.actions),
            "reason": branch.reason,
            "ended_by": reason,
            "started_at_frame": branch.started_at_frame,
            "state_name": branch.state_name,
        }
        self.branch_summaries.append(summary)
        self.log_event("branch_finish", **summary)
        self.current_branch = None


class HotStateIndex:
    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir
        self.names: list[str] = []
        self.index = 0
        self.refresh()

    @property
    def selected(self) -> str | None:
        if not self.names:
            return None
        return self.names[self.index % len(self.names)]

    def refresh(self, prefer: str | None = None) -> None:
        names = sorted(path.stem for path in self.state_dir.glob("*.state"))
        self.names = names
        if not names:
            self.index = 0
            return
        if prefer in names:
            self.index = names.index(prefer)
        else:
            self.index %= len(names)

    def cycle(self, delta: int) -> str | None:
        self.refresh()
        if not self.names:
            return None
        self.index = (self.index + delta) % len(self.names)
        return self.selected


@dataclass
class CheckpointSlot:
    slot: int
    state_data: bytes
    name: str
    frame: int
    branch_id: int
    branch_frame: int
    path: Path


class SMWSpeedrunPractice:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.session_id = utc_id()
        self.session_dir = Path(args.session_dir) if args.session_dir else DEFAULT_RECORDINGS_DIR / self.session_id
        self.recorder = SpeedrunRecorder(
            self.session_dir,
            session_id=self.session_id,
            trace_every=args.trace_every,
        )
        self.hot_states = HotStateIndex(STATE_DIR)
        self.checkpoints: dict[int, CheckpointSlot] = {}
        self.env = None
        self.session = None
        self.last_ram: dict[str, int] = {}
        self.last_game_mode: int | None = None
        self.last_translevel: int | None = None
        self.last_lives: int | None = None
        self.finished = False

    def run(self) -> None:
        from retro_harness.env import make_env
        from retro_harness.play_session import PlaySession

        fresh_sram = self.args.state.upper() == "NONE" and not self.args.keep_sram
        with fresh_game_dir(fresh_sram) as env_game_dir:
            self.env = make_env(
                game=GAME,
                state=self.args.state,
                game_dir=env_game_dir,
                render_mode="rgb_array",
            )
            self.recorder.start_branch("initial", state_name=self.args.state)

            self.session = PlaySession(
                self.env,
                game_dir=str(ROOT),
                game=GAME,
                scale=self.args.scale,
                title="SMW SPEEDRUN PRACTICE",
                action_size=12,
                headless=self.args.headless,
            )
            self.session.on_step = self.on_step
            self.session.on_hud = self.on_hud
            self.session.on_key_down = self.on_key_down
            self.session.on_trigger_save = self.save_checkpoint
            self.session.on_trigger_load = self.load_checkpoint

            self.recorder.log_event(
                "session_start",
                state=self.args.state,
                fresh_sram=fresh_sram,
                session_dir=str(self.session_dir),
                trace_every=self.args.trace_every,
            )

            print("SMW fresh-game speedrun recorder")
            print(f"State: {self.args.state}")
            print(f"Fresh SRAM: {'yes' if fresh_sram else 'no'}")
            print(f"Session: {self.session_dir}")
            print("")
            print("Keyboard:")
            print("  F1-F4 save memory checkpoints, Shift+F1-F4 load")
            print("  F5 save QuickSave, F7/F8 load QuickSave")
            print("  F6 mark hard spot, F9/F10 cycle states, F11 load selected, F12 save state")
            print("  R reset to fresh start, TAB turbo, [/] speed, ESC stop and save")
            print("")
            print("Controller:")
            print("  R2 save checkpoint 1, L2 load checkpoint 1")
            print("")

            try:
                self.session.run()
            except KeyboardInterrupt:
                self.recorder.log_event("keyboard_interrupt")
                print("\n[STOP] interrupted; saving session")
            finally:
                self.finish()

    def on_step(self, obs: Any, reward: float, done: bool, info: dict[str, Any]) -> None:
        assert self.env is not None and self.session is not None
        ram = read_smw_ram(self.env)
        self.last_ram = ram
        raw = normalize_buttons(getattr(self.session, "_last_action_post_sanitize", [0] * 12))
        raw_pre = normalize_buttons(getattr(self.session, "_last_action_pre_sanitize", raw))
        action_idx = buttons_to_action_index(raw, action_table=SMW_SPEED_ACTIONS)
        self.recorder.record_frame(action_idx=action_idx, raw=raw, raw_pre=raw_pre, ram=ram)
        self._log_ram_transitions(ram)

        if self.args.max_frames and self.recorder.global_frame >= self.args.max_frames:
            self.recorder.log_event("max_frames_reached", max_frames=self.args.max_frames)
            self.session.running = False

    def _log_ram_transitions(self, ram: dict[str, int]) -> None:
        game_mode = ram.get("game_mode")
        translevel = ram.get("translevel")
        lives = ram.get("lives")
        if self.last_game_mode is None:
            self.last_game_mode = game_mode
            self.last_translevel = translevel
            self.last_lives = lives
            return

        if game_mode != self.last_game_mode:
            self.recorder.log_event(
                "game_mode_change",
                previous=self.last_game_mode,
                current=game_mode,
                translevel=translevel,
                ram=dict(ram),
            )
            self.last_game_mode = game_mode
        if translevel != self.last_translevel:
            self.recorder.log_event(
                "translevel_change",
                previous=self.last_translevel,
                current=translevel,
                game_mode=game_mode,
                ram=dict(ram),
            )
            self.last_translevel = translevel
        if self.last_lives is not None and lives is not None and lives < self.last_lives:
            self.recorder.log_event(
                "lives_drop",
                previous=self.last_lives,
                current=lives,
                translevel=translevel,
                ram=dict(ram),
            )
        self.last_lives = lives

    def on_hud(self, info: dict[str, Any]) -> list[str]:
        ram = self.last_ram
        timer = f"{ram.get('level_timer_hundreds', 0)}{ram.get('level_timer_tens', 0)}{ram.get('level_timer_ones', 0)}"
        branch = self.recorder.current_branch
        branch_id = branch.branch_id if branch else 0
        branch_frames = len(branch.actions) if branch else 0
        selected = self.hot_states.selected or "no states"
        return [
            f"REC {self.recorder.global_frame}f | branch {branch_id}:{branch_frames}f",
            f"mode=0x{ram.get('game_mode', 0):02X} trans=0x{ram.get('translevel', 0):02X} lives={ram.get('lives', '?')} timer={timer}",
            f"x={ram.get('player_x', 0)} y={ram.get('player_y', 0)} cam={ram.get('camera_x', 0)},{ram.get('camera_y', 0)}",
            f"state: {selected}",
        ]

    def on_key_down(self, key: int) -> bool:
        import pygame as pg

        slot_keys = {pg.K_F1: 1, pg.K_F2: 2, pg.K_F3: 3, pg.K_F4: 4}
        if key in slot_keys:
            slot = slot_keys[key]
            if pg.key.get_mods() & pg.KMOD_SHIFT:
                self.load_checkpoint(slot)
            else:
                self.save_checkpoint(slot)
            return True
        if key == pg.K_F5:
            self.save_hot_state(prefix="QuickSave", fixed_name="QuickSave")
            return True
        if key in (pg.K_F7, pg.K_F8):
            self.load_named_state("QuickSave")
            return True
        if key == pg.K_F6:
            self.mark_hard_spot(source="keyboard")
            return True
        if key == pg.K_F9:
            self.cycle_hot_state(-1)
            return True
        if key == pg.K_F10:
            self.cycle_hot_state(1)
            return True
        if key == pg.K_F11:
            self.load_selected_state()
            return True
        if key == pg.K_F12:
            self.save_hot_state(prefix="Speedrun")
            return True
        if key == pg.K_r:
            self.reset_to_start()
            return True
        return False

    def checkpoint_branch_frame(self) -> int:
        branch = self.recorder.current_branch
        return len(branch.actions) if branch else 0

    def save_checkpoint(self, slot: int) -> None:
        assert self.env is not None
        ram = read_smw_ram(self.env)
        state_data = self.env.em.get_state()
        name = f"SpeedrunSlot{slot}"
        custom_path = STATE_DIR / f"{name}.state"
        archive_path = self.recorder.states_dir / f"{name}_f{self.recorder.global_frame:06d}.state"
        write_state_bytes(custom_path, state_data)
        write_state_bytes(archive_path, state_data)
        branch = self.recorder.current_branch
        self.checkpoints[slot] = CheckpointSlot(
            slot=slot,
            state_data=state_data,
            name=name,
            frame=self.recorder.global_frame,
            branch_id=branch.branch_id if branch else 0,
            branch_frame=self.checkpoint_branch_frame(),
            path=archive_path,
        )
        self.hot_states.refresh(prefer=name)
        self.recorder.log_event(
            "checkpoint_save",
            slot=slot,
            state_name=name,
            state_file=str(archive_path),
            branch_frame=self.checkpoint_branch_frame(),
            ram=dict(ram),
        )
        print(f"[CHECKPOINT {slot}] saved -> {name}")

    def load_checkpoint(self, slot: int) -> None:
        checkpoint = self.checkpoints.get(slot)
        if checkpoint is None:
            fallback = STATE_DIR / f"SpeedrunSlot{slot}.state"
            if not fallback.exists():
                print(f"[CHECKPOINT {slot}] empty")
                return
            self.load_named_state(f"SpeedrunSlot{slot}")
            return
        self.load_state_bytes(
            checkpoint.state_data,
            state_name=checkpoint.name,
            state_file=str(checkpoint.path),
            reason=f"checkpoint_{slot}",
        )
        print(f"[CHECKPOINT {slot}] loaded")

    def cycle_hot_state(self, delta: int) -> None:
        name = self.hot_states.cycle(delta)
        self.recorder.log_event("hot_state_select", state_name=name or "", direction=delta)
        if name:
            print(f"[STATE] selected {name}")

    def save_hot_state(self, *, prefix: str, fixed_name: str | None = None) -> str:
        assert self.env is not None
        ram = read_smw_ram(self.env)
        name = fixed_name or f"{prefix}_{utc_id()}"
        state_data = self.env.em.get_state()
        custom_path = STATE_DIR / f"{name}.state"
        archive_path = self.recorder.states_dir / f"{name}.state"
        write_state_bytes(custom_path, state_data)
        write_state_bytes(archive_path, state_data)
        meta = {
            "session_id": self.session_id,
            "frame": self.recorder.global_frame,
            "state_name": name,
            "custom_state_file": str(custom_path),
            "session_state_file": str(archive_path),
            "ram": ram,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (self.recorder.states_dir / f"{name}.json").write_text(json.dumps(meta, indent=2) + "\n")
        self.hot_states.refresh(prefer=name)
        self.recorder.log_event("state_save", **meta)
        print(f"[STATE SAVED] {name}")
        return name

    def load_selected_state(self) -> None:
        name = self.hot_states.selected
        if not name:
            print("[STATE] no states available")
            return
        self.load_named_state(name)

    def load_named_state(self, name: str) -> None:
        path = STATE_DIR / f"{name}.state"
        if not path.exists():
            print(f"[STATE] missing {path}")
            return
        self.load_state_bytes(read_state_bytes(path), state_name=name, state_file=str(path), reason="state_load")
        self.hot_states.refresh(prefer=name)
        print(f"[STATE LOADED] {name}")

    def load_state_bytes(self, data: bytes, *, state_name: str, state_file: str, reason: str) -> None:
        assert self.env is not None
        self.env.em.set_state(data)
        ram = read_smw_ram(self.env)
        self.recorder.start_branch(reason, state_name=state_name, state_file=state_file, ram=ram)
        self.recorder.log_event("state_load", state_name=state_name, state_file=state_file, reason=reason, ram=dict(ram))
        self._reset_transition_tracking(ram)

    def reset_to_start(self) -> None:
        assert self.env is not None
        self.env.reset()
        ram = read_smw_ram(self.env)
        self.recorder.start_branch("reset", state_name=self.args.state, ram=ram)
        self.recorder.log_event("reset_to_start", state=self.args.state, ram=dict(ram))
        self._reset_transition_tracking(ram)
        print("[RESET] fresh-game branch started")

    def _reset_transition_tracking(self, ram: dict[str, int]) -> None:
        self.last_ram = dict(ram)
        self.last_game_mode = ram.get("game_mode")
        self.last_translevel = ram.get("translevel")
        self.last_lives = ram.get("lives")

    def mark_hard_spot(self, *, source: str) -> None:
        name = self.save_hot_state(prefix="HardSpot")
        self.recorder.log_event("marker", source=source, state_name=name, ram=dict(self.last_ram))
        print(f"[MARKER] hard spot saved as {name}")

    def finish(self) -> None:
        if self.finished:
            return
        self.finished = True
        self.recorder.log_event("session_close_requested")
        self.recorder.finish_branch("session_end")
        summary = {
            "session_id": self.session_id,
            "session_dir": str(self.session_dir),
            "state": self.args.state,
            "total_frames": self.recorder.global_frame,
            "total_seconds": round(self.recorder.global_frame / 60.0, 3),
            "branches": self.recorder.branch_summaries,
            "trace_every": self.args.trace_every,
            "button_order": BUTTON_ORDER,
        }
        self.session_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print("")
        print("Saved SMW speedrun session:")
        print(f"  summary: {self.session_dir / 'summary.json'}")
        print(f"  events:  {self.recorder.events_path}")
        print(f"  frames:  {self.recorder.frames_path}")
        print(f"  branches:{self.recorder.branches_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play SMW from a fresh ROM boot with frame/event recording and hot states."
    )
    parser.add_argument("--state", default="NONE", help="Initial stable-retro state; NONE boots from the ROM")
    parser.add_argument("--scale", type=int, default=3, help="Window scale")
    parser.add_argument("--session-dir", help="Override output session directory")
    parser.add_argument("--trace-every", type=int, default=1, help="Write one RAM frame trace every N frames")
    parser.add_argument("--keep-sram", action="store_true", help="Use existing rom.srm instead of a clean SRAM boot")
    parser.add_argument("--headless", action="store_true", help="Run without a window")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames, useful with --headless")
    parser.add_argument("--list-states", action="store_true", help="List locally available .state files and exit")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.trace_every < 1:
        args.trace_every = 1
    if args.list_states:
        for path in sorted(STATE_DIR.glob("*.state")):
            print(path.stem)
        return
    ensure_rom_installed()
    SMWSpeedrunPractice(args).run()


if __name__ == "__main__":
    main()
