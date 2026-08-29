"""JSON report builder for a finished full Hard run."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from retro_harness.ram_state import GameState
from retro_harness.video import VideoCaptureConfig
from tmnt_iv.assist import (
    EMERGENCY_HP_RESTORE,
    EMERGENCY_HP_THRESHOLD,
    FORM2_IFRAME_VALUE,
    assist_integrity,
)
from tmnt_iv.paths import GAME, ROMS_DIR
from tmnt_iv.run.metrics import HARD_VALUE, RunMetrics, format_duration, metrics_dict


def rom_sha256() -> tuple[str, str]:
    """Return the local ROM filename and digest for reproducibility."""
    roms = sorted(path for path in ROMS_DIR.iterdir() if path.is_file())
    if len(roms) != 1:
        raise RuntimeError(f"expected one TMNT IV ROM, found {len(roms)}")
    path = roms[0]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path.name, digest


def file_sha256(path: Path) -> str:
    """Hash an artifact without retaining it in memory."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def probe_video(path: Path) -> dict[str, Any]:
    """Return ffprobe stream/container data for the finished MP4."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size:stream=index,codec_name,codec_type,width,height,avg_frame_rate,sample_rate,channels",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def intervention_class(*, emergency_hp: bool, iframe_hold: bool) -> str:
    """Human-readable assist class for the report."""
    if not emergency_hp and not iframe_hold:
        return "Clean"
    if emergency_hp and iframe_hold:
        return "Resource-assisted + Protection-assisted"
    if emergency_hp:
        return "Resource-assisted"
    return "Protection-assisted"


def build_full_run_report(
    *,
    metrics: RunMetrics,
    fps: float,
    audio_rate: int,
    width: int,
    height: int,
    frame: int,
    final_state: GameState,
    capture_config: VideoCaptureConfig,
    emergency_hp: bool,
    iframe_hold: bool,
    require_clean_assists: bool,
    clean_mode: bool,
    dry_run: bool,
    video_path: Path | None,
    integrity_flags: dict[str, bool],
) -> dict[str, Any]:
    """Assemble the schema-1 full-run JSON payload."""
    class_name = intervention_class(
        emergency_hp=emergency_hp, iframe_hold=iframe_hold
    )
    rom_name, rom_digest = rom_sha256()
    command = "uv run python -m tmnt_iv.scripts.record_full_hard_run"
    if clean_mode:
        command += " --clean"
    if dry_run:
        command += " --dry-run"
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "success",
        "created_at": datetime.now().astimezone().isoformat(),
        "game": GAME,
        "run": {
            "difficulty": "HARD",
            "difficulty_wram_value": HARD_VALUE,
            "continuous_emulator_session": True,
            "power_on_start": True,
            "start_state": "NONE",
            "save_state_loads": 0,
            "stage_writes": 0,
            "lives_writes": 0,
            "native_audio": not dry_run,
            "assisted": not clean_mode,
            "intervention_class": class_name,
            "clean_track": clean_mode,
            "assists": {
                "health_restore_to_96": False,
                "emergency_hp_enabled": emergency_hp,
                "iframe_hold_enabled": iframe_hold,
                "emergency_hp_threshold": EMERGENCY_HP_THRESHOLD,
                "emergency_hp_restore": EMERGENCY_HP_RESTORE,
                "super_shredder_form2_iframe_value": FORM2_IFRAME_VALUE,
                "require_clean_assists": require_clean_assists,
            },
            "forbidden_a_special_uses": 0,
            "post_boot_start_presses": 0,
        },
        "metrics": metrics_dict(metrics, fps=fps),
        "integrity": integrity_flags,
        "emulator": {
            "screen_rate": fps,
            "audio_rate": audio_rate,
            "native_width": width,
            "native_height": height,
            "frames_executed": frame,
            "video_capture": capture_config.to_dict(),
        },
        "reproducibility": {
            "rom_filename": rom_name,
            "rom_sha256": rom_digest,
            "command": command,
        },
        "final_state": {
            "frame": frame,
            "stage": final_state.stage,
            "event": int(final_state.extras.get("event", -1)),
            "menu": int(final_state.extras.get("menu", -1)),
            "lives": final_state.lives,
        },
        "artifact": None,
    }
    if video_path is not None:
        report["artifact"] = {
            "path": str(video_path),
            "sha256": file_sha256(video_path),
            "ffprobe": probe_video(video_path),
            "capture": capture_config.to_dict(),
        }
    return report


def assert_run_complete(metrics: RunMetrics, *, hard_confirmed: bool) -> None:
    """Raise if Hard, credits, or life-loss integrity failed."""
    if not hard_confirmed:
        raise RuntimeError("Hard difficulty was never confirmed in WRAM")
    if not metrics.hard_credits_event_seen:
        raise RuntimeError("Hard staff/cast credits event was not observed")
    if metrics.credits_complete_frame is None:
        raise RuntimeError("final Splinter credits scene did not complete")
    if metrics.life_losses:
        raise RuntimeError(f"run had {metrics.life_losses} life losses")


def finalize_full_run(
    *,
    metrics: RunMetrics,
    fps: float,
    audio_rate: int,
    width: int,
    height: int,
    frame: int,
    final_state: GameState,
    capture_config: VideoCaptureConfig,
    emergency_hp: bool,
    iframe_hold: bool,
    require_clean_assists: bool,
    clean_mode: bool,
    dry_run: bool,
    video_path: Path | None,
    report_path: Path,
    hard_confirmed: bool,
) -> dict[str, Any]:
    """Validate, build, write, and print the success report."""
    assert_run_complete(metrics, hard_confirmed=hard_confirmed)
    integrity_flags = assist_integrity(
        metrics, require_clean_assists=require_clean_assists
    )
    clean_ok = (not require_clean_assists) or bool(
        integrity_flags.get("clean_assists_zero", False)
    )
    if require_clean_assists and not clean_ok:
        raise RuntimeError(
            "clean integrity failed: "
            f"e-heals={metrics.health_guard_interventions} "
            f"iframe_frames={metrics.final_boss_iframe_guard_frames}"
        )
    class_name = intervention_class(
        emergency_hp=emergency_hp, iframe_hold=iframe_hold
    )
    report = build_full_run_report(
        metrics=metrics,
        fps=fps,
        audio_rate=audio_rate,
        width=width,
        height=height,
        frame=frame,
        final_state=final_state,
        capture_config=capture_config,
        emergency_hp=emergency_hp,
        iframe_hold=iframe_hold,
        require_clean_assists=require_clean_assists,
        clean_mode=clean_mode,
        dry_run=dry_run,
        video_path=video_path,
        integrity_flags=integrity_flags,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert metrics.credits_complete_frame is not None
    print(
        "complete: "
        f"{format_duration(metrics.credits_complete_frame / fps)}  "
        f"damage={metrics.total_damage_taken}  "
        f"life_losses={metrics.life_losses}  "
        f"e-heals={metrics.health_guard_interventions}  "
        f"iframe={metrics.final_boss_iframe_guard_frames}  "
        f"class={class_name}",
        flush=True,
    )
    return report
