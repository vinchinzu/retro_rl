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
from tmnt_iv.paths import GAME, RECORDINGS_DIR, ROMS_DIR
from tmnt_iv.run.metrics import HARD_VALUE, RunMetrics, format_duration, metrics_dict

BASELINE_INDEX_NAME = "baseline_index.json"
BASELINE_INDEX_SCHEMA = {
    "schema_version": 1,
    "baselines": {},
}


def _git_short() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "nogit"
    digest = (result.stdout or "").strip()
    return digest or "nogit"


def immutable_artifact_stem(*, contract: str, digest: str) -> str:
    """Date + short git + contract + digest; used for scratch evidence."""
    date = datetime.now().strftime("%Y%m%d")
    return f"{date}_{_git_short()}_{contract}_{digest[:8]}"


def scratch_report_path(*, contract: str, payload: bytes | str = b"") -> Path:
    """Scratch JSON path under recordings/. Does not overwrite named baselines."""
    raw = payload.encode("utf-8") if isinstance(payload, str) else payload
    digest = hashlib.sha256(raw).hexdigest()
    stem = immutable_artifact_stem(contract=contract, digest=digest)
    return RECORDINGS_DIR / f"scratch_{stem}.json"


def baseline_index_path() -> Path:
    """Gitignored index that points STATUS/BASELINE names at a digest."""
    return RECORDINGS_DIR / BASELINE_INDEX_NAME


def load_baseline_index(path: Path | None = None) -> dict[str, Any]:
    """Load ``recordings/baseline_index.json``, or an empty schema-1 index."""
    target = path if path is not None else baseline_index_path()
    if not target.is_file():
        return {"schema_version": 1, "baselines": {}}
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {"schema_version": 1, "baselines": {}}
    payload.setdefault("schema_version", 1)
    payload.setdefault("baselines", {})
    return payload


def promote_baseline(
    *,
    name: str,
    digest: str,
    path: Path,
    contract: str | None = None,
    frames: int | None = None,
    damage: int | None = None,
) -> dict[str, Any]:
    """Return an updated index row. Does not write STATUS or the index file."""
    index = load_baseline_index()
    row: dict[str, Any] = {
        "digest": digest,
        "path": str(path),
    }
    if contract is not None:
        row["contract"] = contract
    if frames is not None:
        row["frames"] = frames
    if damage is not None:
        row["damage"] = damage
    index.setdefault("baselines", {})[name] = row
    return index


def audit_run_fields(
    *,
    save_state_loads: int = 0,
    stage_writes: int = 0,
    lives_writes: int = 0,
    forbidden_a_special_uses: int = 0,
    post_boot_start_presses: int = 0,
) -> dict[str, int]:
    """Full-run manifest counters derived from the trial audit log."""
    return {
        "save_state_loads": int(save_state_loads),
        "stage_writes": int(stage_writes),
        "lives_writes": int(lives_writes),
        "forbidden_a_special_uses": int(forbidden_a_special_uses),
        "post_boot_start_presses": int(post_boot_start_presses),
    }


def clean_audit_ok(fields: dict[str, int]) -> bool:
    """True when Clean forbids (loads / stage / lives / A / START) are all zero."""
    return all(int(fields.get(key, 0) or 0) == 0 for key in audit_run_fields())


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
    save_state_loads: int = 0,
    stage_writes: int = 0,
    lives_writes: int = 0,
    forbidden_a_special_uses: int = 0,
    post_boot_start_presses: int = 0,
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
    audit = audit_run_fields(
        save_state_loads=save_state_loads,
        stage_writes=stage_writes,
        lives_writes=lives_writes,
        forbidden_a_special_uses=forbidden_a_special_uses,
        post_boot_start_presses=post_boot_start_presses,
    )
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
            "save_state_loads": audit["save_state_loads"],
            "stage_writes": audit["stage_writes"],
            "lives_writes": audit["lives_writes"],
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
            "forbidden_a_special_uses": audit["forbidden_a_special_uses"],
            "post_boot_start_presses": audit["post_boot_start_presses"],
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
    save_state_loads: int = 0,
    stage_writes: int = 0,
    lives_writes: int = 0,
    forbidden_a_special_uses: int = 0,
    post_boot_start_presses: int = 0,
) -> dict[str, Any]:
    """Validate, build, write, and print the success report."""
    assert_run_complete(metrics, hard_confirmed=hard_confirmed)
    integrity_flags = assist_integrity(
        metrics,
        require_clean_assists=require_clean_assists,
        state_loads=save_state_loads,
        stage_writes=stage_writes,
        lives_writes=lives_writes,
    )
    clean_ok = (not require_clean_assists) or bool(
        integrity_flags.get("clean_assists_zero", False)
    )
    audit = audit_run_fields(
        save_state_loads=save_state_loads,
        stage_writes=stage_writes,
        lives_writes=lives_writes,
        forbidden_a_special_uses=forbidden_a_special_uses,
        post_boot_start_presses=post_boot_start_presses,
    )
    if require_clean_assists and (not clean_ok or not clean_audit_ok(audit)):
        raise RuntimeError(
            "clean integrity failed: "
            f"e-heals={metrics.health_guard_interventions} "
            f"iframe_frames={metrics.final_boss_iframe_guard_frames} "
            f"loads={save_state_loads} stage_writes={stage_writes} "
            f"lives_writes={lives_writes} a={forbidden_a_special_uses} "
            f"start={post_boot_start_presses}"
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
        save_state_loads=save_state_loads,
        stage_writes=stage_writes,
        lives_writes=lives_writes,
        forbidden_a_special_uses=forbidden_a_special_uses,
        post_boot_start_presses=post_boot_start_presses,
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
