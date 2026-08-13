"""Optional audio transcription helpers for editor mic annotations."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path


def transcribe_audio_file(
    audio_path: Path,
    *,
    project_root: Path,
    env_prefix: str,
    default_recording_dir: Path,
) -> dict[str, object]:
    """Transcribe ``audio_path`` via ``{PREFIX}_TRANSCRIBE_CMD`` or whisper CLI."""

    def _env(suffix: str) -> str:
        return f"{env_prefix}_{suffix}"

    command = os.environ.get(_env("TRANSCRIBE_CMD"), "").strip()
    if command:
        try:
            result = subprocess.run(
                [*shlex.split(command), str(audio_path)],
                cwd=str(project_root),
                text=True,
                capture_output=True,
                timeout=120,
                check=False,
            )
        except Exception as exc:
            return {"status": "error", "text": "", "error": str(exc), "command": command}
        text = result.stdout.strip()
        return {
            "status": "ok" if result.returncode == 0 else "error",
            "text": text,
            "command": command,
            "returnCode": result.returncode,
            "stderr": result.stderr.strip(),
        }

    whisper = shutil.which("whisper")
    if whisper is None:
        return {
            "status": "not_configured",
            "text": "",
            "hint": (
                f"Install the whisper CLI or set {env_prefix}_TRANSCRIBE_CMD to a command "
                "that accepts the WAV path."
            ),
        }
    output_dir = default_recording_dir / "transcripts"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = os.environ.get(_env("WHISPER_DEVICE")) or (
        "cuda" if shutil.which("nvidia-smi") else "cpu"
    )
    command_args = [
        whisper,
        str(audio_path),
        "--model",
        os.environ.get(_env("WHISPER_MODEL"), "turbo"),
        "--device",
        device,
        "--language",
        os.environ.get(_env("WHISPER_LANGUAGE"), "en"),
        "--output_format",
        "json",
        "--output_dir",
        str(output_dir),
        "--verbose",
        "False",
    ]
    try:
        result = subprocess.run(
            command_args,
            cwd=str(project_root),
            text=True,
            capture_output=True,
            timeout=int(os.environ.get(_env("WHISPER_TIMEOUT"), "300")),
            check=False,
        )
    except Exception as exc:
        return {"status": "error", "text": "", "error": str(exc), "engine": "whisper"}
    transcript_path = output_dir / f"{audio_path.stem}.json"
    transcript: dict[str, object] = {}
    if transcript_path.is_file():
        try:
            transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        except Exception:
            transcript = {}
    text = str(transcript.get("text") or "").strip()
    return {
        "status": "ok" if result.returncode == 0 else "error",
        "text": text,
        "segments": transcript.get("segments", []),
        "engine": "whisper",
        "device": device,
        "command": " ".join(shlex.quote(arg) for arg in command_args),
        "returnCode": result.returncode,
        "stderr": result.stderr.strip(),
        "transcriptPath": str(transcript_path) if transcript_path.is_file() else None,
    }


__all__ = ["transcribe_audio_file"]
