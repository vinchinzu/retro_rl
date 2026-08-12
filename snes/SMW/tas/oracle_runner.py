"""Run the SMW early-level TAS oracle reproducibly under BizHawk 2.11."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Literal

from SMW.tas.bk2 import BK2Movie, parse_bk2, retarget_bk2
from SMW.tas.lsmv import (
    BizHawkCoreProfile,
    LSMVMovie,
    parse_lsmv,
    write_bizhawk_bk2 as write_lsmv_bk2,
)
from SMW.tas.skills import extract_level_skills
from SMW.tas.smv import SMVMovie, parse_smv, write_bizhawk_bk2 as write_smv_bk2

RunnerCoreProfile = Literal["source", "v115", "subframe-v115", "legacy"]

SMW_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SMW_ROOT.parent.parent
DEFAULT_SMV = SMW_ROOT / "tas" / "ref" / "tasvideos_1868_max_exits.unpacked.smv"
DEFAULT_LSMV = SMW_ROOT / "tas" / "ref" / "tasvideos_4144_warps.unpacked.lsmv"
DEFAULT_SOURCE = DEFAULT_LSMV
DEFAULT_ROM = SMW_ROOT / "roms" / "smw.sfc"
DEFAULT_BIZHAWK = Path.home() / ".local" / "bin" / "bizhawk"
DEFAULT_BIZHAWK_CONFIG = Path.home() / ".bizhawk" / "config.ini"
DEFAULT_LUA = SMW_ROOT / "tas" / "oracle" / "verify_first_levels.lua"


def prepare_bizhawk_config(
    source: Path,
    destination: Path,
    artifact_dir: Path,
    *,
    preferred_core: str = "BSNES",
) -> Path:
    """Create an isolated, unthrottled SNES oracle configuration."""

    config: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    config.setdefault("PreferredCores", {})["SNES"] = preferred_core
    config.update(
        {
            "SpeedPercent": 400,
            "SpeedPercentAlternate": 400,
            "FrameSkip": 0,
            "Unthrottled": True,
            "ClockThrottle": False,
            "SoundThrottle": False,
            "VSync": False,
            "VSyncThrottle": False,
            "StartPaused": False,
            "RunLuaDuringTurbo": True,
            "SoundEnabled": False,
            "DispMethod": 1,
            "PlayMovieMatchHash": True,
        }
    )
    isolated_paths = {
        "Save RAM": artifact_dir / "save_ram",
        "Savestates": artifact_dir / "states",
        "Screenshots": artifact_dir / "screenshots",
    }
    for entry in config.get("PathEntries", {}).get("Paths", []):
        if entry.get("System") == "SNES" and entry.get("Type") in isolated_paths:
            entry["Path"] = str(isolated_paths[entry["Type"]].resolve())
    for path in isolated_paths.values():
        path.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return destination


def build_command(
    *,
    bizhawk: Path,
    config: Path,
    movie: Path,
    lua: Path,
    rom: Path,
    output_dir: Path,
    target_levels: int,
    max_frames: int,
) -> list[str]:
    userdata = (
        f"out_dir:{output_dir.resolve()};"
        f"target_levels:{target_levels};max_frames:{max_frames}"
    )
    return [
        "xvfb-run",
        "-a",
        str(bizhawk.resolve()),
        "--chromeless",
        f"--config={config.resolve()}",
        f"--movie={movie.resolve()}",
        f"--lua={lua.resolve()}",
        f"--userdata={userdata}",
        str(rom.resolve()),
    ]


def run_oracle(
    *,
    run_id: str,
    target_levels: int,
    max_frames: int,
    timeout_seconds: int,
    source: Path = DEFAULT_SOURCE,
    core_profile: RunnerCoreProfile = "source",
    rom: Path = DEFAULT_ROM,
    bizhawk: Path = DEFAULT_BIZHAWK,
    source_config: Path = DEFAULT_BIZHAWK_CONFIG,
) -> dict[str, object]:
    """Convert, replay, checkpoint, and extract verified early-level skills."""

    if target_levels < 1:
        raise ValueError("target_levels must be positive")
    if max_frames < 1:
        raise ValueError("max_frames must be positive")
    for required in (source, rom, bizhawk, source_config, DEFAULT_LUA):
        if not required.exists():
            raise FileNotFoundError(required)

    movie: SMVMovie | LSMVMovie | BK2Movie
    source_format: Literal["smv", "lsmv", "bk2"]
    resolved_core_profile: BizHawkCoreProfile = "v115"
    if source.suffix.lower() == ".smv":
        source_format = "smv"
        source_id = "pangaea_1868_snes9x"
        preferred_core = "Snes9x"
        movie = parse_smv(source)
    elif source.suffix.lower() == ".lsmv":
        source_format = "lsmv"
        resolved_core_profile = "v115" if core_profile == "source" else core_profile
        source_id = f"tasvideos_4144_{resolved_core_profile.replace('-', '_')}"
        preferred_core = "BSNES"
        movie = parse_lsmv(source)
    elif source.suffix.lower() == ".bk2":
        source_format = "bk2"
        movie = parse_bk2(source)
        movie.verify_rom(rom)
        source_core = movie.header.get("Core", "")
        preferred_core = "Snes9x" if source_core == "Snes9x" else "BSNES"
        profile_name = source_core or "legacy"
        if core_profile != "source":
            resolved_core_profile = core_profile
            profile_name = core_profile
            preferred_core = "BSNES"
        safe_stem = "".join(
            character if character.isalnum() else "_" for character in source.stem
        ).strip("_")
        source_id = f"{safe_stem}_{profile_name.replace('-', '_')}"
    else:
        raise ValueError(f"unsupported TAS source extension: {source}")

    root = SMW_ROOT / "recordings" / "tas_oracle" / source_id
    root.mkdir(parents=True, exist_ok=True)
    bk2 = root / f"smw_early_{max_frames}.bk2"
    if source_format == "smv":
        write_smv_bk2(movie, bk2, rom_path=rom, max_frames=max_frames)
    elif source_format == "lsmv":
        write_lsmv_bk2(
            movie,
            bk2,
            rom_path=rom,
            max_frames=max_frames,
            core_profile=resolved_core_profile,
        )
    else:
        if core_profile == "source":
            bk2 = source
        else:
            retarget_bk2(source, bk2, core_profile=resolved_core_profile)

    output_dir = root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    config = prepare_bizhawk_config(
        source_config,
        output_dir / "bizhawk_config.ini",
        output_dir,
        preferred_core=preferred_core,
    )
    command = build_command(
        bizhawk=bizhawk,
        config=config,
        movie=bk2,
        lua=DEFAULT_LUA,
        rom=rom,
        output_dir=output_dir,
        target_levels=target_levels,
        max_frames=max_frames,
    )
    (output_dir / "launch.json").write_text(
        json.dumps(
            {
                "command": command,
                "source": str(source.resolve()),
                "source_format": source_format,
                "core_profile": core_profile,
                "resolved_core_profile": (
                    resolved_core_profile if core_profile != "source" else "source"
                ),
                "source_summary": movie.summary(),
                "bk2": str(bk2.resolve()),
                "target_levels": target_levels,
                "max_frames": max_frames,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.setdefault("TERM", "xterm")
    env.setdefault("SDL_AUDIODRIVER", "dummy")
    with (output_dir / "bizhawk.stdout.log").open("w", encoding="utf-8") as stdout:
        with (output_dir / "bizhawk.stderr.log").open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=stdout,
                stderr=stderr,
                timeout=timeout_seconds,
                check=False,
            )
    proof_path = output_dir / "proof.json"
    if not proof_path.exists():
        raise RuntimeError(
            f"BizHawk exited {completed.returncode} without proof; see {output_dir}"
        )
    proof: dict[str, object] = json.loads(proof_path.read_text(encoding="utf-8"))
    segments = proof.get("segments", [])
    if isinstance(segments, list) and segments:
        skill_paths = extract_level_skills(movie, segments, output_dir / "skills")
    else:
        skill_paths = []
    proof["bizhawk_exit_code"] = completed.returncode
    proof["skill_paths"] = [str(path) for path in skill_paths]
    proof["source"] = str(source.resolve())
    proof["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    proof["source_summary"] = movie.summary()
    proof["core_profile"] = core_profile
    proof_path.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    return proof


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--target-levels", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=60000)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--core-profile",
        choices=("source", "v115", "subframe-v115", "legacy"),
        default="source",
        help="preserve a BK2 core, or choose a bsnes core for conversion/retargeting",
    )
    parser.add_argument("--rom", type=Path, default=DEFAULT_ROM)
    parser.add_argument("--bizhawk", type=Path, default=DEFAULT_BIZHAWK)
    parser.add_argument("--bizhawk-config", type=Path, default=DEFAULT_BIZHAWK_CONFIG)
    args = parser.parse_args()
    proof = run_oracle(
        run_id=args.run_id,
        target_levels=args.target_levels,
        max_frames=args.max_frames,
        timeout_seconds=args.timeout_seconds,
        source=args.source,
        core_profile=args.core_profile,
        rom=args.rom,
        bizhawk=args.bizhawk,
        source_config=args.bizhawk_config,
    )
    print(json.dumps(proof, indent=2))
    raise SystemExit(0 if proof.get("status") == "GREEN" else 1)


if __name__ == "__main__":
    main()
