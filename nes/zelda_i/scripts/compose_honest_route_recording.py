"""Build the longest honest Zelda I viewing tape from verified local reels.

The result is deliberately *state-seamed*: it starts with the Clean power-on
Level 1 reel, discloses the unavailable Level 2--4 continuous footage, shows
the Level 4-complete to East Key reel, discloses the East Key to Whistle seam,
and ends with the continuous Whistle basement to Level 5 Triforce reel.

This command never appends the fixture-only Ganon reel.  A real post-Level-8
pin is required before that suffix can be part of a route recording.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import textwrap
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from retro_harness.segment_runner import write_json_report
from retro_harness.video import FrameVideoWriter, concat_videos, probe_video_evidence
from zelda_i.paths import GAME_DIR, RECORDINGS_DIR

INTEGRATION_DIR = GAME_DIR / "custom_integrations" / "LegendOfZelda-Nes"
STITCHES_DIR = RECORDINGS_DIR / "stitches"
REEL_DIR = STITCHES_DIR / "reels" / "power_on_to_level5_honest_seamed"

L1_VIDEO = RECORDINGS_DIR / "level1_complete_natural.mp4"
L4_TO_EAST_KEY_VIDEO = RECORDINGS_DIR / "compose_l4_to_eastkey.mp4"
L5_VIDEO = STITCHES_DIR / "reels" / "l5_whistle_to_tf" / "l5_whistle_to_tf_assisted.mp4"
L5_REPORT = STITCHES_DIR / "l5_whistle04_to_tf_stitch.json"
L9_FIXTURE_REEL = STITCHES_DIR / "reels" / "l9_ganon_fixture_endcard" / "REEL.json"

MILESTONE_PINS = (
    "Level1Complete",
    "Level2Complete",
    "Level3Complete",
    "Level4Complete",
    "Level5Complete",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bk2_frame_count(path: Path) -> int:
    """Count fceumm BK2 controller rows without interpreting button columns."""
    with zipfile.ZipFile(path) as archive:
        lines = archive.read("Input Log.txt").decode("utf-8").splitlines()
    return sum(line.startswith("|") for line in lines)


def has_live_level8_complete_pin(integration_dir: Path = INTEGRATION_DIR) -> bool:
    """Fail-closed check for the exact forward-route boundary."""
    state = integration_dir / "Level8Complete.state"
    provenance = integration_dir / "Level8Complete.provenance.json"
    if not state.is_file() or not provenance.is_file():
        return False
    try:
        payload = json.loads(provenance.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(payload.get("natural_entry") or payload.get("route_eligible"))


def validate_l5_endpoint(report: dict[str, object]) -> None:
    final = report.get("final")
    if not isinstance(final, dict):
        raise ValueError("Level 5 report has no final snapshot")
    if not report.get("ok"):
        raise ValueError("Level 5 report is not successful")
    if int(final.get("level", -1)) != 5 or int(final.get("screen", -1)) != 0x14:
        raise ValueError("Level 5 report does not end in the Triforce room (0x14)")
    if int(final.get("triforce", 0)) & 0x10 == 0:
        raise ValueError("Level 5 report does not have Triforce bit 0x10")
    assist = report.get("assist")
    if not isinstance(assist, dict):
        raise ValueError("Level 5 report is missing Survival telemetry")
    if int(assist.get("progression_writes", -1)) != 0:
        raise ValueError("Level 5 report has progression writes")
    if int(assist.get("capacity_writes", -1)) != 0:
        raise ValueError("Level 5 report has capacity writes")


def _card_image(title: str, lines: list[str]) -> np.ndarray:
    image = Image.new("RGB", (240, 240), (5, 10, 18))
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default(size=14)
    body_font = ImageFont.load_default(size=10)
    draw.text((12, 18), title, fill=(103, 232, 164), font=title_font)
    y = 52
    for line in lines:
        wrapped = textwrap.wrap(line, width=38) or [""]
        for row in wrapped:
            draw.text((12, y), row, fill=(222, 230, 238), font=body_font)
            y += 15
        y += 5
    draw.text(
        (12, 218),
        "Assisted development compose - not Clean",
        fill=(255, 214, 102),
        font=ImageFont.load_default(size=8),
    )
    return np.asarray(image, dtype=np.uint8)


def write_card(
    path: Path, *, title: str, lines: list[str], frames: int
) -> dict[str, object]:
    frame = _card_image(title, lines)
    silence = np.zeros((533, 2), dtype=np.int16)
    with FrameVideoWriter(
        path,
        width=240,
        height=240,
        fps=60,
        scale=2,
        crf=17,
        preset="medium",
        audio_rate=32000,
        footer=False,
    ) as writer:
        for _ in range(frames):
            writer.write(frame, audio=silence)
    return {"path": str(path.resolve()), "frames": frames}


def inventory() -> dict[str, object]:
    states = sorted(INTEGRATION_DIR.glob("*.state"))
    provenances = sorted(INTEGRATION_DIR.glob("*.provenance.json"))
    bk2s = sorted(STITCHES_DIR.rglob("*.bk2"))
    return {
        "state_count": len(states),
        "provenance_count": len(provenances),
        "bk2_count": len(bk2s),
        "states": [str(path.relative_to(GAME_DIR)) for path in states],
        "bk2s": [
            {
                "path": str(path.relative_to(GAME_DIR)),
                "bytes": path.stat().st_size,
                "frames": bk2_frame_count(path),
                "sha256": sha256_file(path),
            }
            for path in bk2s
        ],
        "milestone_pins": {
            pin: {
                "state": (INTEGRATION_DIR / f"{pin}.state").is_file(),
                "provenance": (INTEGRATION_DIR / f"{pin}.provenance.json").is_file(),
            }
            for pin in MILESTONE_PINS
        },
        "live_level8_complete": has_live_level8_complete_pin(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REEL_DIR)
    args = parser.parse_args(argv)

    parts = (L1_VIDEO, L4_TO_EAST_KEY_VIDEO, L5_VIDEO)
    missing = [str(path) for path in (*parts, L5_REPORT) if not path.is_file()]
    if missing:
        parser.error("missing required artifact(s): " + ", ".join(missing))

    l5_report = json.loads(L5_REPORT.read_text())
    validate_l5_endpoint(l5_report)

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    card_l2_l4 = out / "seam_l2_l4.mp4"
    card_l5 = out / "seam_east_key_to_whistle.mp4"
    card1 = write_card(
        card_l2_l4,
        title="DISCLOSED STATE SEAMS",
        lines=[
            "Level 1 Triforce -> Level 2 complete -> Level 3 complete -> Level 4 complete.",
            "Those assisted pins exist, but no continuous L2-L4 tape exists. This card does not pretend otherwise.",
        ],
        frames=360,
    )
    card2 = write_card(
        card_l5,
        title="DISCLOSED LEVEL 5 SEAM",
        lines=[
            "East Key -> Whistle basement footage is missing.",
            "The next continuous reel starts in Whistle basement with the Recorder already earned.",
        ],
        frames=300,
    )

    ordered_parts = [L1_VIDEO, card_l2_l4, L4_TO_EAST_KEY_VIDEO, card_l5, L5_VIDEO]
    source_evidence = [probe_video_evidence(path, -1) for path in ordered_parts]
    expected_frames = sum(int(item["frames"]) for item in source_evidence)
    video_path = out / "power_on_to_level5_honest_seamed.mp4"
    concat_videos(ordered_parts, video_path, reencode=False)
    video_evidence = probe_video_evidence(video_path, expected_frames)

    inv = inventory()
    manifest = {
        "ok": bool(video_evidence["frame_count_matches"]),
        "tape_kind": "state_seamed_viewing_compose",
        "continuous_emulator_session": False,
        "playable_media": True,
        "track": "mixed_clean_l1_and_assisted_development",
        "status_claim": False,
        "clean_claim": False,
        "route_eligible": False,
        "route_eligible_through": "Level1Complete",
        "honest_development_boundary": "Level5Complete (state-seamed assisted)",
        "start": "power-on",
        "last_honest_room": {"name": "Level 5 Triforce room", "hex": "0x14"},
        "final_triforce": "0x1c",
        "final_l5_bit_0x10": True,
        "video": video_evidence,
        "parts": source_evidence,
        "cards": [card1, card2],
        "seams": [
            {
                "after": "Level 1 Triforce room (0x36)",
                "before": "Level 4 complete checkpoint",
                "missing": "continuous Level 2, Level 3, and Level 4 gameplay",
            },
            {
                "after": "East Key Pols Voice (0x77)",
                "before": "Whistle basement (0x04)",
                "missing": "continuous East Key to Recorder acquisition",
            },
            {
                "after": "Level 5 Triforce room (0x14)",
                "before": None,
                "missing": "Level 5 exit, Levels 6-8, and live Level 9 entry",
            },
        ],
        "l5_assist": l5_report["assist"],
        "resource_pokes": [],
        "health_fix_later": l5_report.get("health_drop_note"),
        "inventory": inv,
        "level8_gate": {
            "live_complete_pin": inv["live_level8_complete"],
            "l9_attached": False,
            "reason": "No route-eligible Level8Complete state/provenance pair exists.",
        },
        "ganon_fixture_reel": {
            "path": str(L9_FIXTURE_REEL.resolve()),
            "kept_separate": True,
            "route_eligible": False,
        },
    }
    manifest_path = out / "power_on_to_level5_honest_seamed.json"
    write_json_report(manifest_path, manifest)
    print(f"VIDEO {video_path}")
    print(f"MANIFEST {manifest_path}")
    print(
        "END Level 5 Triforce room (0x14); "
        f"frames={video_evidence['frames']} level8_pin={inv['live_level8_complete']}"
    )
    return 0 if manifest["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
