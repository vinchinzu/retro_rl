"""SMB any% timing contracts aligned with public TAS / RTA figures.

Three clocks are defined so every reported time names its contract:

| Contract id | Start | End | FPS | Public anchor |
|-------------|-------|-----|-----|---------------|
| ``tasvideos_poweron`` | ``env.reset`` / power-on | first 8-4 ending detect | NTSC | HappyLee #1715 04:57.31 |
| ``rta_any_percent`` | first controllable 1-1 frame | first 8-4 ending detect | NTSC | HappyLee RTA note 04:54.032 |
| ``policy_seed`` | continuous seed index 0 | first 8-4 ending detect | 60.0 display | internal seed length |

Ending detect matches ``smb.ram.reached_ending`` (World 8-4 + ``oper_mode=2``).
That is the community axe / bridge-clear moment used for RTA end, not the
120-frame post-ending settle we hold for capture stability.

Segment splits use exit-detect milestones (first frame the post-exit
world/level RAM is observed). That is coarser than TAS flagpole-touch
splits but is deterministic from our runner and comparable run-to-run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

# TASVideos / BizHawk NTSC master clock for NES SMB publications.
NTSC_FPS = 60.0988138974405
# Footer HUD / simple wall-clock display rate used by our video encoder.
DISPLAY_FPS = 60.0

# ---------------------------------------------------------------------------
# Public reference figures (warps any%)
# ---------------------------------------------------------------------------

PUBLIC_REFERENCES: dict[str, dict[str, Any]] = {
    "happylee_warps_tasvideos": {
        "label": 'HappyLee NES Super Mario Bros. "warps"',
        "publication": "https://tasvideos.org/1715M",
        "submission": "https://tasvideos.org/2964S",
        "contract": "tasvideos_poweron",
        "frames": 17_868,
        "framerate": NTSC_FPS,
        "published_time": "04:57.31",
        "game_version": "Super Mario Bros. (W) [!].nes",
    },
    "happylee_warps_rta_note": {
        "label": "HappyLee warps under RTA timing (TASVideos note on #1715)",
        "publication": "https://tasvideos.org/1715M",
        "contract": "rta_any_percent",
        "published_time": "04:54.032",
        # Inverse of published RTA note at NTSC fps (rounded to nearest frame).
        "frames": round( (4 * 60 + 54.032) * NTSC_FPS ),
        "framerate": NTSC_FPS,
        "note": "Publication page: 'Using RTA timing, this run clocks in at 04:54.032'.",
    },
    "maru_rta_rules_perfect": {
        "label": "Maru warps RTA-rules theoretical perfect (community)",
        "contract": "rta_any_percent",
        "published_time": "04:54.265",
        "frames": round( (4 * 60 + 54.265) * NTSC_FPS ),
        "framerate": NTSC_FPS,
        "note": "Human-viable perfect framerule benchmark used by RTA community.",
    },
    "human_rta_wr_band": {
        "label": "Human any% RTA WR band (Niftski / averge11, public 2025)",
        "contract": "rta_any_percent",
        "published_time": "≈04:54.4",
        "note": "Band near perfect RTA-rules; not a single frozen source.",
    },
}


@dataclass(frozen=True)
class TimingContract:
    """Named start/end convention for comparing times."""

    contract_id: str
    label: str
    start: str
    end: str
    fps: float
    public_anchor: str | None = None


CONTRACTS: dict[str, TimingContract] = {
    "tasvideos_poweron": TimingContract(
        contract_id="tasvideos_poweron",
        label="TASVideos power-on movie",
        start="power_on / env.reset frame 0",
        end="first frame reached_ending (8-4, oper_mode=2)",
        fps=NTSC_FPS,
        public_anchor="happylee_warps_tasvideos",
    ),
    "rta_any_percent": TimingContract(
        contract_id="rta_any_percent",
        label="RTA any% (control start → axe)",
        start="first controllable 1-1 frame after boot/settle phase-align",
        end="first frame reached_ending (8-4, oper_mode=2)",
        fps=NTSC_FPS,
        public_anchor="happylee_warps_rta_note",
    ),
    "policy_seed": TimingContract(
        contract_id="policy_seed",
        label="Continuous seed length (display 60 Hz)",
        start="seed frame 0 after phase-align idle",
        end="first frame reached_ending (8-4, oper_mode=2)",
        fps=DISPLAY_FPS,
        public_anchor=None,
    ),
}


def format_time(frames: int, fps: float = NTSC_FPS) -> str:
    """Return ``M:SS.mmm`` (or ``MM:SS.mmm``) for a frame count at ``fps``."""
    if frames < 0:
        raise ValueError(f"frames must be >= 0, got {frames}")
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    total = frames / fps
    minutes = int(total // 60)
    seconds = total - 60 * minutes
    return f"{minutes}:{seconds:06.3f}"


def format_time_mmss(frames: int, fps: float = NTSC_FPS) -> str:
    """Return zero-padded ``MM:SS.mmm``."""
    if frames < 0:
        raise ValueError(f"frames must be >= 0, got {frames}")
    total = frames / fps
    minutes = int(total // 60)
    seconds = total - 60 * minutes
    return f"{minutes:02d}:{seconds:06.3f}"


def frames_from_time_string(time_str: str, fps: float = NTSC_FPS) -> int:
    """Parse ``M:SS.mmm`` / ``MM:SS.mmm`` into nearest frame count."""
    parts = time_str.strip().lstrip("≈").split(":")
    if len(parts) != 2:
        raise ValueError(f"expected M:SS.mmm, got {time_str!r}")
    minutes = int(parts[0])
    seconds = float(parts[1])
    return int(round((minutes * 60 + seconds) * fps))


def rta_frames(*, policy_frames_to_ending: int) -> int:
    """RTA any%: control start (seed 0) → ending detect."""
    if policy_frames_to_ending < 0:
        raise ValueError("policy_frames_to_ending must be >= 0")
    return int(policy_frames_to_ending)


def tasvideos_frames(
    *,
    boot_frames: int,
    settle_frames: int,
    policy_frames_to_ending: int,
) -> int:
    """TASVideos-style power-on: reset → ending detect (no ending settle)."""
    if min(boot_frames, settle_frames, policy_frames_to_ending) < 0:
        raise ValueError("frame counts must be >= 0")
    return int(boot_frames) + int(settle_frames) + int(policy_frames_to_ending)


def segment_splits(
    milestones: Sequence[Mapping[str, Any]],
    *,
    clock_offset: int = 0,
    fps: float = NTSC_FPS,
) -> list[dict[str, Any]]:
    """Build per-exit split table from exit-detect milestones.

    ``milestones[*].frame`` is policy-relative (seed index of the detect).
    ``clock_offset`` shifts into an absolute contract clock (0 for RTA /
    policy_seed; boot+settle for power-on absolute).
    """
    rows: list[dict[str, Any]] = []
    prev_policy = 0
    prev_clock = clock_offset
    for row in milestones:
        exit_id = str(row["exit_id"])
        policy_frame = int(row["frame"])
        clock = clock_offset + policy_frame
        seg = policy_frame - prev_policy
        rows.append(
            {
                "exit_id": exit_id,
                "policy_frame": policy_frame,
                "clock_frame": clock,
                "seg_frames": seg,
                "cum_time": format_time_mmss(clock - clock_offset, fps),
                "seg_time": format_time_mmss(seg, fps),
                "clock_time": format_time_mmss(clock, fps),
                "world": int(row.get("world", -1)),
                "level": int(row.get("level", -1)),
                "lives": int(row.get("lives", -1)),
            }
        )
        prev_policy = policy_frame
        prev_clock = clock
    return rows


def contract_result(
    contract_id: str,
    *,
    frames: int,
    segments: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Serialize one measured result under a named contract."""
    contract = CONTRACTS[contract_id]
    out: dict[str, Any] = {
        "contract_id": contract.contract_id,
        "label": contract.label,
        "start": contract.start,
        "end": contract.end,
        "fps": contract.fps,
        "frames": int(frames),
        "time": format_time_mmss(frames, contract.fps),
        "time_display_60": format_time_mmss(frames, DISPLAY_FPS),
    }
    if contract.public_anchor:
        out["public_anchor"] = contract.public_anchor
        ref = PUBLIC_REFERENCES[contract.public_anchor]
        out["public"] = {
            "label": ref["label"],
            "published_time": ref.get("published_time"),
            "frames": ref.get("frames"),
            "publication": ref.get("publication"),
        }
        pub_frames = ref.get("frames")
        if isinstance(pub_frames, int):
            delta = int(frames) - pub_frames
            out["delta_frames"] = delta
            out["delta_seconds"] = round(delta / contract.fps, 3)
            out["delta_time"] = (
                f"+{format_time_mmss(delta, contract.fps)}"
                if delta >= 0
                else f"-{format_time_mmss(-delta, contract.fps)}"
            )
    if segments is not None:
        out["segments"] = list(segments)
    return out


def build_timing_block(
    *,
    mode: str,
    boot_frames: int | None,
    settle_frames: int | None,
    policy_frames_to_ending: int,
    milestones: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the ``timing`` object attached to warp-finish reports."""
    milestones = list(milestones or [])
    policy_f = int(policy_frames_to_ending)
    rta_f = rta_frames(policy_frames_to_ending=policy_f)
    rta_segs = segment_splits(milestones, clock_offset=0, fps=NTSC_FPS)

    contracts: dict[str, Any] = {
        "rta_any_percent": contract_result(
            "rta_any_percent",
            frames=rta_f,
            segments=rta_segs,
        ),
        "policy_seed": contract_result(
            "policy_seed",
            frames=policy_f,
            segments=segment_splits(milestones, clock_offset=0, fps=DISPLAY_FPS),
        ),
    }

    if mode == "poweron" and boot_frames is not None and settle_frames is not None:
        tv_f = tasvideos_frames(
            boot_frames=boot_frames,
            settle_frames=settle_frames,
            policy_frames_to_ending=policy_f,
        )
        offset = int(boot_frames) + int(settle_frames)
        contracts["tasvideos_poweron"] = contract_result(
            "tasvideos_poweron",
            frames=tv_f,
            segments=segment_splits(
                milestones, clock_offset=offset, fps=NTSC_FPS
            ),
        )

    # Primary comparison table for STATUS / validation.
    comparisons: list[dict[str, Any]] = []
    for cid in ("rta_any_percent", "tasvideos_poweron"):
        row = contracts.get(cid)
        if not row:
            continue
        comparisons.append(
            {
                "contract_id": cid,
                "ours_frames": row["frames"],
                "ours_time": row["time"],
                "public_label": (row.get("public") or {}).get("label"),
                "public_time": (row.get("public") or {}).get("published_time"),
                "public_frames": (row.get("public") or {}).get("frames"),
                "delta_frames": row.get("delta_frames"),
                "delta_time": row.get("delta_time"),
            }
        )

    return {
        "ntsc_fps": NTSC_FPS,
        "display_fps": DISPLAY_FPS,
        "ending": "reached_ending (world 8-4, oper_mode=2); excludes ending settle",
        "segment_method": (
            "exit-detect: first frame post-exit world/level RAM matches "
            "next stage (not flagpole-touch TAS splits)"
        ),
        "mode": mode,
        "boot_frames": boot_frames,
        "settle_frames": settle_frames,
        "policy_frames_to_ending": policy_f,
        "contracts": contracts,
        "comparisons": comparisons,
        "public_references": PUBLIC_REFERENCES,
    }


def timing_from_poweron_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Derive timing contracts from a warp_finish poweron/continuous report."""
    mode = str(report.get("mode") or "poweron")
    stages = report.get("stages") or {}
    continuous = stages.get("continuous") or stages.get("continuous_suffix") or {}
    boot = stages.get("boot") or {}
    settle = stages.get("settle") or {}
    policy_frames = int(
        continuous.get("policy_frames")
        or report.get("policy_frames")
        or 0
    )
    boot_frames = boot.get("frames")
    settle_frames = boot.get("settle_frames")
    if settle_frames is None:
        settle_frames = settle.get("frames")
    if mode != "poweron":
        boot_frames = None
        # continuous Level1_1 settle is not power-on title lag; RTA still = policy.
        if mode == "continuous":
            settle_frames = settle.get("frames")
    return build_timing_block(
        mode=mode,
        boot_frames=int(boot_frames) if boot_frames is not None else None,
        settle_frames=int(settle_frames) if settle_frames is not None else None,
        policy_frames_to_ending=policy_frames,
        milestones=continuous.get("milestones") or report.get("milestones") or [],
    )


def summarize_comparisons(timing: Mapping[str, Any]) -> str:
    """One-line human summary of contract deltas."""
    parts: list[str] = []
    for row in timing.get("comparisons") or []:
        cid = row["contract_id"]
        ours = row["ours_time"]
        pub = row.get("public_time")
        delta = row.get("delta_time")
        if pub and delta is not None:
            parts.append(f"{cid}: ours {ours} vs {pub} ({delta})")
        else:
            parts.append(f"{cid}: ours {ours}")
    return "; ".join(parts)
