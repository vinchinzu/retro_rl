#!/usr/bin/env python3
"""Record a multi-hop pure controller chain to MP4 (debug / visual review).

Not continuous evidence — loads a source state and runs pure controllers only.
Reason-tagged input spans are written into the report JSON so hop bodies can
be turned into ``routes/skills/`` extractions later.

```bash
# Post-Varia reverse → Business → Cathedral → Bubble (default tip debug)
uv run python snes/super_metroid/scripts/probe/record_pure_chain.py

# Cathedral stack only (from Business continuous source)
uv run python snes/super_metroid/scripts/probe/record_pure_chain.py --preset business-to-bubble

# Bubble policy alone (from CATH-04 pure source)
uv run python snes/super_metroid/scripts/probe/record_pure_chain.py --preset bubble

# Post-supers Big Pink: Charge collect + ordinary-jump return (skill source)
uv run python snes/super_metroid/scripts/probe/record_pure_chain.py --preset charge-collect-return

# Same path as continuous big_pink_to_ghz hop (Charge detour → GHZ)
uv run python snes/super_metroid/scripts/probe/record_pure_chain.py --preset big-pink-to-ghz

# K6 product shine chain: Kihunter pre-spark → Moat spark → over-ocean → WS
# (video debug; then guided_human --from ws-entrance for Phantoon ship tape)
uv run python snes/super_metroid/scripts/probe/record_pure_chain.py --preset moat-to-ws
```
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[4]
_SNES_IMPORT_ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT, globals().get("_SNES_IMPORT_ROOT", ROOT)):
    if _p is not None and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from retro_harness.actions import idle_action  # noqa: E402
from retro_harness.video import VideoCaptureConfig, VideoRecorder  # noqa: E402
from super_metroid.assist import UnlimitedResourcesAssist  # noqa: E402
from super_metroid.dev.common import boot_from_state, make_dev_env  # noqa: E402
from super_metroid.paths import GAME_DIR, INTEGRATION_DIR, RECORDINGS_DIR  # noqa: E402
from super_metroid.ram import parse_env_state, probe_pin  # noqa: E402
from super_metroid.routes.kpdr.charge_return import (  # noqa: E402
    CHARGE_BEAM_MASK,
    play_charge_beam_collect,
    play_charge_beam_return,
)
from super_metroid.routes.kpdr.pink_to_ghz import play_big_pink_to_ghz  # noqa: E402
from super_metroid.routes.kpdr.to_bat_cave import (  # noqa: E402
    BubblePhaseStop,
    play_bubble_to_bat_cave,
)
from super_metroid.routes.kpdr.wrecked_ship import (  # noqa: E402
    play_moat_to_west_ocean,
    play_west_ocean_to_ws,
)
from super_metroid.routes.kpdr.k4_norfair import (  # noqa: E402
    play_business_to_cathedral_entrance,
    play_cathedral_entrance_to_cathedral,
    play_cathedral_to_rising_tide,
    play_rising_tide_to_bubble,
)
from super_metroid.routes.kpdr import (  # noqa: E402
    play_baby_to_kihunter_return,
    play_eye_to_baby_return,
    play_kihunter_to_zeela_return,
    play_kraid_to_eye_return,
    play_varia_to_kraid,
    play_warehouse_to_business,
    play_zeela_to_warehouse_return,
)

SCRATCH = GAME_DIR / "custom_integrations" / "SuperMetroid-Snes" / "scratch"

PlayFn = Callable[[object], object]


def _pin_extra(state) -> dict[str, object]:
    """Inventory bits useful for item-detour skill extraction."""
    beams = int(getattr(state, "collected_beams", 0) or 0)
    items = int(getattr(state, "collected_items", 0) or 0)
    return {
        "beams": f"0x{beams:04X}",
        "hasCharge": bool(beams & CHARGE_BEAM_MASK),
        "items": f"0x{items:04X}",
    }


class _RecordingSession:
    """ControllerSession-compatible probe that writes every frame to video.

    Also collapses consecutive ``reason`` labels into spans for skill extraction
    (``hold(..., reason=...)`` is the primary signal controllers already emit).
    """

    def __init__(
        self,
        env,
        assist: UnlimitedResourcesAssist,
        writer: VideoRecorder | None,
    ) -> None:
        self.env = env
        self.assist = assist
        self.writer = writer
        self.frame = 0
        self.state = parse_env_state(env, mode="nav")
        self.last_action = None
        self.segment_marks: list[dict[str, object]] = []
        self.reason_spans: list[dict[str, object]] = []
        self._open_span: dict[str, object] | None = None

    def mark(self, name: str) -> None:
        st = self.state
        mark: dict[str, object] = {
            "segment": name,
            "frame": self.frame,
            "roomIdHex": f"0x{int(st.room_id):04X}",
            "samusX": int(st.samus_x),
            "samusY": int(st.samus_y),
            "pose": int(st.pose),
        }
        mark.update(_pin_extra(st))
        self.segment_marks.append(mark)

    def _close_span(self) -> None:
        if self._open_span is not None:
            self.reason_spans.append(self._open_span)
            self._open_span = None

    def _note_reason(self, reason: str, action) -> None:
        label = reason or ""
        st = self.state
        # Encode a short action fingerprint when available (list/tuple of ints).
        action_key: object
        try:
            action_key = list(action) if action is not None else None
        except TypeError:
            action_key = str(action)

        if (
            self._open_span is not None
            and self._open_span.get("reason") == label
            and self._open_span.get("action") == action_key
        ):
            self._open_span["endFrame"] = self.frame
            self._open_span["frames"] = (
                int(self._open_span["endFrame"]) - int(self._open_span["startFrame"]) + 1
            )
            self._open_span["endX"] = int(st.samus_x)
            self._open_span["endY"] = int(st.samus_y)
            self._open_span["endPose"] = int(st.pose)
            self._open_span["endRoomIdHex"] = f"0x{int(st.room_id):04X}"
            return

        self._close_span()
        self._open_span = {
            "reason": label,
            "action": action_key,
            "startFrame": self.frame,
            "endFrame": self.frame,
            "frames": 1,
            "startX": int(st.samus_x),
            "startY": int(st.samus_y),
            "startPose": int(st.pose),
            "startRoomIdHex": f"0x{int(st.room_id):04X}",
            "endX": int(st.samus_x),
            "endY": int(st.samus_y),
            "endPose": int(st.pose),
            "endRoomIdHex": f"0x{int(st.room_id):04X}",
        }

    def step(self, action, reason: str = ""):
        obs, *_ = self.env.step(action)
        self.frame += 1
        self.state = parse_env_state(self.env, frame=self.frame, mode="nav")
        self.assist.apply(self.env.data, self.state)
        self.last_action = action
        self._note_reason(reason, action)
        if self.writer is not None:
            self.writer.write_from_env(
                self.env,
                obs,
                action=action,
                frame_index=self.frame,
                room_id=int(self.state.room_id),
            )
        return self.state

    def flush_spans(self) -> None:
        self._close_span()


# Ordered pure hops: name → play fn
POST_VARIA_TO_BUBBLE: tuple[tuple[str, PlayFn], ...] = (
    ("varia-to-kraid", play_varia_to_kraid),
    ("kraid-to-eye-return", play_kraid_to_eye_return),
    ("eye-to-baby-return", play_eye_to_baby_return),
    ("baby-to-kihunter-return", play_baby_to_kihunter_return),
    ("kihunter-to-zeela-return", play_kihunter_to_zeela_return),
    ("zeela-to-warehouse-return", play_zeela_to_warehouse_return),
    ("warehouse-to-business", play_warehouse_to_business),
    ("business-to-cathedral-entrance", play_business_to_cathedral_entrance),
    ("cathedral-entrance-to-cathedral", play_cathedral_entrance_to_cathedral),
    ("cathedral-to-rising-tide", play_cathedral_to_rising_tide),
    ("rising-tide-to-bubble", play_rising_tide_to_bubble),
    ("bubble-to-bat-cave", play_bubble_to_bat_cave),
)

BUSINESS_TO_BUBBLE: tuple[tuple[str, PlayFn], ...] = (
    ("business-to-cathedral-entrance", play_business_to_cathedral_entrance),
    ("cathedral-entrance-to-cathedral", play_cathedral_entrance_to_cathedral),
    ("cathedral-to-rising-tide", play_cathedral_to_rising_tide),
    ("rising-tide-to-bubble", play_rising_tide_to_bubble),
    ("bubble-to-bat-cave", play_bubble_to_bat_cave),
)

BUBBLE_ONLY: tuple[tuple[str, PlayFn], ...] = (
    ("bubble-to-bat-cave", play_bubble_to_bat_cave),
)

# Post-supers Big Pink (K1 after Spore Super collect). Source is main-shaft
# anchor without Charge; hop bodies already tag hold() reasons for extraction.
CHARGE_COLLECT_RETURN: tuple[tuple[str, PlayFn], ...] = (
    ("charge-collect", play_charge_beam_collect),
    ("charge-return", play_charge_beam_return),
)

BIG_PINK_TO_GHZ: tuple[tuple[str, PlayFn], ...] = (
    ("big-pink-to-ghz", play_big_pink_to_ghz),
)

# K6 product shine: Moat spark → West Ocean over-ocean → green Super WS.
# Source pin only (not continuous STATUS); WS pin feeds Phantoon ship record.
MOAT_TO_WS: tuple[tuple[str, PlayFn], ...] = (
    ("moat-to-west-ocean", play_moat_to_west_ocean),
    ("west-ocean-to-ws", play_west_ocean_to_ws),
)

# Named integration anchors (not scratch) preferred when continuous-like pure
# sources are not yet cataloged for this room.
_BIG_PINK_MAIN = INTEGRATION_DIR / "dev_b1_bigpink_main_controller.state"

PRESETS: dict[str, tuple[Path, tuple[tuple[str, PlayFn], ...], str]] = {
    "post-varia-to-bubble": (
        SCRATCH / "post_varia_continuous.state",
        POST_VARIA_TO_BUBBLE,
        "post_varia_to_bubble_debug",
    ),
    "business-to-bubble": (
        SCRATCH / "post_business_continuous.state",
        BUSINESS_TO_BUBBLE,
        "business_to_bubble_debug",
    ),
    "bubble": (
        SCRATCH / "post_rising_tide_to_bubble_pure.state",
        BUBBLE_ONLY,
        "bubble_to_bat_debug",
    ),
    "charge-collect-return": (
        _BIG_PINK_MAIN,
        CHARGE_COLLECT_RETURN,
        "charge_collect_return_debug",
    ),
    "big-pink-to-ghz": (
        _BIG_PINK_MAIN,
        BIG_PINK_TO_GHZ,
        "big_pink_to_ghz_debug",
    ),
    "moat-to-ws": (
        SCRATCH / "post_kihunter_pre_moat_spark.state",
        MOAT_TO_WS,
        "moat_to_ws_debug",
    ),
}


def run_chain(
    *,
    source: Path,
    hops: tuple[tuple[str, PlayFn], ...],
    video_path: Path,
    report_path: Path,
    audio: bool = True,
    scale: int = 2,
    crf: int = 20,
    preset: str | None = None,
) -> dict[str, object]:
    if not source.is_file():
        raise FileNotFoundError(f"source state missing: {source}")

    video_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    writer: VideoRecorder | None = None
    session: _RecordingSession | None = None
    t0 = time.perf_counter()
    hop_results: list[dict[str, object]] = []
    error: str | None = None
    failed_hop: str | None = None

    try:
        boot_from_state(env, source)
        for _ in range(5):
            env.step(idle_action())
            assist.apply(env.data, parse_env_state(env, mode="nav"))

        obs = env.render()
        if obs is None:
            obs, *_ = env.step(idle_action())
            assist.apply(env.data, parse_env_state(env, mode="nav"))
            obs = env.render()
        assert obs is not None

        config = VideoCaptureConfig(
            fps=60,
            scale=scale,
            crf=crf,
            preset="veryfast",
            audio=audio,
            footer=True,
        )
        audio_rate = None
        if audio:
            audio_rate = int(env.em.get_audio_rate())  # type: ignore[attr-defined]
        writer = VideoRecorder(
            video_path,
            width=int(obs.shape[1]),
            height=int(obs.shape[0]),
            config=config,
            audio_rate=audio_rate,
        )
        session = _RecordingSession(env, assist, writer)
        # Opening freeze frame
        writer.write_from_env(
            env,
            obs,
            action=None,
            frame_index=0,
            room_id=int(session.state.room_id),
        )
        session.mark("_boot")

        for name, play in hops:
            start_frame = session.frame
            session.mark(f"{name}:start")
            print(
                f"[record] hop {name} @ frame {start_frame} "
                f"room=0x{int(session.state.room_id):04X} "
                f"xy=({int(session.state.samus_x)},{int(session.state.samus_y)})",
                flush=True,
            )
            try:
                play(session)  # type: ignore[arg-type]
                st = session.state
                hop: dict[str, object] = {
                    "segment": name,
                    "ok": True,
                    "startFrame": start_frame,
                    "endFrame": session.frame,
                    "frames": session.frame - start_frame,
                    "roomIdHex": f"0x{int(st.room_id):04X}",
                    "samusX": int(st.samus_x),
                    "samusY": int(st.samus_y),
                    "pose": int(st.pose),
                    "probePin": probe_pin(st),
                }
                hop.update(_pin_extra(st))
                hop_results.append(hop)
                session.mark(f"{name}:ok")
                print(
                    f"[record]   OK {name} frames={session.frame - start_frame} "
                    f"→ 0x{int(st.room_id):04X} "
                    f"({int(st.samus_x)},{int(st.samus_y)})",
                    flush=True,
                )
            except BubblePhaseStop as phase_stop:
                st = phase_stop.state
                hop = {
                    "segment": name,
                    "ok": False,
                    "phaseStop": phase_stop.phase,
                    "startFrame": start_frame,
                    "endFrame": session.frame,
                    "frames": session.frame - start_frame,
                    "roomIdHex": f"0x{int(st.room_id):04X}",
                    "samusX": int(st.samus_x),
                    "samusY": int(st.samus_y),
                    "pose": int(st.pose),
                    "error": str(phase_stop),
                    "metrics": dict(phase_stop.metrics),
                    "probePin": probe_pin(st),
                }
                hop.update(_pin_extra(st))
                hop_results.append(hop)
                failed_hop = name
                error = str(phase_stop)
                print(f"[record]   PHASE STOP {name}: {phase_stop}", flush=True)
                break
            except Exception as exc:
                st = session.state
                hop = {
                    "segment": name,
                    "ok": False,
                    "startFrame": start_frame,
                    "endFrame": session.frame,
                    "frames": session.frame - start_frame,
                    "roomIdHex": f"0x{int(st.room_id):04X}",
                    "samusX": int(st.samus_x),
                    "samusY": int(st.samus_y),
                    "pose": int(st.pose),
                    "error": f"{type(exc).__name__}: {exc}",
                    "probePin": probe_pin(st),
                }
                hop.update(_pin_extra(st))
                hop_results.append(hop)
                failed_hop = name
                error = f"{type(exc).__name__}: {exc}"
                print(f"[record]   FAIL {name}: {error}", flush=True)
                break
    finally:
        encoded = 0
        if session is not None:
            session.flush_spans()
        if writer is not None:
            encoded = writer.frames
            writer.close()
        env.close()

    elapsed = time.perf_counter() - t0
    final = session.state if session is not None else None
    final_payload: dict[str, object] | None = None
    if final is not None:
        final_payload = {
            "roomIdHex": f"0x{int(final.room_id):04X}",
            "samusX": int(final.samus_x),
            "samusY": int(final.samus_y),
            "pose": int(final.pose),
            "probePin": probe_pin(final),
        }
        final_payload.update(_pin_extra(final))

    reason_spans = session.reason_spans if session is not None else []
    # Compact skill-facing rollup: unique reason labels with total frames.
    reason_totals: dict[str, int] = {}
    for span in reason_spans:
        key = str(span.get("reason") or "")
        reason_totals[key] = reason_totals.get(key, 0) + int(span.get("frames") or 0)

    report: dict[str, object] = {
        "kind": "pure_chain_debug_recording",
        "preset": preset,
        "source": str(source.resolve()),
        "video": str(video_path.resolve()),
        "success": error is None and all(h.get("ok") for h in hop_results),
        "failedHop": failed_hop,
        "error": error,
        "frames": session.frame if session is not None else 0,
        "encodedFrames": encoded,
        "elapsedSec": round(elapsed, 2),
        "hops": hop_results,
        "segmentMarks": session.segment_marks if session is not None else [],
        "reasonSpans": reason_spans,
        "reasonTotals": reason_totals,
        "final": final_payload,
        "note": (
            "Debug pure-chain video only — not continuous integrity evidence. "
            "reasonSpans/reasonTotals are for skill extraction from hold() labels."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="post-varia-to-bubble",
        help="Which pure chain to record (default: post-varia-to-bubble)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print presets (source, hops, default stem) and exit",
    )
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--crf", type=int, default=20)
    args = parser.parse_args()

    if args.list:
        for name in sorted(PRESETS):
            source, hops, stem = PRESETS[name]
            print(f"{name}")
            print(f"  source={source}")
            print(f"  stem={stem}")
            print(f"  hops={[n for n, _ in hops]}")
        return

    source, hops, stem = PRESETS[args.preset]
    if args.source is not None:
        source = args.source
    video = args.video or (RECORDINGS_DIR / f"{stem}.mp4")
    report = args.report or (RECORDINGS_DIR / f"{stem}.json")

    print(f"[record] preset={args.preset}", flush=True)
    print(f"[record] source={source}", flush=True)
    print(f"[record] video={video}", flush=True)
    print(f"[record] hops={[n for n, _ in hops]}", flush=True)

    result = run_chain(
        source=source,
        hops=hops,
        video_path=video,
        report_path=report,
        audio=not args.no_audio,
        scale=args.scale,
        crf=args.crf,
        preset=args.preset,
    )
    # Slim stdout: full reasonSpans live in the report JSON.
    slim = {
        k: result[k]
        for k in (
            "kind",
            "preset",
            "source",
            "video",
            "success",
            "failedHop",
            "error",
            "frames",
            "encodedFrames",
            "elapsedSec",
            "hops",
            "reasonTotals",
            "final",
            "note",
        )
        if k in result
    }
    slim["reasonSpanCount"] = len(result.get("reasonSpans") or [])
    print(json.dumps(slim, indent=2))
    print(f"[record] wrote {video}", flush=True)
    print(f"[record] wrote {report}", flush=True)
    # Exit 0 even on expected RED so the MP4 + reasonSpans are the deliverable.
    sys.exit(0 if result.get("video") else 1)


if __name__ == "__main__":
    main()
