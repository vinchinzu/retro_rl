"""Capture natural Bomb Torizo activation from the continuous prefix.

Runs the accepted power-on → bombs prefix until Torizo's spritemap leaves the
idle statue (``0x87D0``), then writes a scratch save-state for strategy / RL
evaluation. This is development infrastructure — not continuous evidence.

```bash
uv run python super_metroid/scripts/probe/bomb_torizo_combat.py capture-natural
uv run python super_metroid/scripts/probe/bomb_torizo_combat.py prove-natural
```
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from retro_harness.env import write_state_bytes
from super_metroid.assist import UnlimitedAmmoAssist
from super_metroid.combat.features import bomb_torizo_catalog, features_from_state
from super_metroid.paths import SCRATCH_STATE_DIR
from super_metroid.progression import EARLY_GAME_GRAPH
from super_metroid.routes.continuous import play_start_to_bombs
from super_metroid.routes.runtime import PlayContext, run_continuous

DEFAULT_NATURAL_ACTIVE_STATE = SCRATCH_STATE_DIR / "natural_bomb_torizo_active.state"
DEFAULT_PROVENANCE = SCRATCH_STATE_DIR / "natural_bomb_torizo_active.provenance.json"


class TorizoActivationCaptured(Exception):
    """Internal stop signal once combat AI is live."""

    def __init__(self, frame: int, features: dict[str, object]) -> None:
        super().__init__(f"torizo_active_at_frame_{frame}")
        self.frame = frame
        self.features = features


@dataclass(frozen=True)
class NaturalCaptureResult:
    success: bool
    state_path: str
    provenance_path: str | None
    capture_frame: int | None
    features: dict[str, object] | None
    outcome: str
    total_frames: int
    development_only: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "success": self.success,
            "statePath": self.state_path,
            "provenancePath": self.provenance_path,
            "captureFrame": self.capture_frame,
            "features": self.features,
            "outcome": self.outcome,
            "totalFrames": self.total_frames,
            "developmentOnly": self.development_only,
            "acceptanceWarning": (
                "Scratch capture for strategy/RL iteration only; continuous "
                "acceptance still uses hash-pinned pit_to_post_torizo replay."
            ),
        }


def _is_capture_frame(state, catalog, *, mode: str) -> bool:
    """True when the continuous prefix has a usable Torizo fight snapshot.

    ``mode``:
      - ``active``: combat AI live at full HP (strategy/RL start)
      - ``statue``: settled idle statue at full HP (approach + touch)
    """
    if state.room_id != catalog.room_id:
        return False
    if state.enemy0_hp != catalog.max_hp:
        return False
    if state.num_enemies > 4 or state.enemy0_spritemap == 0:
        return False
    feat = features_from_state(state, catalog)
    if mode == "statue":
        # Settled idle chozo (after spawn 0x804F settles to 0x87D0).
        return state.enemy0_spritemap == 0x87D0 and state.samus_x < 400
    if mode == "active":
        # Combat AI live at full HP (spritemap left spawn + statue set).
        return feat.enemy_active
    raise ValueError(f"unknown capture mode: {mode!r}")


def capture_natural_bomb_torizo_activation(
    *,
    output: Path = DEFAULT_NATURAL_ACTIVE_STATE,
    provenance_path: Path | None = DEFAULT_PROVENANCE,
    max_prefix_frames: int = 60_000,
    mode: str = "active",
) -> NaturalCaptureResult:
    """Power-on continuous bombs prefix; save state at first real fight frame.

    Skips room-load garbage (leftover enemy0 from Flyway) by requiring
    ``enemy0_hp == 800`` and a non-zero spritemap with few enemy slots.
    """
    output = Path(output)
    catalog = bomb_torizo_catalog()
    captured: dict[str, object] = {}

    def play(ctx: PlayContext) -> None:
        session = ctx.session
        original_step = session.step

        def step_with_capture(action, reason: str):
            state = original_step(action, reason)
            if session.frame > max_prefix_frames:
                raise TimeoutError(
                    f"natural Torizo capture exceeded {max_prefix_frames} frames "
                    f"without activation (room 0x{state.room_id:04X}, "
                    f"hp={state.enemy0_hp}, spritemap 0x{state.enemy0_spritemap:04X})"
                )
            if _is_capture_frame(state, catalog, mode=mode):
                feat = features_from_state(state, catalog)
                write_state_bytes(output, session.env.em.get_state())  # type: ignore[attr-defined]
                captured["frame"] = session.frame
                captured["features"] = feat.to_dict()
                raise TorizoActivationCaptured(session.frame, feat.to_dict())
            return state

        session.step = step_with_capture  # type: ignore[method-assign]
        play_start_to_bombs(session, ctx.splits, ctx.segments)

    assist = UnlimitedAmmoAssist(enabled=True)
    result = run_continuous(
        play=play,
        assist=assist,
        graph=EARLY_GAME_GRAPH,
        video_path=None,
        success_outcome="natural_torizo_activation_captured",
    )

    if isinstance(result.failure, TorizoActivationCaptured):
        prov_path: str | None = None
        if provenance_path is not None:
            provenance = {
                "schemaVersion": 1,
                "kind": "natural_bomb_torizo_activation",
                "mode": mode,
                "capturedAt": datetime.now(timezone.utc).isoformat(),
                "captureFrame": result.failure.frame,
                "statePath": str(output.resolve()),
                "features": result.failure.features,
                "source": "continuous play_start_to_bombs prefix",
                "developmentOnly": True,
                "notes": (
                    f"mode={mode}: first frame with enemy0_hp==800 and "
                    f"{'combat spritemap (not 0x87D0)' if mode == 'active' else 'idle statue 0x87D0'} "
                    "on the accepted continuous bombs prefix. Room-load "
                    "garbage (leftover Flyway enemy0) is ignored."
                ),
            }
            provenance_path = Path(provenance_path)
            provenance_path.parent.mkdir(parents=True, exist_ok=True)
            provenance_path.write_text(
                json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
            )
            prov_path = str(provenance_path.resolve())
        return NaturalCaptureResult(
            success=True,
            state_path=str(output.resolve()),
            provenance_path=prov_path,
            capture_frame=result.failure.frame,
            features=result.failure.features,
            outcome="natural_torizo_activation_captured",
            total_frames=result.session.frame,
        )

    if result.failure is not None:
        return NaturalCaptureResult(
            success=False,
            state_path=str(output),
            provenance_path=None,
            capture_frame=None,
            features=None,
            outcome=result.outcome,
            total_frames=result.session.frame,
        )

    # Full bombs prefix finished without the capture hook firing (unexpected).
    return NaturalCaptureResult(
        success=False,
        state_path=str(output),
        provenance_path=None,
        capture_frame=None,
        features=captured.get("features"),  # type: ignore[arg-type]
        outcome="completed_without_capture",
        total_frames=result.session.frame,
    )
