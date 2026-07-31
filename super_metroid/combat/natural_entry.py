"""Natural boss activation capture harness.

Captures a scratch save-state the first time a boss is *active* on a
continuous (or probe) prefix. Development infrastructure — not continuous
evidence by itself. Continuous acceptance still requires full power-on
integrity with natural boss flags.

Bomb Torizo remains the default capture path; other bosses use
:func:`capture_natural_activation` with a custom play prefix + catalog.

```bash
uv run python super_metroid/scripts/probe/bomb_torizo_combat.py capture-natural
uv run python super_metroid/scripts/probe/bomb_torizo_combat.py prove-natural
```

See ``docs/BOSS_PIPELINE.md``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from retro_harness.env import write_state_bytes
from super_metroid.assist import UnlimitedAmmoAssist
from super_metroid.combat.features import (
    BossCatalogEntry,
    bomb_torizo_catalog,
    features_from_state,
    get_boss_catalog,
    validate_live_enemy,
)
from super_metroid.paths import SCRATCH_STATE_DIR
from super_metroid.progression import EARLY_GAME_GRAPH
from super_metroid.routes.continuous import play_start_to_bombs
from super_metroid.routes.runtime import PlayContext, run_continuous

DEFAULT_NATURAL_ACTIVE_STATE = SCRATCH_STATE_DIR / "natural_bomb_torizo_active.state"
DEFAULT_PROVENANCE = SCRATCH_STATE_DIR / "natural_bomb_torizo_active.provenance.json"

PlayPrefixFn = Callable[[PlayContext], None]


class BossActivationCaptured(Exception):
    """Internal stop signal once combat AI is live (or capture mode hits)."""

    def __init__(
        self,
        frame: int,
        features: dict[str, object],
        *,
        boss_id: str,
    ) -> None:
        super().__init__(f"{boss_id}_active_at_frame_{frame}")
        self.frame = frame
        self.features = features
        self.boss_id = boss_id


# Back-compat alias for Bomb Torizo call sites / tests.
TorizoActivationCaptured = BossActivationCaptured


@dataclass(frozen=True)
class NaturalCaptureResult:
    success: bool
    state_path: str
    provenance_path: str | None
    capture_frame: int | None
    features: dict[str, object] | None
    outcome: str
    total_frames: int
    boss_id: str = "bomb_torizo"
    development_only: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "success": self.success,
            "bossId": self.boss_id,
            "statePath": self.state_path,
            "provenancePath": self.provenance_path,
            "captureFrame": self.capture_frame,
            "features": self.features,
            "outcome": self.outcome,
            "totalFrames": self.total_frames,
            "developmentOnly": self.development_only,
            "acceptanceWarning": (
                "Scratch capture for strategy/RL iteration only; continuous "
                "acceptance still requires natural-entry power-on evidence."
            ),
        }


@dataclass(frozen=True)
class NaturalCaptureConfig:
    """Parameters for a reusable natural-entry capture."""

    boss_id: str
    catalog: BossCatalogEntry
    play_prefix: PlayPrefixFn
    output: Path
    provenance_path: Path | None = None
    max_prefix_frames: int = 60_000
    mode: str = "active"  # active | full_hp | room_entry
    graph: Any = None
    kind: str = "natural_boss_activation"
    source_note: str = ""
    assist_factory: Callable[[], Any] | None = None


def is_capture_frame(
    state: Any,
    catalog: BossCatalogEntry,
    *,
    mode: str = "active",
) -> bool:
    """True when the live enemy0 snapshot is a usable fight start.

    ``mode``:
      - ``active``: combat AI live (features.enemy_active)
      - ``full_hp``: in room at catalog max HP, non-garbage spritemap
      - ``room_entry``: first settled ordinary frame in boss room
      - ``statue``: Bomb Torizo idle statue (legacy)
    """
    if state.room_id != catalog.room_id:
        return False
    if mode == "room_entry":
        return (
            getattr(state, "game_state", 8) == 8
            and getattr(state, "door_transition", 0) == 0
        )
    if mode == "statue":
        # Bomb Torizo-specific idle statue settle.
        if state.enemy0_hp != catalog.max_hp:
            return False
        if state.num_enemies > catalog.max_enemy_slots or state.enemy0_spritemap == 0:
            return False
        return state.enemy0_spritemap == 0x87D0 and state.samus_x < 400
    if mode == "full_hp":
        if catalog.max_hp > 0 and state.enemy0_hp != catalog.max_hp:
            return False
        if state.num_enemies > catalog.max_enemy_slots or state.enemy0_spritemap == 0:
            return False
        if (
            catalog.inactive_spritemaps
            and state.enemy0_spritemap in catalog.inactive_spritemaps
        ):
            return False
        return True
    if mode == "active":
        if catalog.max_hp > 0 and state.enemy0_hp != catalog.max_hp:
            # Prefer first active frame at full HP for strategy/RL starts.
            # Fall through to features.enemy_active if already damaged mid-fight.
            pass
        if state.num_enemies > catalog.max_enemy_slots or state.enemy0_spritemap == 0:
            return False
        feat = features_from_state(state, catalog)
        if catalog.max_hp > 0 and state.enemy0_hp == catalog.max_hp:
            return feat.enemy_active
        return feat.enemy_active
    raise ValueError(f"unknown capture mode: {mode!r}")


# Back-compat private name used by older imports.
def _is_capture_frame(state, catalog, *, mode: str) -> bool:
    return is_capture_frame(state, catalog, mode=mode)


def capture_natural_activation(
    config: NaturalCaptureConfig,
) -> NaturalCaptureResult:
    """Run ``config.play_prefix`` until capture predicate; write scratch state."""
    output = Path(config.output)
    catalog = config.catalog
    boss_id = config.boss_id
    captured: dict[str, object] = {}

    def play(ctx: PlayContext) -> None:
        session = ctx.session
        original_step = session.step

        def step_with_capture(action, reason: str):
            state = original_step(action, reason)
            if session.frame > config.max_prefix_frames:
                raise TimeoutError(
                    f"natural {boss_id} capture exceeded "
                    f"{config.max_prefix_frames} frames without activation "
                    f"(room 0x{state.room_id:04X}, hp={state.enemy0_hp}, "
                    f"spritemap 0x{state.enemy0_spritemap:04X})"
                )
            if is_capture_frame(state, catalog, mode=config.mode):
                # Extra validation for non-statue modes.
                if config.mode in {"active", "full_hp"}:
                    fails = validate_live_enemy(
                        state,
                        catalog,
                        require_active=(config.mode == "active"),
                        require_full_hp=(
                            config.mode == "full_hp" and catalog.max_hp > 0
                        ),
                    )
                    # validate_live_enemy may flag inactive for active mode
                    # if features disagree — still require empty failures.
                    if fails:
                        return state
                feat = features_from_state(state, catalog)
                write_state_bytes(output, session.env.em.get_state())  # type: ignore[attr-defined]
                captured["frame"] = session.frame
                captured["features"] = feat.to_dict()
                raise BossActivationCaptured(
                    session.frame, feat.to_dict(), boss_id=boss_id
                )
            return state

        session.step = step_with_capture  # type: ignore[method-assign]
        config.play_prefix(ctx)

    assist = (
        config.assist_factory()
        if config.assist_factory is not None
        else UnlimitedAmmoAssist(enabled=True)
    )
    result = run_continuous(
        play=play,
        assist=assist,
        graph=config.graph if config.graph is not None else EARLY_GAME_GRAPH,
        video_path=None,
        success_outcome=f"natural_{boss_id}_activation_captured",
    )

    if isinstance(result.failure, BossActivationCaptured):
        prov_path: str | None = None
        provenance_path = config.provenance_path
        if provenance_path is not None:
            provenance = {
                "schemaVersion": 1,
                "kind": config.kind,
                "bossId": boss_id,
                "mode": config.mode,
                "capturedAt": datetime.now(timezone.utc).isoformat(),
                "captureFrame": result.failure.frame,
                "statePath": str(output.resolve()),
                "features": result.failure.features,
                "source": config.source_note or f"natural capture for {boss_id}",
                "developmentOnly": True,
                "catalog": {
                    "name": catalog.name,
                    "roomId": catalog.room_id,
                    "maxHp": catalog.max_hp,
                },
                "notes": (
                    f"mode={config.mode}: first matching frame on the configured "
                    "prefix. Room-load garbage filtered via catalog max_enemy_slots "
                    "and inactive spritemaps."
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
            outcome=f"natural_{boss_id}_activation_captured",
            total_frames=result.session.frame,
            boss_id=boss_id,
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
            boss_id=boss_id,
        )

    return NaturalCaptureResult(
        success=False,
        state_path=str(output),
        provenance_path=None,
        capture_frame=None,
        features=captured.get("features"),  # type: ignore[arg-type]
        outcome="completed_without_capture",
        total_frames=result.session.frame,
        boss_id=boss_id,
    )


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

    def play_prefix(ctx: PlayContext) -> None:
        play_start_to_bombs(ctx.session, ctx.splits, ctx.segments)

    return capture_natural_activation(
        NaturalCaptureConfig(
            boss_id="bomb_torizo",
            catalog=bomb_torizo_catalog(),
            play_prefix=play_prefix,
            output=Path(output),
            provenance_path=Path(provenance_path) if provenance_path else None,
            max_prefix_frames=max_prefix_frames,
            mode=mode,
            graph=EARLY_GAME_GRAPH,
            kind="natural_bomb_torizo_activation",
            source_note="continuous play_start_to_bombs prefix",
        )
    )


def default_scratch_paths(boss_id: str) -> tuple[Path, Path]:
    """Default ``(state, provenance)`` paths under scratch for ``boss_id``."""
    state = SCRATCH_STATE_DIR / f"natural_{boss_id}_active.state"
    prov = SCRATCH_STATE_DIR / f"natural_{boss_id}_active.provenance.json"
    return state, prov


def capture_config_for_boss(
    boss_id: str,
    play_prefix: PlayPrefixFn,
    *,
    mode: str = "active",
    max_prefix_frames: int = 120_000,
    graph: Any = None,
    source_note: str = "",
) -> NaturalCaptureConfig:
    """Build a :class:`NaturalCaptureConfig` for any registered catalog boss."""
    catalog = get_boss_catalog(boss_id)
    state_path, prov_path = default_scratch_paths(boss_id)
    return NaturalCaptureConfig(
        boss_id=boss_id,
        catalog=catalog,
        play_prefix=play_prefix,
        output=state_path,
        provenance_path=prov_path,
        max_prefix_frames=max_prefix_frames,
        mode=mode,
        graph=graph,
        kind=f"natural_{boss_id}_activation",
        source_note=source_note or f"natural capture prefix for {boss_id}",
    )
