"""Continuous power-on route: Morph → … → Varia → Business → Bat Cave.

One tip interface: :class:`~super_metroid.routes.tips.TipSpec` (see
:mod:`super_metroid.routes.tips`). Early tips (morph→supers) and Super+ tips
are hop-composed via ``parent_tip_id`` + spines; all finish through
:func:`~super_metroid.routes.tips.run_tip` (assist + condition plugins on
TipSpec). ``run_to`` dispatches via
:func:`~super_metroid.routes.tips.run_to_tip`.

Extend Super+: pure controller → graph → spine SpineHop/TipSegment (with CLI
fields) → TipSpec registration (derives ContinuousTip) → ``run_to`` — never
another clone runner pair.

**Public continuous API:** ``run_to``, ``play_tip`` / ``run_tip``, early
``play_*`` / ``run_*``. Report type is :class:`ContinuousRunReport` only.

Named hop tables live on :mod:`super_metroid.routes.kpdr.hops` (not re-exported
here). Super+ tips go through ``play_tip`` / ``run_tip`` / ``run_to`` — no
per-tip module aliases.
"""

from __future__ import annotations

from pathlib import Path

from super_metroid.paths import ROOM_TIMINGS_DIR
from super_metroid.routes.early_continuous import (
    CONTROLLER_PATH,
    EARLY_TIP_BY_ID,
    EARLY_TIP_SPECS,
    play_bombs,
    play_morph,
    play_spore,
    play_supers,
    run_bombs,
    run_morph,
    run_spore,
    run_supers,
)
from super_metroid.routes.kpdr.hops import (
    SUPER_TIP_BY_ID,
    SUPER_TIP_SPECS,
)
from super_metroid.routes.catalog import (
    CONTINUOUS_TIPS,
    DEFAULT_CONTINUOUS_TIP,
    ContinuousTip,
    get_continuous_tip,
    list_continuous_tips,
    register_continuous_segments,
)
from super_metroid.routes.runtime import (
    ROUTE_PLAN_PATH,
    ActionSpan,
    ContinuousRunReport,
    ProgressEvent,
    Split,
    default_artifacts,
    resolve_clean_resources,
    write_room_timing_artifact,
)
from super_metroid.routes.tips import (
    TIP_BY_ID,
    TIP_SPECS,
    TipSpec,
    play_hops,
    play_tip,
    run_tip,
    run_to_tip,
)
from super_metroid.video import VideoCaptureConfig

# Ensure both early + Super+ rows are registered (import order safe).
import super_metroid.routes.early_continuous as _early  # noqa: F401
import super_metroid.routes.kpdr.hops as _hops  # noqa: F401

_THIS = Path(__file__)

__all__ = [
    "ActionSpan",
    "Split",
    "ProgressEvent",
    "ContinuousRunReport",
    "CONTROLLER_PATH",
    "ROUTE_PLAN_PATH",
    "CONTINUOUS_TIPS",
    "DEFAULT_CONTINUOUS_TIP",
    "ContinuousTip",
    "get_continuous_tip",
    "list_continuous_tips",
    "play_morph",
    "play_bombs",
    "play_spore",
    "play_supers",
    "run_morph",
    "run_bombs",
    "run_spore",
    "run_supers",
    "run_to",
    "default_tip_artifact_paths",
    "default_tip_room_timing_path",
    "default_artifact_paths",
    "write_room_timing_artifact",
    "TipSpec",
    "TIP_SPECS",
    "TIP_BY_ID",
    "EARLY_TIP_SPECS",
    "EARLY_TIP_BY_ID",
    "SUPER_TIP_SPECS",
    "SUPER_TIP_BY_ID",
    "play_hops",
    "play_tip",
    "run_tip",
    "run_to_tip",
]


# ===========================================================================
# Tip dispatch + artifact paths
# ===========================================================================


def _resolve_tip(tip: str | ContinuousTip | None = None) -> ContinuousTip:
    if tip is None:
        return get_continuous_tip(DEFAULT_CONTINUOUS_TIP)
    if isinstance(tip, ContinuousTip):
        return tip
    return get_continuous_tip(tip)


def default_tip_artifact_paths(
    tip: str | ContinuousTip | None = None,
    *,
    clean: bool = False,
) -> tuple[Path, Path]:
    """Video/report paths for a continuous tip (default: current tip).

    When ``clean=True``, basenames use the ``_clean`` stem so Clean-track
    artifacts never overwrite assisted baselines.
    """
    return default_artifacts(_resolve_tip(tip).artifact_stem, clean=clean)


def default_tip_room_timing_path(
    tip: str | ContinuousTip | None = None,
    *,
    clean: bool = False,
) -> Path:
    """Opt-in room-timing JSON path for a tip (gitignored)."""
    resolved = _resolve_tip(tip)
    stem = resolved.artifact_stem
    if clean and not stem.endswith("_clean"):
        stem = f"{stem}_clean"
    ROOM_TIMINGS_DIR.mkdir(parents=True, exist_ok=True)
    return ROOM_TIMINGS_DIR / f"{stem}_room_timing.json"


def default_artifact_paths(*, clean: bool = False) -> tuple[Path, Path]:
    """Video/report paths for the current continuous tip (Bat Cave)."""
    return default_tip_artifact_paths(clean=clean)


def run_to(
    tip: str = DEFAULT_CONTINUOUS_TIP,
    *,
    video_path: str | Path | None = None,
    video_config: VideoCaptureConfig | None = None,
    report_path: str | Path | None = None,
    unlimited_energy: bool = True,
    unlimited_ammo: bool = True,
    room_timing_path: str | Path | None = None,
    state_output: str | Path | None = None,
    require_clean_resources: bool | None = None,
) -> ContinuousRunReport:
    """Power-on once through a named continuous tip (``--to`` target).

    Tips compose as prefixes through business; ``frog`` and ``bat_cave`` are
    sibling extensions of business (Frog Save vs Cathedral first Bubble).
    Room-timing and checkpoint output are gated by
    :class:`~super_metroid.routes.catalog.ContinuousTip` capability flags —
    not hard-coded tip-id allowlists. Default tip is the furthest
    integrity-green tip (Bat Cave / KPDR K4.4).

    Defaults keep resource assists **on**. Pass both assists off (or set
    ``require_clean_resources=True``) for Clean-track integrity.
    Early tips (morph/bombs) accept ``unlimited_energy=False`` as a no-op
    because energy refill only starts at spore+.
    """
    resolved = get_continuous_tip(tip)
    clean = resolve_clean_resources(
        unlimited_energy=unlimited_energy,
        unlimited_ammo=unlimited_ammo,
        require_clean_resources=require_clean_resources,
    )
    kwargs: dict[str, object] = {
        "video_path": video_path,
        "video_config": video_config,
        "report_path": report_path,
        "unlimited_ammo": unlimited_ammo,
        "unlimited_energy": unlimited_energy,
        "require_clean_resources": clean,
    }
    if room_timing_path is not None:
        if not resolved.supports_room_timing:
            raise ValueError(
                f"tip {resolved.tip_id!r} does not support room timing "
                f"(supported: tips with supports_room_timing=True)"
            )
        kwargs["room_timing_path"] = room_timing_path
    if state_output is not None:
        if not resolved.supports_checkpoint:
            raise ValueError(
                f"tip {resolved.tip_id!r} does not support checkpoint output "
                f"(set TipSpec.supports_checkpoint=True)"
            )
        kwargs["state_output"] = state_output

    return run_to_tip(resolved.tip_id, **kwargs)


def _play_tip_bound(tip_id: str):
    """Bind ``play_tip`` to one Super+ tip id for the segment registry."""

    def _play(session: object, splits: list[object], segments_list: list[object]) -> object:
        return play_tip(tip_id, session, splits, segments_list)

    _play.__name__ = f"play_{tip_id}"
    _play.__qualname__ = f"play_{tip_id}"
    return _play


def _continuous_segment_registry() -> dict[str, object]:
    """Tip-id → play callable from the unified TipSpec table."""
    segments: dict[str, object] = {
        "run_to": run_to,
        "morph": play_morph,
        "bombs": play_bombs,
        "spore": play_spore,
        "supers": play_supers,
    }
    for tip_id in TIP_BY_ID:
        if tip_id not in segments:
            segments[tip_id] = _play_tip_bound(tip_id)
    return segments


register_continuous_segments(_continuous_segment_registry())
