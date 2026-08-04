"""Continuous power-on route: Morph → … → Varia → Business → Bat Cave.

One tip interface: :class:`~super_metroid.routes.tips.TipSpec` (see
:mod:`super_metroid.routes.tips`). Early tips (morph→supers) and Super+ tips
are hop-composed via ``parent_tip_id`` + spines; all finish through
:func:`~super_metroid.routes.tips.run_tip` (assist + condition plugins on
TipSpec). ``run_to`` dispatches via
:func:`~super_metroid.routes.tips.run_to_tip`.

Extend Super+: pure controller → graph → spine SpineHop/TipSegment → catalog
``ContinuousTip`` → ``run_to`` — never another clone runner pair.

**Public continuous API:** ``run_to``, ``play_tip`` / ``run_tip``, early
``play_*`` / ``run_*``. Report type is :class:`ContinuousRunReport` only.

Named hop tables live on :mod:`super_metroid.routes.kpdr.hops` (not re-exported
here). Super+ ``play_<tip>`` / ``run_<tip>`` aliases are still bound on this
module for segment registry / scripts.
"""

from __future__ import annotations

from pathlib import Path

from super_metroid.paths import ROOM_TIMINGS_DIR
from super_metroid.routes.early_continuous import (
    CONTROLLER_PATH,
    EARLY_TIP_BY_ID,
    EARLY_TIP_SPECS,
    KPDR_SUPER_ROOM_PATH,
    SPORE_CONTROLLER_PATH,
    early_prefix_conditions,
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
    POST_SUPERS_TIP_BY_ID,
    POST_SUPERS_TIP_SPECS,
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
from super_metroid.routes.post_supers_aliases import install_post_supers_aliases
from super_metroid.routes.runtime import (
    ROUTE_PLAN_PATH,
    ActionSpan,
    ContinuousRunReport,
    ProgressEvent,
    RouteSession,
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
    "POST_SUPERS_TIP_SPECS",
    "POST_SUPERS_TIP_BY_ID",
    "SUPER_TIP_SPECS",
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
                f"(set ContinuousTip.supports_checkpoint=True)"
            )
        kwargs["state_output"] = state_output

    return run_to_tip(resolved.tip_id, **kwargs)


# Super+ play_<tip> / run_<tip> on this module (data-driven; not in __all__).
_POST_SUPERS_ALIASES = install_post_supers_aliases(
    globals(),
    POST_SUPERS_TIP_BY_ID,
    play_spec=play_tip,
    run_spec=run_tip,
)


def _continuous_segment_registry() -> dict[str, object]:
    """Tip-id → play callable from the unified TipSpec table."""
    segments: dict[str, object] = {}
    segments["run_to"] = run_to
    # Public play_* wrappers for early tips (hop-composed; no custom_play).
    _early_play = {
        "morph": play_morph,
        "bombs": play_bombs,
        "spore": play_spore,
        "supers": play_supers,
    }
    for tip_id, spec in TIP_BY_ID.items():
        if spec.custom_play is not None:
            segments[tip_id] = spec.custom_play
        elif tip_id in _early_play:
            segments[tip_id] = _early_play[tip_id]
        else:
            segments[tip_id] = _POST_SUPERS_ALIASES[f"play_{tip_id}"]
    return segments


register_continuous_segments(_continuous_segment_registry())
