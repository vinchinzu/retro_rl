"""Classic L↔R arm-pump helpers and knockback recovery for Ceres.

YT reference (Kentroid TFsGVxQReMw chunk ``k0_ceres``) lives at
``policies/early_game/ceres_kentroid_spans.json`` + gitignored
``refs/yt_reference/.../chunks/k0_ceres/``. Absolute Input Display replay
desyncs on elevator lag / magnet geometry — do **not** restore fixed product
open-loop when a later leg breaks. Speed every section; re-solve tails with
room / pose / y / knockback reads (same idea as K4 knockback skills).

Classic arm-pump: dir+B with L↔R angle spam (``runway_dash`` period-2).
"""

from __future__ import annotations

from retro_harness.actions import buttons
from super_metroid.ram import GS_ORDINARY
from super_metroid.routes.kpdr.ceres.geometry import _CERES_ARM_PUMP_PERIOD
from super_metroid.routes.runtime import ActionSpan, RouteSession
from super_metroid.routes.skills.knockback import is_knockback
from super_metroid.takeoff import shoulder_pump_button


def _arm_pump_dash_spans(
    direction: str,
    frames: int,
    reason: str,
    *,
    period: int = _CERES_ARM_PUMP_PERIOD,
) -> list[ActionSpan]:
    """Expand ``dir+B`` into classic L↔R arm-pump (``runway_dash`` pattern)."""
    period = max(1, period)
    out: list[ActionSpan] = []
    i = 0
    while i < frames:
        ang = shoulder_pump_button(i, period)
        chunk = min(period, frames - i)
        out.append(ActionSpan((direction, "B", ang), chunk, reason))
        i += chunk
    return out


def _ceres_clear_knockback(
    session: RouteSession,
    direction: str,
    *,
    reason: str,
    max_frames: int = 40,
) -> None:
    """Spin-escape knockback using WRAM pose (no fixed open-loop restore)."""
    for i in range(max_frames):
        if not is_knockback(session.state):
            return
        # Short run then spin in travel direction.
        if i < 6:
            session.step(buttons(direction, "B"), f"{reason}_kb_run")
        else:
            session.step(buttons(direction, "B", "A"), f"{reason}_kb_spin")


def _ceres_wait_ordinary(
    session: RouteSession, room_id: int, *, reason: str, timeout: int = 200
) -> None:
    session.wait_until(
        lambda s: s.room_id == room_id and s.game_state == GS_ORDINARY,
        timeout=timeout,
        reason=reason,
    )


__all__ = [
    "_arm_pump_dash_spans",
    "_ceres_clear_knockback",
    "_ceres_wait_ordinary",
]
