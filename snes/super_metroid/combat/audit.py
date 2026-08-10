"""Benchmark audit bridge for the structured Super Metroid combat path."""

from __future__ import annotations

from typing import Any

from retro_harness.audit import AuditCapabilities, AttemptAudit
from super_metroid.assist import AssistTelemetry


STRUCTURED_COMBAT_AUDIT_CAPABILITIES = AuditCapabilities.all(
    "super_metroid.combat.structured-v1"
)


def structured_combat_audit_info(
    telemetry: AssistTelemetry,
    *,
    mid_run_loads: int = 0,
) -> dict[str, Any]:
    """Return the complete intervention trail owned by the combat env.

    The structured env owns every state load and all resource writes. State
    selection occurs before an attempt starts, so ordinary episodes report no
    mid-run loads. Resource telemetry counts each actual backend RAM write.
    """
    ram_writes = telemetry.energy.writes + sum(
        counter.writes for counter in telemetry.ammo.values()
    )
    assists = {"unlimited_resources": ram_writes} if ram_writes else {}
    return {
        "ram_writes": ram_writes,
        "mid_run_loads": mid_run_loads,
        "assists": assists,
        "audit_capabilities": STRUCTURED_COMBAT_AUDIT_CAPABILITIES.to_record(),
    }


def structured_combat_attempt_audit(
    telemetry: AssistTelemetry,
    *,
    mid_run_loads: int = 0,
) -> AttemptAudit:
    """Typed dry-run/publication adapter for structured combat telemetry."""
    return AttemptAudit.from_info(
        structured_combat_audit_info(telemetry, mid_run_loads=mid_run_loads)
    )
