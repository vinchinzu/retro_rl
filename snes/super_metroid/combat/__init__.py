"""Full-knowledge combat: bosses plus room-enemy Overlay.

Vision-only policies are deferred until gold. Boss work here uses RAM
positions, HP, spritemaps, and known hitbox dimensions (sm-json-data), then
optional structured-state RL on top of a rule-based strategy.

Import bosses from their modules (``combat.protocol``, ``combat.bomb_torizo``,
``combat.features``, …). Room enemies live in ``combat.enemies``: scan +
Stance (Engage / Avoid / Absorb / Ignore). This package does not re-export
those names, so Overlay cannot cycle through protocol / natural_entry.

Pipeline: ``docs/BOSS_PIPELINE.md``. ADR: ``docs/adr/0008-room-enemy-overlay.md``.
"""
