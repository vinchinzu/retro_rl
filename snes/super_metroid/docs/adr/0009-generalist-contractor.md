# 0009 — Generalist is a contractor, not the spine

## Status

Accepted

## Context

The skill library (SpineHop, reactive room policy, boss strategy) owns the
product. Many rooms have no compiled skill yet. A neural net trained across
practice-hack startpoints can cover those rooms while a hop is being Chipped,
and can hand off when a LeaveSpec matches.

Practice-hack presets are exact on the practice ROM after the loader returns
to game state 8. Practice WRAM is not a vanilla pin.

## Decision

- The generalist is a **contractor**: goal-conditioned, not a Skill, not the Tip.
- Training pins are practice-ROM gs=8 captures. Product evidence never loads
  the practice ROM.
- Success is **Join** (`hop_glance` against the next LeaveSpec), not room-id.
- STATUS and `DEFAULT_CONTINUOUS_TIP` do not move because a net dualled green.

## Consequences

RoomAutopilot may run the generalist when no verified skill exists. Capture
artifacts live under `custom_integrations/SuperMetroid-Snes/practice_repertoire/`
and are gitignored `*.state` files.
