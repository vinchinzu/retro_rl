# TASK SM-SPAZER-GRAPH: Progression edges for Spazer detour

## Recipe step
2 graph edge

## Model
Luna

## Wave type
implement

## Own files only
- `progression.py` — edges Below Spazer ↔ Spazer Room + collect outcome
- `tests/test_progression.py` and/or `tests/test_spazer_graph.py` (**create**)
- residual optional

Depends on: `SM-SPAZER-PURE` pure green.

## Context
- Epic: [`SPAZER_EARLY.md`](SPAZER_EARLY.md)
- Continuous still uses Below Spazer → West without Spazer; graph must allow
  optional detour with Spazer beam capability flag after collect.
- Mark edges `controller_dev` until compose lands (project convention).

## Read first
- `progression.py` Bat → Below Spazer / Below → West blocks
- `docs/tasks/SM-GRAPH-BRANCH.md` style
- `tests/test_k4_speed_branches.py` (branch contract style)

## Do
1. Add room node Spazer `0xA447` if missing.
2. Edges: `below_spazer_to_spazer`, `spazer_collect` (or single collect hop),
   `spazer_to_below_spazer` return.
3. Capability: post-collect path requires / grants Spazer beam in graph terms
   consistent with other items.
4. Unit tests lock edge presence and no accidental requirement of Spazer on
   minimal warehouse path (until `SM-SPAZER-FOLD`).

## Do not
- Default continuous hops fold (that is `SM-SPAZER-FOLD`)
- STATUS claims

## Acceptance
- [ ] Graph tests green
- [ ] Minimal K2 warehouse path still valid without Spazer until fold
- [ ] Detour path expressible in graph

## Verify commands
```bash
uv run pytest super_metroid/tests/test_progression.py super_metroid/tests/test_spazer_graph.py -q
```
