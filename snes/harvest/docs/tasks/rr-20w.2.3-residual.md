## Residual — rr-20w.2.3 D2 field clearing

**Status:** IN PROGRESS. 4/4 boulders are pin-green; 2 stumps and the
natural-entry Day 2 rung are not.
**Natural entry:** power-on. Named states below are diagnostic pins and do
not promote STATUS.

### Verified this session

- Hammer/axe multi-hit must stay planted. `$096D` STZs on a d-pad walk.
  `handle_tool_clear` faces once, then Y-only; `_handle_clearing` does not
  re-center after that face. Hits are credited from a live tool-counter or
  stamina edge after a short post-swing wait (the ROM can lag one frame).
  A genuine miss stays on the same stand; three misses try another footprint
  side (that walk resets the counter, which is expected).
- From `Y1_D2_After_Stones` (65 stam, 18:04, hammer fetched 1094f):
  `CLEAR_ROCKS` **4/4 GREEN**, `51 → 47` large rocks, 3747f / 62.45s,
  stam `65 → 17`. Sequential planted hits 1–5 then break. Successor pin
  `Y1_D2_After_Rocks` (tile 39,13, hammer selected, 17 stam).
- Leftover probe inserts spa on smash `stamina_low` and before stumps when
  live stamina cannot finish an 8-swing 2×2 (`should_spa_retry` /
  `needs_spa_before_next_smash`).
- Focused non-ROM suite for the changed modules passed (125+). `test_d2_spine`
  still has two unrelated evidence failures.

### Current red: stumps blocked on shed + west-gate, not on axe swings

Do not use `Y1_D2_Stumps_Frontier` (axe-route timeout overwrite, then an
earlier spa hug). Resume from `Y1_D2_After_Rocks`.

`--section stumps` from that pin:

1. `ENSURE_AXE` failed `multi_nav timeout` at 8000f. Start (39,13), shed
   waypoint ~(456,489), stuck (29,25) after push-facing (30,20)/(30,21).
   Hammer fetch from the pond stand `Y1_D2_After_Stones` (29,35) was fine;
   north-farm leftover debris still seals the shed hop.
2. Earlier spa insert from a 13-stam After_Rocks **did fire** (correct), then
   `route_mountain` pixel-stuck at (118,421) tile (7,26) A8 “gate road”
   going to (40,424). First spa red, different checkbox from the axe-shed
   timeout. A8 is in `FARM_WALKABLE` but that row hugs the shipping ditch /
   pond edge.

Planted axe policy is the same helper as hammer; it has not been live-proven
on a stump yet.

### Exact next action

From `Y1_D2_After_Rocks`, get the axe without a 8k shed timeout (open the
(30,20) corridor or start the ensure from a loaded shed-adjacent stand),
clear 2 stumps with the planted Y-only policy, and compose spa when stamina
drops below the 8-swing budget. Do not weaken the stump quota. Do not treat
the west-gate A8 hug as spa success. After those are green, the remaining
product proof is one Clean power-on through field clearing, eight wet
potatoes, and shipping before 17:00.

### Non-claims

- No STATUS promotion
- No natural power-on Day 2 completion
- No claim that all rocks or stumps are gone
- No claim that spa filled stamina and returned to work
- No claim that `Y1_D2_Stumps_Frontier` is a valid successor
