# SM-LATE-DRAY-RIDLEY-SURVEY — Draygon / Lower Norfair / Ridley diagnostic map

Date: 2026-08-01
Scope: diagnostic-only survey of the post-Draygon/Space Jump handoff and the
Lower Norfair path to Ridley. This card is not a route implementation card.

## Boundary and classification

`observed` means a development anchor or door-warp probe settled in the named
room. `partial` additionally has an existing isolated-practice result. `blocked`
means a required natural-entry or boss gate was not available. None of these
labels is a controller clearance, continuous result, or route-progress claim.

The only late sources used here were
`custom_integrations/SuperMetroid-Snes/dev_route_anchor_draygon.state` and
`custom_integrations/SuperMetroid-Snes/dev_route_anchor_ridley.state`. Per
`SOURCE_STATES.md`, they are topology-only fixtures with granted full loadout
and boss bits; they are not representative of a clean KPDR predecessor.

No tracked state, route code, progression data, registry, `STATUS.md`, or debug
artifact was written. The topology runner used development door-warps,
full-loadout grants, and boss-bit writes in its private emulator session. Its
conditional placement recovery was not needed by the listed in-bounds receipts.
All of those mechanics are evidence of room/door topology only.

## Probes actually run

1. Direct diagnostic A: load `dev_route_anchor_draygon.state` with
   `boot_from_state`, then send 60 `idle_action()` frames. Result: `0xDA60`,
   x=576/y=252/pose=155, `enemy0Hp=0`, development-only.
2. Direct diagnostic B: from that same Draygon anchor, run one `door_warp` on
   `0xA978` with expected room `0xD9AA`, then send 60 idle frames. It settled
   at game state 8; final `0xD9AA`, x=264/y=192/pose=155,
   `items=0xF32F`, development-only.
3. Topology receipt (the actual command):

   ```bash
   uv run python super_metroid/scripts/probe/route.py leg draygon ridley \
     --source-state super_metroid/custom_integrations/SuperMetroid-Snes/dev_route_anchor_draygon.state
   ```

   Result: `success=true`, 28/28 door-warp hops settled. The ten relevant
   Lower Norfair-to-Ridley receipts below each reached game state 8.
4. Direct diagnostic C: load `dev_route_anchor_ridley.state`, then 60 idle
   frames. Final `0xB32E`, x=352/y=105/pose=155, `enemy0Hp=18000`,
   `numEnemies=1`, development-only. No combat input was sent.

The K8 topology is not a direct Space Jump exit. The surveyed completion-path
chain enters Lower Norfair from Bubble Mountain:

```text
0xACB3 → 0xAD5E → 0xB656 → 0xB510 → 0xB6EE → 0xB585
       → 0xB5D5 → 0xB62B → 0xB482 → 0xB37A → 0xB32E
```

Space Jump is a Draygon side-room return (`0xDA60 --0xA978--> 0xD9AA`), so it
was separately touched rather than misreported as part of that door chain.

## Room ledger

| Room | Best known source / entry evidence | Probe actually run and result / final pin | Classification and capability or fixture gate | Next atomic card if blocked |
|---|---|---|---|---|
| `0xDA60` Draygon's Room | `dev_route_anchor_draygon.state`; full development loadout, post-boss topology fixture | Direct diagnostic A: 60 idle frames; final x=576/y=252/pose=155, enemy0Hp=0 | **observed**. No natural Draygon entry, active fight, defeat event, or clean Space Jump collect was tested. The source is explicitly dev-only. | Shared upstream gate: `SM-K4-SPEEDWAY-PURE`; then a natural K6/K7 source chain before any Draygon/Space Jump promotion. |
| `0xD9AA` Space Jump Room | Draygon left-door pointer `0xA978`; graph requires Draygon defeated | Direct diagnostic B: one warp from the Draygon anchor + 60 idle; entered game state 8 at x=264/y=192, final pose=155 | **observed**. `items=0xF32F` means the fixture cannot validate the real Space Jump PLM/item delta. | `SM-ROOM-SEG-04` is the separate dev-practice card; natural-source gate remains the shared upstream gate. |
| `0xAD5E` Single Chamber | Development topology predecessor `0xACB3` Bubble Mountain, door `0x9582` | Probe 3 receipt `0xACB3 --0x9582--> 0xAD5E`: game state 8, x=120/y=210 | **observed**. `SOURCE_STATES.md` lists no clean Bubble source; it requires capture after the open K4 forward chain. | `SM-K4-SPEEDWAY-PURE` (first missing natural predecessor). |
| `0xB656` The Musketeers' Room | `0xAD5E`, door `0x95FA` | Receipt: game state 8, x=175/y=210 | **observed**. Only full-loadout warp evidence; geometry and heat traversal were not controlled. | `SM-K4-SPEEDWAY-PURE` (shared source gate). |
| `0xB510` Lower Norfair Springball Maze Room | `0xB656`, door `0x9A56` | Receipt: game state 8, x=120/y=210 | **observed**. No clean entry fixture or maze controller was exercised. | `SM-K4-SPEEDWAY-PURE` (shared source gate). |
| `0xB6EE` Lower Norfair Fireflea Room | `0xB510`, door `0x99BA` | Receipt: game state 8, x=120/y=180 | **observed**. Only topology settle; no movement/combat policy tested. | `SM-K4-SPEEDWAY-PURE` (shared source gate). |
| `0xB585` Red Kihunter Shaft | `0xB6EE`, door `0x9AAA` | Receipt: game state 8, x=320/y=180 | **observed**. Forward exit `0xB585 → 0xB5D5` is power-bomb-gated in the graph; the dev loadout masked that prerequisite. | `SM-K4-SPEEDWAY-PURE` (shared source gate). |
| `0xB5D5` Wasteland | `0xB585`, door `0x99EA` | Receipt: game state 8, x=1344/y=40 | **observed**. Forward exit to Metal Pirates requires Super Missiles; this was not validated against a clean capacity source. | `SM-K4-SPEEDWAY-PURE` (shared source gate). |
| `0xB62B` Metal Pirates Room | `0xB5D5`, door `0x9A1A`; route graph requires Super Missiles on that entry | Receipt: game state 8, x=776/y=40. Existing isolated-practice result is the only movement/combat evidence: `SM-ROOM-METAL-04` is **PARTIAL**, pin x=699/y=187/pose=137, `max_supers=5`, `enemy0Hp=1800`. | **partial** (practice only). The topology probe did not clear the local enemy lock; room exit requires `A`, `clear_local_lock`, and `clear_room_enemies`. | Proposed `SM-LATE-DRAY-RIDLEY-METAL-01`: one Super-Missile aim/fire-range knob from a doorway-natural `0xB5D5` source, after the shared source gate exists. |
| `0xB482` Plowerhouse Room | `0xB62B`, door `0x9A32` | Receipt: game state 8, x=720/y=40 | **observed**. Its entry receipt is downstream of the unresolved Metal enemy-clear gate. | `SM-LATE-DRAY-RIDLEY-METAL-01` after its source prerequisite. |
| `0xB37A` Lower Norfair Farming Room | `0xB482`, door `0x995A` | Receipt: game state 8, x=920/y=40 | **observed**. The following Ridley eye door has `clear_local_lock`; no real door shot or room-clear action was attempted. | `SM-LATE-DRAY-RIDLEY-RIDLEY-EYE-01`: one eye-door opening probe from a natural `0xB37A` successor source (not created by this survey). |
| `0xB32E` Ridley's Room | `0xB37A`, door `0x98CA`; also `dev_route_anchor_ridley.state` | Receipt: game state 8, x=352/y=40. Direct diagnostic C confirms active fixture pin x=352/y=105/pose=155, enemy0Hp=18000, numEnemies=1 after 60 idle frames. | **blocked**. The boss pipeline requires a natural continuous-room entry before a boss strategy can be evidence; the anchor has full dev loadout/boss-state context and was not fought. | Proposed `SM-LATE-DRAY-RIDLEY-RIDLEY-NATURAL-ENTRY-01`: capture and fingerprint an ordinary `0xB37A → 0xB32E` entry only after the whole K4–K8 predecessor holds. |

## Survey result and blockers

- Touched rooms: **12/12** in scope — Draygon, Space Jump, and all ten
  `0xAD5E` through `0xB32E` completion-path rooms.
- Lower Norfair-to-Ridley classifications: **8 observed, 1 partial
  (Metal Pirates practice only), 1 blocked (Ridley natural-entry/boss gate)**.
- Every topology receipt is development-only. The `route.py leg` runner
  reports its own `developmentOnly=true`; its 28-hop success must not be used
  as a natural route exit, item collection, boss defeat, or continuous claim.
- Primary blocker: the accepted played spine ends at `0xB167` Frog
  Savestation. `SM-K4-SPEEDWAY-PURE` is the first real predecessor card; no
  continuous-like Bubble/Lower-Norfair source exists to make a K8 controller
  attempt meaningful.
- Secondary blockers: clean Power Bomb and Super Missile capacities are not
  established for the K8 source; Metal Pirates needs an enemy-clear tactic;
  the Ridley eye door needs a local clear; and Ridley itself needs natural
  entry before boss work under `BOSS_PIPELINE.md`.

## Validation

`uv run pytest tests/test_docs.py -q` — **8 passed** (0.08s).
