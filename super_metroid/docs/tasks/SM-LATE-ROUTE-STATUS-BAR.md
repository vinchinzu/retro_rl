# SM-LATE-ROUTE-STATUS-BAR — consolidated late-route room ledger

## Scope and reading rule

This is the single-room reconciliation of:

- `SM-LATE-GRAVITY-DRAY-SURVEY.md` (18 rows),
- `SM-LATE-DRAY-RIDLEY-SURVEY.md` (12 rows), and
- `SM-LATE-RIDLEY-MB-ESCAPE-SURVEY.md` (48 rows).

The surveys contain 78 row occurrences.  Removing the 17 repeated occurrences
leaves **61 unique scoped rooms**.  A room appears exactly once in the grouped
ledger below, even when a later survey also observed it from the reverse route.

> **DEVELOPMENT-ONLY TOPOLOGY — not route evidence.** `DEV` means a probe
> settled in ordinary gameplay after an absolute door warp or dev-anchor boot.
> The anchors have granted late loadout and major-boss context; the route runner
> can also write boss/event state.  These receipts establish room/door topology
> only.  They do not establish movement, a natural predecessor, an item/PLM
> delta, a boss defeat, a timer, a ship trigger, or a continuous suffix.

> **ROUTE-READY / NATURAL CLEAR — none.** No **observed** state in this bar is
> a natural route clear.  The accepted played spine still ends at Frog
> Savestation `0xB167`; `post_frog_continuous.state` is the only relevant
> integrity-green source named here.

### Primary-state convention

Each scoped room has exactly one primary state:

- **observed** — a `DEV` topology receipt only;
- **partial** — a targeted residual, practice result, or unit-tested scaffold
  exists, but its natural source/event requirement is missing; or
- **blocked** — the next meaningful evidence is a natural source, boss/event,
  or item-closeout gate, not another dev warp.

On an overlap, the bar retains the more conservative actionable state.  Thus
Draygon and Space Jump remain **blocked** despite later `DEV` settles, Metal
Pirates remains **partial** despite a reverse-route receipt, and Ridley remains
**blocked** despite the dev-only combat scaffold.  This is a work-state
reconciliation, not a claim that a diagnostic could not load the room.

### Evidence inventory — not progress

`[ observed ████████████████████ 50 | partial ███ 7 | blocked ██ 4 ] = 61 unique rooms`

| Primary state | Count | Meaning |
|---|---:|---|
| observed | **50** | Development-only topology settle; no natural clear. |
| partial | **7** | Isolated evidence exists, but the valid source/event is absent. |
| blocked | **4** | Natural gate must be established before local work is useful. |
| **total** | **61** | Deduplicated late corridor inventory. |

## Dependency spine: Frog Save → ship/credits

```text
Frog Save 0xB167 (accepted continuous source)
  → Frog Speedway 0xB106 [first missing natural hop: SM-K4-SPEEDWAY-PURE]
  → K4 Speed / Wave / Ice loop [unplayed] → Alpha Power Bombs [unplayed]
  → K6 Ship / Phantoon → Gravity
  → K7 Botwoon → Draygon → Space Jump (item closeout)
  → K8 re-entry at Bubble Mountain (not a direct Space Jump edge)
  → Lower Norfair → Ridley → post-Ridley return → G4
  → Tourian → Mother Brain → Escape 1–4 → Climb → Parlor → Landing Site
  → ship trigger / ending / credits
```

The first executable dependency is **`SM-K4-SPEEDWAY-PURE`** from
`scratch/post_frog_continuous.state` in `0xB167`.  `SOURCE_STATES.md` has no
clean Bubble, K6, K7, K8, G4, Tourian, MB, or escape source.  In particular,
the surveyed K8 door chain starts at Bubble Mountain; Space Jump is Draygon's
side-room closeout, not a direct `0xD9AA → Lower Norfair` edge.

## Deduplicated room ledger

`DEV-G` = `dev_route_anchor_gravity_suit`; `DEV-B` =
`dev_route_anchor_botwoon`; `DEV-D` = `dev_route_anchor_draygon`; `DEV-R` =
`dev_route_anchor_ridley`; `DEV-M` = `dev_route_anchor_mother_brain`.  All are
topology-only fixtures with granted late loadout/boss context.  `P1`–`P5` are
the development-only probes defined in the Ridley→MB→Escape survey; `A` and
`B` are the Gravity→Botwoon and Botwoon→Draygon legs respectively.

### K6/K7 boundary — Gravity → Botwoon → Draygon → Space Jump (18)

| Room | Primary state | Development-only receipt / reconciliation note |
|---|---|---|
| `0xCE40` Gravity Suit Room | **partial** | `DEV-G` is transition-unstable; SEG-08 reached the West Ocean boundary but did not prove the live Gravity PLM. See blocker register. |
| `0x93FE` West Ocean | **observed** | A, `DEV-G` → ordinary settle at x=1720/y=853. |
| `0x95FF` The Moat | **observed** | A, `DEV-G` → x=640/y=85. |
| `0x948C` Crateria Kihunter Room | **observed** | A, `DEV-G` → x=840/y=85. |
| `0x962A` Elevator to Caterpillar | **observed** | A, `DEV-G` → x=120/y=180. |
| `0xA322` Caterpillar Room | **observed** | A, `DEV-G` → x=128/y=44.  P1 later re-observed its reverse route; that does not add a row. |
| `0xD104` Red Fish Room | **observed** | A, `DEV-G` → x=120/y=180.  P1 later re-observed its reverse route. |
| `0xD0B9` Mt. Everest | **observed** | A, `DEV-G` → x=632/y=40.  P1 later re-observed its reverse route. |
| `0xD1A3` Crab Shaft | **observed** | A, `DEV-G` → x=120/y=180. |
| `0xD5A7` Aqueduct | **observed** | A, `DEV-G` → x=120/y=203. |
| `0xD617` Botwoon Hallway | **observed** | A, `DEV-G` → x=120/y=356. |
| `0xD95E` Botwoon's Room | **blocked** | A arrived from `DEV-G`; `DEV-B` already has the Botwoon bit. No natural active fight/exit. See blocker register. |
| `0xD7E4` Botwoon Energy Tank Room | **observed** | B, `DEV-B` → x=120/y=180; its post-Botwoon meaning was fixture-forged. |
| `0xD913` Halfie Climb Room | **observed** | B, `DEV-B` → x=120/y=210. |
| `0xD72A` Colosseum | **observed** | B, `DEV-B` → x=120/y=210. |
| `0xD78F` The Precious Room | **observed** | B, `DEV-B` → x=120/y=187. |
| `0xDA60` Draygon's Room | **blocked** | B / `DEV-D` settled the room; neither is a natural Botwoon exit or active Draygon fight. See blocker register. |
| `0xD9AA` Space Jump Room | **blocked** | `DEV-D --0xA978→` settled it, but the fixture already owns Space Jump. No gray-door/PLM closeout was testable. See blocker register. |

### K8 — Bubble Mountain → Lower Norfair → Ridley (11)

The ten Lower Norfair forward receipts came from the 28-hop `DEV-D` leg.
P1 later traversed the same chain in reverse after a fixture-marked Ridley;
those overlapping receipts are retained as provenance, not duplicate rooms.

| Room | Primary state | Development-only receipt / reconciliation note |
|---|---|---|
| `0xACB3` Bubble Mountain | **observed** | P1 `DEV-R` reverse receipt at (504,326); there is no clean Bubble source. |
| `0xAD5E` Single Chamber | **observed** | `DEV-D` forward receipt `(120,210)`; P1 later reverse receipt `(1584,70)`. |
| `0xB656` The Musketeers' Room | **observed** | `DEV-D` forward `(175,210)`; P1 reverse `(1128,582)`. |
| `0xB510` Lower Norfair Springball Maze Room | **observed** | `DEV-D` forward `(120,210)`; P1 reverse `(672,326)`. |
| `0xB6EE` Lower Norfair Fireflea Room | **observed** | `DEV-D` forward `(120,180)`; P1 reverse `(216,838)`. |
| `0xB585` Red Kihunter Shaft | **observed** | `DEV-D` forward `(320,180)`; P1 reverse `(687,1350)`; dev loadout masks the PB gate. |
| `0xB5D5` Wasteland | **observed** | `DEV-D` forward `(1344,40)`; P1 reverse `(175,699)`; dev loadout masks the Super gate. |
| `0xB62B` Metal Pirates Room | **partial** | `DEV-D` forward and P1 reverse settle; `SM-ROOM-METAL-04` is only isolated practice and did not clear the local enemy lock. See blocker register. |
| `0xB482` Plowerhouse Room | **observed** | `DEV-D` forward `(720,40)`; P1 reverse `(120,210)`; both depend on the unresolved Metal gate. |
| `0xB37A` Lower Norfair Farming Room | **observed** | `DEV-D` forward `(920,40)`; P1 fixture-marked post-Ridley return `(120,210)`. |
| `0xB32E` Ridley's Room | **blocked** | `DEV-R` can idle at active fixture HP, and P1 booted it, but no natural `0xB37A → 0xB32E` entry exists. See blocker register. |

### Post-Ridley return and G4, new rooms only (15)

`0xB37A` through `0xACB3` are already deduplicated in K8 above.  P1 also
re-observed `0xD0B9`, `0xD104`, `0xA322`, `0x962A`, and `0x948C`, which remain
in the K7 group rather than being counted again here.

| Room | Primary state | Development-only receipt / reconciliation note |
|---|---|---|
| `0xAFA3` Rising Tide | **observed** | P1 `DEV-R` → (1472,70). |
| `0xA788` Cathedral | **observed** | P1 `DEV-R` → (904,326). |
| `0xA7B3` Cathedral Entrance | **observed** | P1 `DEV-R` → (848,70). |
| `0xA7DE` Business Center | **observed** | P1 `DEV-R` → (295,838); earlier continuous Business evidence is not this post-Ridley handoff. |
| `0xA6A1` Warehouse Entrance | **observed** | P1 `DEV-R` → (128,291); earlier continuous Warehouse evidence is not this post-Ridley handoff. |
| `0xCF80` East Tunnel | **observed** | P1 `DEV-R` → (328,271). |
| `0xCEFB` Glass Tunnel | **observed** | P1 `DEV-R` → (295,271). |
| `0xCFC9` Main Street | **observed** | P1 `DEV-R` → (295,1946). |
| `0x95D4` Crateria Tube | **observed** | P1 `DEV-R` → (328,168). |
| `0x91F8` Landing Site | **observed** | P1 `(2343,1192)` and P5 forced-escape arrival `(500,300)`; neither tests ship/credits. |
| `0x92FD` Parlor and Alcatraz | **observed** | P1 `(1264,168)` and P5 escape-direction arrival `(376,1357)`; no timer-bearing source. |
| `0x990D` Terminator Room | **observed** | P1 `DEV-R` → (1720,168). |
| `0x99BD` Green Pirates Shaft | **observed** | P1 `DEV-R` → (384,1192). |
| `0xA5ED` Statues Hallway | **observed** | P1 `DEV-R` → (120,180); four-boss/statue state was fixture-written. |
| `0xA66A` Statues Room | **observed** | P1 `DEV-R` → (128,180); no natural G4 departure. |

### K9 — Tourian → Mother Brain → Escape (17)

| Room | Primary state | Development-only receipt / reconciliation note |
|---|---|---|
| `0xDAAE` Tourian Elevator | **observed** | P2 booted `DEV-R` and used absolute door `0x9222` → (128,44); not a physical G4 departure. |
| `0xDAE1` Metroid Room 1 | **observed** | P3 `DEV-R` / absolute Tourian fixture → (1576,50). |
| `0xDB31` Metroid Room 2 | **observed** | P3 → (240,50). |
| `0xDB7D` Metroid Room 3 | **observed** | P3 → (39,50). |
| `0xDBCD` Metroid Room 4 | **observed** | P3 → (120,180). |
| `0xDC19` Tourian Hopper Room | **observed** | P3 → (376,40). |
| `0xDC65` Dust Torizo Room | **observed** | P3 → (576,40). |
| `0xDCB1` Big Boy Room | **observed** | P3 → (1032,40). |
| `0xDCFF` Seaweed Room | **observed** | P3 → (208,40). |
| `0xDDC4` Tourian Eye Door Room | **observed** | P3 → (7,40). |
| `0xDDF3` Rinka Shaft | **observed** | P3 → (120,210); no natural MB doorway activation. |
| `0xDD58` Mother Brain's Room | **partial** | P3 and `DEV-M` locate it; the MB scaffold is unit-tested only, with no natural Rinka arrival, fight, defeat, or exit. See blocker register. |
| `0xDE4D` Tourian Escape Room 1 | **partial** | P4 reached it only after `DEV-M` wrote MB/Tourian events and armed the escape fixture. See blocker register. |
| `0xDE7A` Tourian Escape Room 2 | **partial** | P4 forced-event/timer fixture → (8,70); no played escape edge. See blocker register. |
| `0xDEA7` Tourian Escape Room 3 | **partial** | P4 forced-event/timer fixture → (120,210); no played escape edge. See blocker register. |
| `0xDEDE` Tourian Escape Room 4 | **partial** | P4 forced-event/timer fixture → (200,180); P5 then used an absolute exit. See blocker register. |
| `0x96BA` The Climb | **observed** | P5 absolute escape fixture → (120,180); no natural timer-bearing Escape 4 departure. |

## Partial / blocked register: fixture quality, exact blocker, next atom

The row state above is primary.  This register supplies the source quality and
the immediate atomic work for every non-observed room; it does not promote a
future card to an existing result.  `proposed` means the named atomic card was
specified by a source survey but is not asserted to be a tracked card yet.

| Room / state | Source or fixture quality | Exact blocker | Next atomic card |
|---|---|---|---|
| `0xCE40` Gravity — **partial** | `DEV-G` is a full-loadout/boss-complete topology fixture that remains in `gameState=11` transition; SEG-08 only reached the West Ocean residual pin. | Need a controllable natural `0xC98E → 0xCE40` source with the Phantoon/WS power bit set and Gravity uncollected, then the live PLM delta. | `SM-ROOM-SEG-08-R1` (capture that exact source; `SM-ROOM-SEG-08-SRC` is the catalog companion). |
| `0xD95E` Botwoon — **blocked** | A arrived on `DEV-G`; `DEV-B` has full loadout and Botwoon's boss bit already set. | No natural active Botwoon entry from `0xD617`, fight, defeat bit, or post-fight exit. | `SM-BOTW-NATURAL-ENTRY` (**proposed**): capture only an active natural `0xD617 → 0xD95E` entry. |
| `0xDA60` Draygon — **blocked** | B and `DEV-D` are boss-complete/full-loadout topology inputs; their idle receipts do not activate a real fight. | No natural Botwoon exit → active Draygon entry, legitimate defeat, or closeout provenance. | `SM-DRAY-NATURAL-ENTRY` (**proposed**), after the global first missing predecessor `SM-K4-SPEEDWAY-PURE`. |
| `0xD9AA` Space Jump — **blocked** | `DEV-D --0xA978→` settles the room with `items=0xF32F`, which already contains Space Jump. | Need natural Draygon defeat, gray-door closeout, an uncollected live Space Jump PLM, and its item delta/fanfare. | `SM-DRAY-CLOSEOUT-01` (**proposed**) from the natural Draygon completion. |
| `0xB62B` Metal Pirates — **partial** | Forward/reverse receipts are full-loadout topology only. `SM-ROOM-METAL-04` is isolated practice (pin x=699/y=187/pose=137, `max_supers=5`, `enemy0Hp=1800`), not a doorway-natural clear. | No natural `0xB5D5` source with real Super capacity; local enemy/lock clear still requires `A`, `clear_local_lock`, and `clear_room_enemies`. | `SM-LATE-DRAY-RIDLEY-METAL-01` (**proposed**): one Super-Missile aim/fire-range knob from that natural source. |
| `0xB32E` Ridley — **blocked** | `DEV-R` has granted late context; P1 merely booted it, and the dev combat scaffold has tests only. | No ordinary `0xB37A → 0xB32E` entry, natural fight/defeat, or post-fight exit; the boss pipeline forbids promotion before natural entry. | `SM-LATE-RIDLEY-NATURAL-ENTRY-01`: capture and fingerprint one settled unmodified doorway-natural entry. |
| `0xDD58` Mother Brain — **partial** | P3 is an absolute Tourian fixture; `DEV-M` is a granted late anchor. The 17-test combat/escape scaffold check is not a fight result. | No natural Rinka Shaft arrival, active MB activation, rainbow/hyper handling, defeat event, or exit. | `SM-LATE-MB-NATURAL-ENTRY-01`; only then `SM-LATE-MB-ACTIVATION-01`. |
| `0xDE4D` Escape 1 — **partial** | P4 begins from `DEV-M` after development MB/Tourian/event writes and escape-timer setup; escape scaffold tests are shell coverage only. | No naturally defeated-MB doorway transition and no gameplay-origin timer. | `SM-LATE-ESCAPE1-NATURAL-SRC-01`. |
| `0xDE7A` Escape 2 — **partial** | P4's timer/event source is fixture-forced; no movement controller was exercised. | Requires the natural Escape 1 source and its real first exit while preserving timer provenance. | `SM-LATE-ESCAPE1-NATURAL-SRC-01`, then `SM-LATE-ESCAPE1-EDGE-01`. |
| `0xDEA7` Escape 3 — **partial** | Same P4 forced-event/timer fixture; settle only. | Requires the natural Escape 1 source and real preceding escape edges with a live timer. | `SM-LATE-ESCAPE1-NATURAL-SRC-01`, then `SM-LATE-ESCAPE1-EDGE-01`. |
| `0xDEDE` Escape 4 — **partial** | Same P4 forced-event/timer fixture; P5 leaves by absolute door. | Requires a natural timer-bearing chain through Escape 1–3 and a real Escape 4 departure. | `SM-LATE-ESCAPE1-NATURAL-SRC-01`, then `SM-LATE-ESCAPE1-EDGE-01`; follow with `SM-LATE-ESCAPE-RETURN-SRC-01`. |

## Boundary non-claims

- The bar does not change `STATUS.md`, the path board, the KPDR route, or any
  source-state provenance.
- `DEV` settles and scaffold unit tests are not natural entries or clears.
- Landing at `0x91F8` is not completion.  The final atom is
  `SM-LATE-SHIP-TRIGGER-01`, which must record the ship-entry ending/credits
  transition from a verified natural escape source.
