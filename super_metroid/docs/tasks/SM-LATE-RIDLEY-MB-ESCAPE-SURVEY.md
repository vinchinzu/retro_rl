# SM-LATE-RIDLEY-MB-ESCAPE-SURVEY — late-route diagnostic survey

## Result

**Development-only topology observed.** Five approved `route.py leg` probes
completed **49/49 hops** and entered **48 distinct room IDs** from post-Ridley
through the Landing Site return. This is not natural route progress: every
probe used absolute door warps plus the route fixture's granted late loadout;
the Mother Brain → escape leg additionally applied its development boss/event
writes. No controller, boss, timer, ship-trigger, or credits claim follows
from these results.

This card is the only tracked survey output. No source states, route code,
board, registry, `continuous.py`, `progression.py`, or `STATUS.md` were
changed.

## Read-first evidence

- `AGENTS.md` and `super_metroid/AGENTS.md`
- `docs/SOURCE_STATES.md` — both prescribed anchors are topology-only;
  `dev_route_*` uses granted full loadout and boss bits.
- `docs/routes/ROUTE_KPDR.md` K9 — status is open; only MB entry/spray probes
  existed before this survey.
- `docs/research/PATH_ROOM_BOARD.md` W7/W8 and room IDs 85–106.
- `docs/BOSS_PIPELINE.md` — boss work is deferred until a natural entry exists.
- Existing cards: `SM-RIDLEY-01.md`, `SM-MB-01.md`, and `SM-ESCAPE-01.md`.

## Probe contract and commands

`super_metroid/scripts/probe/route.py leg` boots the named dev state, grants
the route fixture loadout, marks major bosses for topology, resolves the listed
bank-$83 door pointer, and asserts that the arrived room is the expected ID.
The command reports `developmentOnly: true`; a successful `gameState: 8` is a
room-load observation only.

| ID | Actual command | Source fixture | Result |
|---|---|---|---|
| P1 | `uv run python super_metroid/scripts/probe/route.py leg ridley statues --source-state super_metroid/custom_integrations/SuperMetroid-Snes/dev_route_anchor_ridley.state` | `dev_route_anchor_ridley.state` (`0xB32E`) | 30/30 hops, final `0xA66A` |
| P2 | `uv run python super_metroid/scripts/probe/route.py leg statues tourian_elevator --source-state super_metroid/custom_integrations/SuperMetroid-Snes/dev_route_anchor_ridley.state` | Ridley anchor boot; absolute `0x9222` fixture door (not a physical Statues departure) | 1/1, final `0xDAAE` |
| P3 | `uv run python super_metroid/scripts/probe/route.py leg tourian_elevator mother_brain --source-state super_metroid/custom_integrations/SuperMetroid-Snes/dev_route_anchor_ridley.state` | Ridley anchor boot; Tourian absolute-door fixture | 11/11, final `0xDD58` |
| P4 | `uv run python super_metroid/scripts/probe/route.py leg mother_brain tourian_escape_4 --source-state super_metroid/custom_integrations/SuperMetroid-Snes/dev_route_anchor_mother_brain.state` | `dev_route_anchor_mother_brain.state` (`0xDD58`) | 4/4, final `0xDEDE`; the leg's MB-source fixture writes the MB/Tourian unlocks and escape event before warping |
| P5 | `uv run python super_metroid/scripts/probe/route.py leg tourian_escape_4 landing_site_finish --source-state super_metroid/custom_integrations/SuperMetroid-Snes/dev_route_anchor_mother_brain.state` | MB anchor boot; absolute escape fixture doors | 3/3, final `0x91F8` |

Scaffold check (no emulator state mutation):

```bash
uv run pytest super_metroid/tests/test_ridley_combat.py super_metroid/tests/test_mother_brain_combat.py super_metroid/tests/test_escape_scaffold.py -q
# 17 passed in 0.17s
```

## Per-room survey

Status vocabulary:

- **observed** — the indicated dev topology probe reached and asserted this
  room; it is not a playable-policy result.
- **partial** — observed plus the named dev-only combat/escape scaffold has
  unit-test coverage. It still lacks natural-entry evidence.
- **blocked** is reserved for a room that the diagnostic could not enter. No
  room was diagnostic-blocked in this pass; all route-real readiness is still
  blocked by the fixture/capability stated in the final two columns.

Pins are the final `samusX,samusY` emitted by the successful hop. `—` means
the room was the booted source fixture, whose load pin was not emitted by the
CLI, not that its room ID was unverified.

| Room | Name | Status | Source / entry evidence and final pin | Actual probe | Capability or fixture blocker | Next atomic card |
|---|---|---|---|---|---|---|
| `0xB32E` | Ridley's Room | partial | P1 booted prescribed `dev_route_anchor_ridley.state`; source catalog asserts `0xB32E`; pin — | P1 | No route-real Lower Norfair predecessor or natural fight/exit; `combat/ridley.py` is dev-only. | `SM-LATE-RIDLEY-NATURAL-ENTRY-01` |
| `0xB37A` | Lower Norfair Farming Room | observed | P1 `0xB32E --0x98BE→ 0xB37A`; got `0xB37A`, `(120,210)` | P1 | Fixture exit after a skipped/marked Ridley; no natural post-fight handoff. | `SM-LATE-RIDLEY-EXIT-01` |
| `0xB482` | Plowerhouse Room | observed | P1 `0xB37A --0x98D6→`; got, `(120,210)` | P1 | No natural predecessor state or movement policy for this return direction. | `SM-LATE-RIDLEY-RETURN-EDGE-01` |
| `0xB62B` | Metal Pirates Room | observed | P1 `0xB482 --0x9966→`; got, `(120,187)` | P1 | Same return-chain source/policy gap. | `SM-LATE-RIDLEY-RETURN-EDGE-01` |
| `0xB5D5` | Wasteland | observed | P1 `0xB62B --0x9A3E→`; got, `(175,699)` | P1 | Same return-chain source/policy gap. | `SM-LATE-RIDLEY-RETURN-EDGE-01` |
| `0xB585` | Red Kihunter Shaft | observed | P1 `0xB5D5 --0x9A26→`; got, `(687,1350)` | P1 | Same return-chain source/policy gap. | `SM-LATE-RIDLEY-RETURN-EDGE-01` |
| `0xB6EE` | Lower Norfair Fireflea Room | observed | P1 `0xB585 --0x9A02→`; got, `(216,838)` | P1 | Same return-chain source/policy gap. | `SM-LATE-RIDLEY-RETURN-EDGE-01` |
| `0xB510` | Lower Norfair Springball Maze Room | observed | P1 `0xB6EE --0x9A92→`; got, `(672,326)` | P1 | Same return-chain source/policy gap. | `SM-LATE-RIDLEY-RETURN-EDGE-01` |
| `0xB656` | The Musketeers' Room | observed | P1 `0xB510 --0x99AE→`; got, `(1128,582)` | P1 | Same return-chain source/policy gap. | `SM-LATE-RIDLEY-RETURN-EDGE-01` |
| `0xAD5E` | Single Chamber | observed | P1 `0xB656 --0x9A4A→`; got, `(1584,70)` | P1 | Same return-chain source/policy gap. | `SM-LATE-RIDLEY-RETURN-EDGE-01` |
| `0xACB3` | Bubble Mountain | observed | P1 `0xAD5E --0x95CA→`; got, `(504,326)` | P1 | Same return-chain source/policy gap. | `SM-LATE-RIDLEY-RETURN-EDGE-01` |
| `0xAFA3` | Rising Tide | observed | P1 `0xACB3 --0x955E→`; got, `(1472,70)` | P1 | Same return-chain source/policy gap. | `SM-LATE-RIDLEY-RETURN-EDGE-01` |
| `0xA788` | Cathedral | observed | P1 `0xAFA3 --0x9732→`; got, `(904,326)` | P1 | Same return-chain source/policy gap. | `SM-LATE-RIDLEY-RETURN-EDGE-01` |
| `0xA7B3` | Cathedral Entrance | observed | P1 `0xA788 --0x928E→`; got, `(848,70)` | P1 | Same return-chain source/policy gap. | `SM-LATE-RIDLEY-RETURN-EDGE-01` |
| `0xA7DE` | Business Center | observed | P1 `0xA7B3 --0x92A6→`; got, `(295,838)` | P1 | Existing early Business evidence is not a post-Ridley natural handoff. | `SM-LATE-RIDLEY-RETURN-EDGE-01` |
| `0xA6A1` | Warehouse Entrance | observed | P1 `0xA7DE --0x92EE→`; got, `(128,291)` | P1 | Existing early Warehouse evidence is not a post-Ridley natural handoff. | `SM-LATE-RIDLEY-RETURN-EDGE-01` |
| `0xCF80` | East Tunnel | observed | P1 `0xA6A1 --0x922E→`; got, `(328,271)` | P1 | Same return-chain source/policy gap. | `SM-LATE-RIDLEY-RETURN-EDGE-01` |
| `0xCEFB` | Glass Tunnel | observed | P1 `0xCF80 --0xA378→`; got, `(295,271)` | P1 | Same return-chain source/policy gap. | `SM-LATE-RIDLEY-RETURN-EDGE-01` |
| `0xCFC9` | Main Street | observed | P1 `0xCEFB --0xA330→`; got, `(295,1946)` | P1 | Same return-chain source/policy gap. | `SM-LATE-RIDLEY-RETURN-EDGE-01` |
| `0xD0B9` | Mt. Everest | observed | P1 `0xCFC9 --0xA3CC→`; got, `(95,666)` | P1 | Same return-chain source/policy gap. | `SM-LATE-RIDLEY-RETURN-EDGE-01` |
| `0xD104` | Red Fish Room | observed | P1 `0xD0B9 --0xA42C→`; got, `(607,563)` | P1 | Same return-chain source/policy gap. | `SM-LATE-RIDLEY-RETURN-EDGE-01` |
| `0xA322` | Caterpillar Room | observed | P1 `0xD104 --0xA480→`; got, `(808,819)` | P1 | Same return-chain source/policy gap. | `SM-LATE-RIDLEY-RETURN-EDGE-01` |
| `0x962A` | Elevator To Caterpillar | observed | P1 `0xA322 --0x90BA→`; got, `(128,291)` | P1 | Same return-chain source/policy gap. | `SM-LATE-RIDLEY-RETURN-EDGE-01` |
| `0x948C` | Crateria Kihunter Room | observed | P1 `0x962A --0x8AF6→`; got, `(384,680)` | P1 | Same return-chain source/policy gap. | `SM-LATE-RIDLEY-RETURN-EDGE-01` |
| `0x95D4` | Crateria Tube | observed | P1 `0x948C --0x8A2A→`; got, `(328,168)` | P1 | Same return-chain source/policy gap. | `SM-LATE-RIDLEY-RETURN-EDGE-01` |
| `0x91F8` | Landing Site | observed | P1 `0x95D4 --0x8AC6→`; got `(2343,1192)`. P5 final escape arrival `(500,300)`. | P1, P5 | No timer-bearing natural escape predecessor and no ship/credits predicate probe. | `SM-LATE-SHIP-TRIGGER-01` |
| `0x92FD` | Parlor and Alcatraz | observed | P1 got `(1264,168)`; P5 escape-direction arrival got `(376,1357)`. | P1, P5 | No timer-bearing natural escape predecessor or controller. | `SM-LATE-ESCAPE-RETURN-SRC-01` |
| `0x990D` | Terminator Room | observed | P1 `0x92FD --0x895E→`; got, `(1720,168)` | P1 | Existing early traversal is not a post-Ridley natural handoff. | `SM-LATE-RIDLEY-RETURN-EDGE-01` |
| `0x99BD` | Green Pirates Shaft | observed | P1 `0x990D --0x8BE6→`; got, `(384,1192)` | P1 | Existing early traversal is not a post-Ridley natural handoff. | `SM-LATE-RIDLEY-RETURN-EDGE-01` |
| `0xA5ED` | Statues Hallway | observed | P1 `0x99BD --0x8C52→`; got, `(120,180)` | P1 | Four-boss / statue state was fixture-written; no natural G4 evidence. | `SM-LATE-G4-OPEN-01` |
| `0xA66A` | Statues Room | observed | P1 `0xA5ED --0x91F2→`; got, `(128,180)` | P1 | Four-boss / statue state was fixture-written; no natural G4 evidence. | `SM-LATE-G4-OPEN-01` |
| `0xDAAE` | Tourian Elevator | observed | P2 `0xA66A --0x9222→`; got, `(128,44)` | P2 | P2 booted Ridley anchor then used an absolute door pointer; no physical G4 departure or natural Tourian source. | `SM-LATE-TOURIAN-SRC-01` |
| `0xDAE1` | Metroid Room 1 | observed | P3 `0xDAAE --0xA984→`; got, `(1576,50)` | P3 | No natural Tourian elevator source or Metroid-room controller/capability evidence. | `SM-LATE-TOURIAN-EDGE-01` |
| `0xDB31` | Metroid Room 2 | observed | P3 `0xDAE1 --0xA9B4→`; got, `(240,50)` | P3 | Same Tourian source/controller gap. | `SM-LATE-TOURIAN-EDGE-01` |
| `0xDB7D` | Metroid Room 3 | observed | P3 `0xDB31 --0xA9CC→`; got, `(39,50)` | P3 | Same Tourian source/controller gap. | `SM-LATE-TOURIAN-EDGE-01` |
| `0xDBCD` | Metroid Room 4 | observed | P3 `0xDB7D --0xA9E4→`; got, `(120,180)` | P3 | Same Tourian source/controller gap. | `SM-LATE-TOURIAN-EDGE-01` |
| `0xDC19` | Tourian Hopper Room | observed | P3 `0xDBCD --0xA9FC→`; got, `(376,40)` | P3 | Same Tourian source/controller gap. | `SM-LATE-TOURIAN-EDGE-01` |
| `0xDC65` | Dust Torizo Room | observed | P3 `0xDC19 --0xAA14→`; got, `(576,40)` | P3 | Same Tourian source/controller gap. | `SM-LATE-TOURIAN-EDGE-01` |
| `0xDCB1` | Big Boy Room | observed | P3 `0xDC65 --0xAA2C→`; got, `(1032,40)` | P3 | Same Tourian source/controller gap. | `SM-LATE-TOURIAN-EDGE-01` |
| `0xDCFF` | Seaweed Room | observed | P3 `0xDCB1 --0xAA44→`; got, `(208,40)` | P3 | Same Tourian source/controller gap. | `SM-LATE-TOURIAN-EDGE-01` |
| `0xDDC4` | Tourian Eye Door Room | observed | P3 `0xDCFF --0xAA5C→`; got, `(7,40)` | P3 | Same Tourian source/controller gap. | `SM-LATE-TOURIAN-EDGE-01` |
| `0xDDF3` | Rinka Shaft | observed | P3 `0xDDC4 --0xAAA4→`; got, `(120,210)` | P3 | Same Tourian source/controller gap; no natural MB doorway activation. | `SM-LATE-MB-NATURAL-ENTRY-01` |
| `0xDD58` | Mother Brain's Room | partial | P3 `0xDDF3 --0xAAC8→`; got, `(1088,210)`. Prescribed MB anchor also asserts room. | P3; 17-test scaffold check | `combat/mother_brain.py` is development-only; no natural Rinka arrival, rainbow/hyper handling, natural defeat event, or exit proof. | `SM-LATE-MB-ACTIVATION-01` |
| `0xDE4D` | Tourian Escape Room 1 | partial | P4 `0xDD58 --0xAA8C→`; got, `(520,210)` | P4; 17-test scaffold check | P4 wrote MB/Tourian events and armed the escape fixture; `combat/escape.py` is stub-only. | `SM-LATE-ESCAPE1-NATURAL-SRC-01` |
| `0xDE7A` | Tourian Escape Room 2 | partial | P4 `0xDE4D --0xAAEC→`; got, `(8,70)` | P4; 17-test scaffold check | Same forced-event/timer fixture; no room movement controller. | `SM-LATE-ESCAPE1-EDGE-01` |
| `0xDEA7` | Tourian Escape Room 3 | partial | P4 `0xDE7A --0xAB04→`; got, `(120,210)` | P4; 17-test scaffold check | Same forced-event/timer fixture; no room movement controller. | `SM-LATE-ESCAPE1-EDGE-01` |
| `0xDEDE` | Tourian Escape Room 4 | partial | P4 `0xDEA7 --0xAB1C→`; got, `(200,180)` | P4; P5 source leg | Same forced-event/timer fixture; P5 then applies an absolute exit door rather than play. | `SM-LATE-ESCAPE1-EDGE-01` |
| `0x96BA` | The Climb | observed | P5 `0xDEDE --0xAB34→`; got, `(120,180)` | P5 | No natural timer-bearing Escape 4 exit state or escape-direction controller. | `SM-LATE-ESCAPE-RETURN-SRC-01` |

## Atomic next-card definitions

The card labels in the table are deliberately narrow; none authorizes a broad
controller rewrite or a continuous/STATUS update.

| Card | One atomic objective | Precondition / acceptance |
|---|---|---|
| `SM-LATE-RIDLEY-NATURAL-ENTRY-01` | Capture and validate one settled, unmodified doorway-natural Ridley entry. | Requires an actual Lower Norfair predecessor; record room, pose, x/y, inventory, and zero forbidden write deltas. Planner gate until then. |
| `SM-LATE-RIDLEY-EXIT-01` | From that validated entry only, prove natural Ridley defeat and one post-fight exit into `0xB37A`. | Assert natural boss/event transition; dump the resulting B37A handoff; no placement, warp, or boss/event write. |
| `SM-LATE-RIDLEY-RETURN-EDGE-01` | From the B37A handoff, exercise only the first physical B37A→B482 edge and record its exit pin. | Do not extend into a multi-room return controller; a green handoff becomes the next room's source. |
| `SM-LATE-G4-OPEN-01` | From the real predecessor, observe the four-statue sequence and arrive at settled G4/Statues room without writing boss bits. | Requires all four natural boss outcomes; prove the statue condition from RAM observations only. |
| `SM-LATE-TOURIAN-SRC-01` | Capture one doorway-natural Tourian Elevator state immediately after the real G4 transition. | Validate `0xDAAE`, inventory, and no writes; do not cross a Tourian room. |
| `SM-LATE-TOURIAN-EDGE-01` | From that exact Tourian Elevator source, play only the elevator→Metroid Room 1 boundary and emit a final pin. | Bounded single-room/door probe; no full Tourian controller. |
| `SM-LATE-MB-NATURAL-ENTRY-01` | Capture an active, natural Rinka Shaft→MB doorway state. | Requires preceding Tourian edges; prove room, enemy activation, and zero forged boss/event bits. |
| `SM-LATE-MB-ACTIVATION-01` | From that source, characterize activation/phase transition only; stop before any defeat attempt. | No rainbow/hyper controller or escape event write; output evidence solely for the existing MB strategy. |
| `SM-LATE-ESCAPE1-NATURAL-SRC-01` | Capture Escape 1 immediately after a naturally observed MB defeat/door transition. | Verify timer and events arose from gameplay, not `start_escape_timer` or `mark_mother_brain_defeated`. |
| `SM-LATE-ESCAPE1-EDGE-01` | From that exact Escape 1 source, play only Escape 1 to its real next-room boundary. | Preserve natural timer; report timer remaining, final room/pin, and no writes. |
| `SM-LATE-ESCAPE-RETURN-SRC-01` | Capture the settled Climb-side source after a real Escape 4 departure. | Preserve timer provenance; do not attempt Parlor or ship entry. |
| `SM-LATE-SHIP-TRIGGER-01` | From a verified Landing Site escape source, test only the ship-entry/ending predicate. | Requires the natural return chain; record ending/credits transition rather than treating landing in `0x91F8` as completion. |

## Blockers and non-claims

1. The only late sources used here are explicitly development anchors. They
   carry a granted late inventory and major-boss state; they cannot establish
   route-ready entries.
2. P2, P3, and P5 booted a convenient dev state before applying absolute door
   pointers. Their successful expected-room checks validate map topology, not
   the source room's physical exit.
3. P4 is especially non-natural: its MB source processing marks the Tourian
   boss/event unlocks before the escape door chain. The escape timer and
   defeat state were never earned by the probe.
4. The existing Ridley and Mother Brain strategy modules and escape scaffold
   are testable shells only. `17 passed` confirms import/contract coverage,
   not fights, escape play, a ship trigger, ending, or credits.
5. No continuous, `STATUS`, GREEN, natural-entry, boss-bit, event, item,
   timer, ship, or credits claim is made by this survey.

