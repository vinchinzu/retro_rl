# L7-L9 route continuation and review — 2026-09-03

Branch: `main` at `67fd793b`. No branch was created. This note does not
promote `STATUS.md`; all evidence below remains Survival/fixture development
evidence with `natural_entry=false` and `route_eligible=false` unless stated
otherwise in the owning handoff.

## Dispatch status

The requested Luna continuation, Sol review, and Sol spec-review child all
failed before completing a turn because the account usage limit was reached.
No result should be inferred from those failed dispatches. The existing L9
`x=120` edit is a prediction in `67fd793b`, not a new live green.

## Verified boundaries

- L8: the guarded fixture replay confirmed `0x5E -> 0x4E` at `(120,205)`.
  No combat or resource use occurred in `0x4E`; deaths and forbidden writes
  were zero. Next: key-N toward predicted `0x3E`, stopping at the first
  settled room. See `l8-handoff.md`.
- L9: three full fixture replays stopped at the required session limit. The
  latest live boundary is OW `0x38 (48,133)`, tile `0x95`; `UP` at that pose
  is blocked. The left Spectacle Rock bomb and expected room `0x76` were not
  exercised. See `l9-handoff.md`.
- L7 exit area: natural `DOWN` from L7 room `0x79` reached OW `0x42
  (112,93)`. The pond refilled and straight `DOWN` collided with tile `0x8F`.
  No later screen was attempted. Bead `rr-6o7.4` remains in progress.

## Review findings

### Standards

1. `level8/overworld.py` is now 918 lines, below the soft 1,000-line limit but
   close to it. The fixture-only `Level7PondToLevel8BushController` should be
   folded into the existing Zelda `SpineHop`/Composer seam if it becomes a
   durable route skill; do not add another sibling dispatcher.
2. The new `scratch/probe_l7_exit_to_l8_bush.py` and existing L8/L9 probe
   runners are evidence tools, not alternate public route CLIs. Delete or fold
   them after their policies move into the owning Composer, per
   `CODING_STANDARDS.md` (Composer/Delete rules).
3. Controller reports in `level8/overworld.py` and `level9/overworld.py`
   contain constant zero write counters. These are declarations, not audits.
   Only the runner's measured assist/AuditedEnv telemetry may support a
   zero-write claim.

### Spec

1. Commit `67fd793b` changes the L9 `0x38` north alignment from `x=48` to
   `x=120`, and its unit test proves only which input the controller emits.
   No new emulator report or handoff result proves `0x38 -> 0x28`. Keep the
   policy labeled hypothesis until one screenshot/RAM-graded trial settles
   `0x28`.
2. `rr-6o7.4` correctly stops at the first pond miss and does not satisfy the
   natural L7-to-L8 seam. The true post-L7 fanfare leftover is still
   unmeasured, so `MEASURED_POST_L7_EXIT` must remain false.
3. L8 `0x5E -> 0x4E` satisfies only fixture recon. It does not satisfy
   `rr-6o7.1`, whose acceptance remains a power-on entry from the measured L7
   leave.

## Bug reassessment

- L9 is a controller-policy regression/unknown geometry, not evidence that
  the earlier checkpoint route was false. The fast red-capable seam is the
  full fixture replay: it must fail unless RAM settles on `0x28`. The current
  unit test cannot catch a wrong mouth coordinate.
- The L7 refilled pond is presently expected emulator state for this fixture,
  not a confirmed controller defect. The route assumption was wrong: an
  interior exit pose at the north shore is not interchangeable with the
  disclosed `OW_L7Pond (128,221)` south-side pose or a measured post-fanfare
  leave.
- No forbidden-write or false route-eligibility success was found in the
  focused tests. The main remaining risk is provenance: do not treat constant
  report fields or policy-only unit tests as live audit evidence.

## Next actions

1. L9: inspect the v3 final PNG, predict `x=120` explicitly, run exactly one
   full fixture replay, and stop at the first miss or settled `0x28`.
2. L7-to-L8: start from `OW_L7Pond (128,221)` and grade only `DOWN -> 0x52`,
   or occupancy-plan around/re-drain the refilled pond before acting.
3. L8 interior: replay through `0x4E`, attempt only the key-N boundary to
   predicted `0x3E`, then census the first settled destination.

Do not STATUS-promote, close the natural-entry beads, or claim Clean results
from any of these fixture starts.

## Update — 2026-09-03 (L8 lane, next-actions #2 and #3)

- **#2 L7→L8**: `Level7PondToLevel8BushController` rebuilt from the live
  forward-pond point lists (reversed) — `OW_L7Pond` south-shore start now
  reaches **OW 0x6D 2/2 byte-identical** (frames 4980, `deaths=0`, forbidden
  writes 0). Refilled-pond natural `Level7Entrance` start fails closed
  (reachability proves Link cannot skirt the pool). Still
  `route_eligible=false`, `MEASURED_POST_L7_EXIT` stays false. See
  `l8-handoff.md`.
- **#3 L8 interior**: replayed through `0x4E`, took the north **key** door
  (not a shutter) — **2/2**: keys 10→9 (one natural spend), settled `0x3E`
  `(120,205)`, census 6× blue Darknut `0x0C`, item `0x03`. Recon-wired as
  `LEVEL8_INTERIOR_0X3E_RECON` (not a `DungeonRoomSpec`, not on `L8_THROUGH`);
  `Level8Interior3EReconFixture` saved. See `l8-handoff.md`.
- L9 (#1) untouched by this lane.
