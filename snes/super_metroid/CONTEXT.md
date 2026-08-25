# Super Metroid

Scriptable vanilla Super Metroid (SNES): a skill API that plays any% KPDR from
power-on through the end of credits. First credits is Survival, noob loadout,
and a two-hour class is fine. Then rewrite toward speed. 100% is a later year.
Solver/SMZ3 is downstream.

## Language

**Finish**:
A natural power-on playthrough that sits through the **end** of credits and
shows an **RTA**. Any% KPDR, noob loadout, Survival. A two-hour class is fine.
Not 100%. Not WR. Not hop-replay from a late pin.
_Avoid_: STATUS JSON as the completion artifact, 100% as the fail line, TAS
movie playback as the run, credits from a Gravity/G4 pin, sub-hour as this
year’s fail line

**First run**:
Power-on through credits with an RTA, noob loadout, Survival. A two-hour class
is fine. Then refactor and rewrite. It is not the speed rewrite.
_Avoid_: sub-hour as the first-credits fail line, WR, starting credits from a
late pin

**RTA**:
Whatever time the power-on run shows on screen. First path only needs a time
to exist. Sub-hour is a later rewrite.
_Avoid_: WR as a fail line, requiring sub-hour before first credits

**Survival**:
First-pass intervention: refill current energy (after Ceres) and currently
unlocked ammo, up to natural capacity. No item, capacity, boss, door, pose, or
timer writes. Harder than Harvest is why this stays; it is not a skip.
_Avoid_: item poke, boss flags, door warp as arrival, Gravity bit, G4 flags

**Clean**:
Zero energy and ammo writes. Parallel track. Not the first-credits fail line.
_Avoid_: claiming Clean greens as the program tip, disabling Survival on the
default continuous CLI

**Tip**:
The one living power-on product. Today that is **Phantoon**. Ice, Wave, Speed,
and earlier stems are prefix CI, not the board. Other work is scratch, parallel,
or getting ahead. Rung green is power-on to this tip.
_Avoid_: Ice as the living tip, a zoo of equally published tips, pin-bench as
the tip, practice greens as the tip

**Scratch**:
Power-on or pin duals that may lead the tip. Not the living tip. Not Finish.
_Avoid_: treating get-ahead as rung green, a second published tip beside
Phantoon

**Skill**:
A composable, rewritable controller with a stable API. The campaign is a
skillset. A **Tape** is only a guideline until a skill exists.
_Avoid_: multi-minute open-loop movie as the route, TAS concat, timing stitch
as playback

**Tape**:
A human or TAS recording used as a guideline to build a **Skill**. Once
processed, it is not source of truth and may be trashed.
_Avoid_: keeping the mega-tape as gold, full-tape open-loop as pin recovery,
TAS concat onto the tip

**Noob route**:
First path keeps convenience majors already on the late tapes (Grapple, Plasma,
Screw as taken). They are easier to tune than every walljump and grappleless
line. They are not rungs. Cut on the rewrite.
_Avoid_: grappleless / WJ-perfect first path, treating those majors as board
rows, Golden Torizo / all tanks / maps as first-path rungs

**Slop**:
Rooms already on a green power-on tip that still need a skill rewrite. Green
tip is not “those rooms are done.”
_Avoid_: freezing the Phantoon prefix because the dual is green, shipping a
leave that the next skill cannot **Sync**

**Chip**:
One room at a time: TAS or human **Tape** as guideline → **Skill** on the tip.
Not TAS concat. Not a geographic freeze except the serial NOW (Gravity).
_Avoid_: movie splice as the runner, waiting on a pretty Ice spine before
Gravity

**Sync**:
The leave must tie cleanly to the next room. A doorway pause and a few lost
frames are allowed. If the seam will not join, both rooms are one change.
Re-pin the next hop. If Red Tower drops twenty fake jumps and Hellway cannot
be joined, the rewrite did not land.
_Avoid_: hope the tail survives, frame-append across a new leave, full-tape
open-loop to invent the next pin, treating a few door frames as a fail

**Milestone dual**:
Power-on dual to the living tip at a major milestone (Gravity, a new living
tip, credits) and before **Publish**. Not after every Chip.
_Avoid_: 54-minute dual on every slop hop, skipping the dual when the living
tip itself moved

**Publish**:
A watchable power-on tape to the living tip after 20–30 working sessions with
material progress. Not a calendar week. Not STATUS. Material = a dual-green
bead on the living tip or the next compose.
_Avoid_: a weekly clock with no progress, highlight cuts, pin benches as the
upload, waiting on a STATUS promote to publish, counting Harvest or Clean or
RED windows

**Any% KPDR**:
The first-path route: Kraid, Phantoon, Draygon, Ridley, then Tourian and
credits, on the **Noob route** loadout.
_Avoid_: 100%, PRKD, ship-first

**100%**:
A later-year rewrite. Tanks, packs, maps, Golden Torizo are not first path.
_Avoid_: 100% as this year’s Finish

**Program gate**:
An M0–M8 label for the game matrix. Not the Super Metroid working board.
_Avoid_: using M-gates as the session ladder
