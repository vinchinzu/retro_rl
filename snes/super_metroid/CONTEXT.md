# Super Metroid

Scriptable vanilla Super Metroid (SNES): a skill API that plays any% KPDR from
power-on through the end of credits. First credits is Survival, noob loadout,
and a two-hour class is fine. **Gut** the tree until a **Skill** can be A/B’d
without rewriting the harness; **Speed** (RTA / button-press) runs through that
loop and is not a fail line until then. Both may run in the same week. 100% is
a later year. Solver/SMZ3 is downstream.

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
is fine. **Gut** may run before or beside it. It is not **Speed**.
_Avoid_: sub-hour as the first-credits fail line, WR, starting credits from a
late pin

**Gut**:
File/module structure until a **Skill** has a stable **A/B loop**. Merge into
the **Composer** (`tips.play_hops` / `TipSpec`) or delete. Soft max ~1000 LOC;
no sibling extract. Probe CLIs, clone runners, and leftover packages go unless
they *are* the loop.
_Avoid_: minting `start_to_*.py`, a 13-file extract, calling this **Speed**

**A/B loop**:
Load a pin, play two **Skills** or **Tapes**, compare RAM/video. **Speed**
must go through it. **Gut** must not rewrite it.
_Avoid_: a new probe CLI as the compare tool, TAS concat as the compare

**Speed**:
Button-press / RTA work through the **A/B loop**. May run in the same week as
**Gut**. Must not mint a runner or rewrite the loop. Sub-hour is **Speed**,
not a first-credits fail line.
_Avoid_: sub-hour as Finish, a new timer script per room, “rewrite toward speed”
as a second tree

**RTA**:
Whatever time the power-on run shows on screen. First path only needs a time
to exist. Sub-hour is **Speed**, after the **A/B loop** is stable.
_Avoid_: WR as a fail line, requiring sub-hour before first credits, requiring
sub-hour before **Gut**

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
line. They are not rungs. Cut on **Speed**.
_Avoid_: grappleless / WJ-perfect first path, treating those majors as board
rows, Golden Torizo / all tanks / maps as first-path rungs

**Slop**:
Rooms already on a green power-on tip that still need a skill rewrite. Green
tip is not “those rooms are done.”
_Avoid_: freezing the Phantoon prefix because the dual is green, shipping a
leave that the next skill cannot **Sync**

**Sitting**:
One agent session on the living checkbox. It ends when that checkbox greens
or the human/context dies.
_Avoid_: three red windows as the sitting-end, treating a sitting as a hop restart

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

**Enemy**:
A currently active, non-boss room entity with its own health, position, and status.
_Avoid_: mob, “the Atomic” as the individual entity

**Species**:
The kind shared by Enemies, including maximum health, live and frozen Contact, and default Stance.
_Avoid_: using an Enemy instance as the kind

**Contact**:
What overlap does: knockback, solid, platform, none.
_Avoid_: collision as the domain word, keepaway as a Contact kind

**Stance**:
This-frame choice toward an Enemy: Engage (kill/freeze), Avoid (path around / wait), Absorb (take the hit; Survival refills), Ignore (out of band).
_Avoid_: keepaway, tank (comment, not Stance), run (that is Avoid)

**Overlay**:
The per-frame room-enemy Skill that cooperates with a movement hop without owning the room.
_Avoid_: BossSegment for room enemies, SpineHop per enemy

**Generalist**:
A goal-conditioned neural contractor that covers rooms the skill library does not own. Not a Skill. Not the Tip.
_Avoid_: treating a generalist door as a continuous green, STATUS from a net pin

**Contractor**:
The solver-layer job of synthesizing or covering missing L1. Existing skills stay the library.
_Avoid_: replacing SpineHop with a net, calling the generalist a Skill

**Steering**:
Injecting a Goal (next door, node, or repertoire session) without taking the controller.
_Avoid_: full human takeover as the only way to change destination

**Join**:
A `hop_glance` green against the next hop's LeaveSpec so a Skill can Sync. Room-change alone is not Join.
_Avoid_: morph-in-door as success, room-id as the handoff
