# Harvest Moon

Scriptable full-game Harvest Moon (SNES): a reactive skill API that plays the
whole game from power-on, Clean, then a 10–20 hour YouTube through the end of
credits with a score on the board. First credits is a basic path. **Gut** the
tree until a **Skill** can be A/B’d without rewriting the harness; **Speed**
runs through that loop.

## Language

**Finish**:
The published YouTube of a natural power-on playthrough that sits through the
**end** of credits and shows **Score**. Not the first credits tape. Not ranch
master.
_Avoid_: first credits frame as done, ranch-master 999, Gate B, summer entry,
crop-loop closed, leftover quotas, date-patched credits, speedrun input movies

**First run**:
An intermediate natural playthrough through credits with a score, basics only.
**Gut** may run before or beside it. It is not Finish. Built as **Skills**,
not a one-off tape.
_Avoid_: treating first credits as the YouTube, ranch-master route as the first
path, freezing a TAS movie as the route

**Gut**:
File/module structure until a **Skill** has a stable **A/B loop**. Merge into
the **Composer** (`DayPlanTask` / skill table) or delete. Soft max ~1000 LOC;
no sibling extract. Mixin clusters, `utils/` graveyards, and clone runners go
unless they *are* the loop.
_Avoid_: 13 `crop_*` files as a “split”, calling this **Speed**

**A/B loop**:
Load a pin, play two **Skills** or input sequences, compare RAM/video.
**Speed** must go through it. **Gut** must not rewrite it.
_Avoid_: a new CrossMap tape as the compare tool, a new probe CLI per shop

**Speed**:
Button-press / policy work through the **A/B loop**. May run in the same week
as **Gut**. Must not mint a runner or rewrite the loop. Ranch master / 999 is
**Speed**, not a first-credits fail line.
_Avoid_: “then rewrite” as a second tree, baking D2 as an unparameterized script
to shave minutes before the loop is stable

**Published video**:
A 10–20 hour YouTube of the entire game (split into parts as needed). Emulator
frames at 60fps playback; live watch speed is a viewer knob.
_Avoid_: highlight montage as the completion artifact, wall-clock of a turbo
watch as the runtime

**Score**:
Whatever number the credits evaluation shows on a natural run. First path only
needs a score to exist. Ranch master / 999 is later **Speed**, not required.
_Avoid_: ranch master as a fail line, 999, requiring the ranch-master branch

**Clean**:
No RAM writes on Harvest tapes and rungs. Resource pokes were a past speed
trick; we are past them. Native ROM edges stay; we do not write around them.
_Avoid_: money poke, date poke, stamina poke, LiveRamEditor as arrival

**Skill**:
A composable, rewritable controller with a stable API (navigate, interact,
verify, sequence). The campaign is a skillset, not a fragile button string.
_Avoid_: speedrun tape, CrossMap movie as the high-level route, one-off mashed A

**Steer**:
User input changes the current goal so the bot is not stuck in a hardcoded rut.
Microphone voice is a 2027 stretch (needs a much faster model). Typed or session
input is the seam to leave now.
_Avoid_: wiring STT before Finish, baking D2 as an unparameterized script

**Natural campaign**:
Progress from power-on (or the real predecessor that power-on produced). Calendar
and ending bytes are not skipped with RAM to claim a later rung.
_Avoid_: ending_probe showcase states as route evidence, `set_calendar_date` as
arrival

**Scratch ending**:
Existing Year 3 / credits probes and patched showcase saves. Sequence research
only. Re-work from natural arrival when the campaign gets there.
_Avoid_: treating those saves as the route, “already have the ending”

**Harvest rungs**:
The working board, in order: D2 farm clear → first potato harvest → Spring to
Summer → animals → Y1 done → marriage → Y3 credits → published video.
_Avoid_: Gate A, Gate B, Gate C, M3.1, leftover quotas as rungs

**D2 farm clear**:
Spring Day 2: every weed, stone, fence, large rock, and stump is gone; potatoes
are planted and watered; goods are in the shipping bin before 17:00. After
hour 18 the in-game clock’s hour does not advance; work continues until the farm
is clear. All fences includes the house-row posts. A “six hour” tape is a guess,
not a fail clock.
_Avoid_: leftover quotas (10 / 10 / 4 / 2), the 19 boxed posts as an exception,
pocket CLEAR_PLOT as the day, pasture grass as the wipe target, forcing sleep to
end D2

**Weed**:
A farm bush (`0x03`) that must leave the farm on D2 farm clear.
_Avoid_: grass, pasture, grass seed

**Grass**:
Pasture / feed grass. Not the D2 wipe target.
_Avoid_: weed, bush

**Grass seed**:
The free bag on the shed shelf. An item, not a farm tile. May need a shed pickup
so it is not left as clutter.
_Avoid_: grass, weed

**Program gate**:
An M0–M8 label for the game matrix. Not the Harvest working board.
_Avoid_: using M-gates as the session ladder, inventing Harvest-private maturity
letters
