# Super Metroid Clean finish plan

Finish vanilla any% KPDR on stable-retro snes9x from power-on, through the
final “SEE YOU NEXT MISSION” screen. Keep Crocomire and Grapple on the human
route. The deliverables are the controller tape, an independent replay report
with zero loads, RAM writes and deaths, the final screen, and elapsed real
time from the input frames. Speed and 100% follow the first finish.
[Clean contract](CLEAN_TRACK.md) · [Verified leaves](STATUS.md)

## Start here next session

Resume at row 212 in Space Jump Room (`0xD9AA`), right after defeating
Draygon (row 210, `combat.draygon.play_clean_draygon_leave`) and collecting
Space Jump (row 211, `tape+idle:0`). Samus starts at `(85,155)` with 71/399
energy, 28/35 missiles, 1/5 Supers, 3/5 Power Bombs, items `0x7325`, beams
`0x1007`, and `$7E:D82C = 0x03`. [Clean evidence](CLEAN_TRACK.md)

Verify the current prefix with both the guarded chain replay and the
independent zero-harness audit before extending to Plasma Beam (row 223).

```bash
QT_QPA_PLATFORM=offscreen SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m super_metroid.splice chain --replay --through-row 211
QT_QPA_PLATFORM=offscreen SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python snes/super_metroid/scripts/tools/audit_master_tape.py
```

## End-to-end coverage

The Clean candidate board has 340 rows, including one boot metadata row,
the power-on sequence, 316 planned room crossings and 22 same-room rows
through Landing Site ship launch. Its 27 source tapes and two existing
crossing controllers connect every planned visit. The physical room graph
contains all 316 crossings. The route visits 168 of the 262 rooms in that
graph. The other 94 rooms each have a development entry in
`maps/room_problems.json`. An all-room run still needs an ordered route
through those rooms and recordings that connect them.

| Rows | Route section | Crossings | Open source-entry work |
|---|---|---:|---|
| 1–83 | Power-on to Business Center after Kraid | 76 | Verified from power-on. The actual replay includes 16 additional farming transitions in rows 41 and 46. |
| 84–134 | Upper Norfair to the Moat | 46 | Verified from power-on, including the Sloaters Energy Refill detour in row 127. |
| 135–148 | Moat, Wrecked Ship and Gravity | 13 | Verified from power-on, including Phantoon, the Attic enemy lock, and Gravity Suit. |
| 149–185 | Return to Norfair, Crocomire and Grapple | 36 | Verified from power-on, including the Maridia Tube break, Crocomire defeat and Grapple pickup. |
| 186–223 | Maridia, Botwoon, Draygon and Plasma | 35 | Verified through 211 (Botwoon, Draygon and Space Jump). Next is row 212 toward Plasma. |
| 224–272 | Lower Norfair, Screw Attack and Ridley | 46 | 251, 262 and 271. Resolve Spring Ball before 253. |
| 273–315 | Lower Norfair return, statues and Tourian entry | 43 | 282 |
| 316–339 | Tourian, Mother Brain, escape and ship launch | 21 | 319 retimed. Red rows 331 and 332. The credits tail follows 339. |

The Survival report independently covers 120 crossings through Phantoon.
Every one appears in the Clean board's physical links. The runs take
different sequences through those links, so resume Clean from its own master
state. The source-entry census remains a record of which archived candidates
sync from their source states. Cathedral, Rising Tide, Bubble Mountain, and
the Ice return needed different inputs on the Clean chain. Each remaining
candidate still needs to clear from its actual Clean predecessor.

## Route milestones

Rows below are inclusive IDs on the rebuilt Clean board. A row can be a
same-room pickup or a controller crossing. Each milestone uses the same
master tape. Promote it when an independent replay reaches its row **and**
shows the listed result. Check boss flags in bank 7E, collected item/beam
bits and actual ammo capacities at the appropriate split.

| Through row | Milestone | What the replay must show |
|---:|---|---|
| 44 | Brinstar Supers | Morph, Bombs, Bomb Torizo and Spore Spawn cleared, first five-Super capacity |
| 55 | Charge and Spazer | Big Pink Charge pickup, Super gate open, Spazer collected in A447 |
| 74 | Kraid and Varia | Hi-Jump and its tank, Kraid defeat flag, Varia collected in A6E2 |
| 130 | Upper Norfair and Power Bombs | Speed, Wave and Ice collected, first Power Bomb capacity in A3AE |
| 148 | Phantoon and Gravity | Cross the Moat and West Ocean, Phantoon flag set, Gravity collected in CE40 |
| 185 | Crocomire and Grapple | Crocomire defeated, Grapple collected in AC2B |
| 223 | Draygon and Plasma | Botwoon and Draygon defeated, Space Jump and Plasma collected |
| 272 | Ridley | Lower Norfair traversed with Screw Attack, Ridley defeated, Ridley Tank Room reached |
| 315 | Tourian access | All four major boss flags, natural statue opening, Tourian entrance and save-room return |
| 338 | Mother Brain and escape | Mother Brain defeated, escape rooms traversed before timeout, Landing Site reached |
| 339 + credits tail | Finish | Samus boards the ship. Replay all neutral tail inputs, verify the fully drawn final message, capture final screen and report elapsed time |

## Parallel work and integration (first dispatch, now closed)

| Owner during the first dispatch | Input | Work returned to the integrator |
|---|---|---|
| Frontier Sol | The last accepted master row | Solve consecutive leaves until the next failure. Save the best viable prefix and its replay, with entry/exit energy and ammo. This worker alone changes the canonical tape. |
| Kraid Sol | Repaired s10 doorway, then row 72's actual leave when available | Record buttons for the quick kill, Varia pickup and return. Report required ammo, first damage and the successor's settled position. |
| Credits Sol | Final Landing Site source, then the natural early-game farming state | Validate the credits detector and tail against the cartridge. On completion, search for a natural resource farm upstream of the Spazer gate. |
| Integrator | Clean board, source pins, independent replay | Grade room inputs on the same row IDs used by the solver. Test the two Moat/West Ocean crossings together. Update this handoff and STATUS only with the appropriate evidence. |

Independent workers may use an exact room-entry state, read RAM and restore
to search. Deliver SNES-12 buttons or a reusable room controller, source
identity, frame range, energy and ammo costs, and the next room. When the
master reaches that entry, try the candidate and run the next row. A different
loadout, input phase, or enemy RNG can change the result.

After Kraid, move the fight worker to Phantoon, then Botwoon/Draygon, then
Ridley. Move the terminal worker to the first resource shortage or uncovered
connection ahead of the master. A failed downstream room that needs ammo sends
work back to the nearest affordable pickup, enemy drop or refill station.
Develop alternative prefixes under separate artifact stems so the integrator
can compare the downstream result before replacing the accepted tape.

The 340-row source census graded 320 as recorded, one after a three-frame
delay (row 319), and 16 red. Two crossings require controllers and one row is
boot metadata. The saved results are in `recordings/tape_census_clean.json`.
Rows 331–332 are consecutive Mother Brain visits from different takes.

## Known resource and route gaps

- **Early Brinstar.** Super selection, the remaining pickup animation and
  enemy drops differ at the s6→s8 handoff. Spend Supers deliberately on gates.
  Keep enough energy and ammo to survive Green Hill Zone, get Spazer and kill
  Kraid. Measure this on the actual prefix.
- **Maridia, row 212.** The source ammo jumps from 30 to 60 Missiles,
  4 to 15 Supers and 5 to 10 Power Bombs. These are current counts.
  Measure actual capacities and stock on the Clean entry and budget
  natural pickups or drops for later fights.
- **Lower Norfair, row 253.** s31 introduces Spring Ball (bit 0002) without
  a pickup. The room graph gives an approach from Oasis (row 229) via
  East Sand Hall, Pants Room and Shaktool to the Spring Ball Room. The
  return leads through Shaktool, East Pants Room, Pants Room and East
  Sand Hall to Oasis. Develop the crossings and pickup as a row 229
  candidate, then prove the original exit to West Sand Hall. Alternatively,
  prove the later sections with the current loadout and update their item
  requirements in the board. The detour has topology evidence only.
- **Tourian, row 307 onward.** At the Landing Site seam, source energy
  jumps from 344 to 499, Missiles from 65 to 70 and Power Bombs from 2
  to 15. Budget Metroid rooms, barriers, Mother Brain's drain and escape
  from the actual Clean loadout. The end-of-credits detector has been
  verified from the ending pin. The master must continue past state 39
  to the fully drawn final message.

## Commands

Run from the repository root with the local ROM and human tapes installed.

```bash
export QT_QPA_PLATFORM=offscreen SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy

# Validate joins, physical room links, input slices and the ship-launch row.
uv run python snes/super_metroid/scripts/tools/tape_census.py --audit

# Source-entry candidates on the current board, including the late game.
uv run python snes/super_metroid/scripts/tools/tape_census.py
uv run python snes/super_metroid/scripts/tools/tape_census.py --rows 70-76 \
  --out snes/super_metroid/recordings/kraid_census_clean.json

# Advance to a milestone, then independently verify that milestone again.
uv run python -m super_metroid.splice chain --through-row 55
uv run python -m super_metroid.splice chain --replay --through-row 55

# Include the actual room sequence and its owning rows in the replay output.
uv run python -m super_metroid.splice chain --replay --proof-json

# Final acceptance returns success only for a complete loadless replay.
uv run python -m super_metroid.splice chain --replay --require-credits
```

Use `--override ROW=path.npz` while extending an unaccepted room. An override
for a row already on the master requires a new prefix ending before that row.
The board can be regenerated with `chain --rebuild-board` during a solver run.
Keep all development output under Super Metroid. The older assisted Phantoon
run, reference TAS movies and parked lsnes work are described in the
[game instructions](../AGENTS.md).
