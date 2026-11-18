# Clean track

Clean is the goal: power-on through the end of credits using controller
inputs and no external RAM writes. The assisted run to
Phantoon ([ASSIST_CONTRACT.md](ASSIST_CONTRACT.md)) is the older verified
tip and keeps its own artifacts. Benchmark labels:
[BENCHMARK_SPEC.md](../../../docs/BENCHMARK_SPEC.md). Work: [plan.md](plan.md).

## What Clean allows

| Write | Allowed |
|-------|---------|
| Controller buttons | Yes |
| Read-only RAM | Yes |
| Pickups, enemy drops, and recharge stations activated in-game | Yes |
| Energy or ammo restored by an external write | No |
| Capacity, items, boss, doors, pose, timer | No |
| State load inside the claimed run | No |

Moonwalk (`$09E4`) is a file option, not a resource; the legacy
`continuous.py --to bombs --clean` controller path may set it for Climb's
first descent and clear it before the Pit seed. `master_tape_clean` makes
zero RAM writes of any kind (including `$09E4`).

## Lookahead while building

Building a Clean run may save the core, play candidates ahead, and restore
the same frame. That is how a room is searched and how a fight picks its
next action. It is the ruling ALttP has had since 2026-10-02
(`snes/alttp/docs/ASSIST_CONTRACT.md`), adopted here 2026-11-16.

Clean development and diagnostic scripts must also stay write-free: use
read-only RAM inspection and save/restore lookahead only, never RAM-write
probes (such as pinning energy or ammo).

Lookahead is development. It is not the claim. A run that used it reports
`rollout_restores` and is quoted as development evidence only.

## Source buttons from human tapes

Human recordings were captured with refills on, so a human tape by itself is
never a Clean claim. Human-tape slices *are* valid candidate button sources
for `master_tape_clean` because only raw binary SNES-12 button arrays are
copied into the master tape. A plain button replay cannot execute RAM writes;
when the resulting master tape passes power-on replay on the unmodified
vanilla ROM, every energy, ammo, item, capacity, and boss transition is
produced natively by the game.

## The claim

A Clean claim is the master tape replayed from power-on (`retro.State.NONE`)
on the unmodified vanilla NTSC ROM (`SHA-1 da957f0d63d14cb441d215462904c4fa8519c613`)
in one emulator session with:

- zero state loads and zero restores
- zero RAM writes of any kind
- zero deaths
- natural inventory, capacity, boss, and event advancement
- the leave it names (room, game state 8, items) read from RAM at the end

Self-graded harness output alone is not sufficient. Every Clean milestone
claim must pass **both**:

1. `uv run python -m super_metroid.splice chain --replay` (instrumented
   load/write guard counters, vanilla ROM SHA-1 check, planned crossings,
   settled leaves, and full bank-`$7E` final state).
2. `uv run python snes/super_metroid/scripts/tools/audit_master_tape.py`
   (zero imports from `super_metroid` or `retro_harness`; boots
   `retro.State.NONE`, steps the raw `.npz` button rows only, and verifies
   vanilla ROM SHA-1, zero death frames, final RAM state, and bit-exact
   emulator snapshot SHA-256).

A room cleared from a human save state, a pin, or a run that restored is
not a Clean claim. Neither is the human tape itself: it was recorded with
refills on.

## Artifacts

Clean reports and tapes use a `_clean` stem and never overwrite the
assisted `<tip>.json` / `.mp4`. A red Clean run does not demote the
assisted tip.

## Verified so far

| Leave | Frames | Evidence |
|-------|-------:|----------|
| Morph | 26,824 (split on the bombs run) | `recordings/bombs_clean.json` |
| Bomb Torizo exit, Parlor `0x92FD`, items `0x1004` | 49,321 ×2 (2026-08-06) | `recordings/bombs_clean*.json` |
| Spore Spawn Room `0x9DC7`, 114/199 energy, 0 missiles | 55,860 (2026-11-17) | `recordings/master_tape_clean.json` + `.npz` |
| Spore Spawn defeated and Super Missiles collected in `0x9B5B`, 96/199 energy, 5 Super Missiles | 67,594 (2026-11-17) | `recordings/master_tape_clean.json` + `.npz` |
| Kraid defeated, Varia collected, returned to Baby Kraid Room A521 after row 77 | 98,785 | `recordings/master_tape_clean.json` + `.npz` |
| Warehouse Kihunter Room `0xA4DA` after row 78 | 102,313 | `recordings/master_tape_clean.json` + `.npz` |
| Warehouse Energy Tank collected and Business Center `0xA7DE` reached after row 83 | 107,310 | `recordings/master_tape_clean.json` + `.npz` |
| Bubble Mountain `0xACB3` after both save-room visits, row 92 | 112,778 | `recordings/master_tape_clean.json` + `.npz` |
| Speed Booster, Wave Beam and Ice Beam collected. Ice Beam Gate Room `0xA815` after row 118 | 132,095 | `recordings/master_tape_clean.json` + `.npz` |
| Alpha Power Bombs collected and Wrecked Ship Main Shaft `0xCAF6` reached after row 137 | 154,144 | `recordings/master_tape_clean.json` + `.npz` |
| Phantoon defeated and Gravity Suit collected in `0xCE40` after row 148 | 178,278 | `recordings/master_tape_clean.json` + `.npz` |
| Both yellow hatches opened and Hellway `0xA2F7` reached after row 154 | 183,706 | `recordings/master_tape_clean.json` + `.npz` |
| Hellway crossed and Red Tower `0xA253` reached from Noob Bridge after row 157 | 185,403 | `recordings/master_tape_clean.json` + `.npz` |
| Sloaters refill, Geega Power Bomb farm and Fireflea Room `0xA293` entry after row 158 | 194,587 | `recordings/master_tape_clean.json` + `.npz` |
| Maridia Tube broken and Crocomire's Room `0xA98D` reached after row 180 | 207,514 | `recordings/master_tape_clean.json` + `.npz` |
| Crocomire defeated and Grapple collected in `0xAC2B` after row 185 | 220,509 | `recordings/master_tape_clean.json` + `.npz` |
| Maridia reached through the broken tube and Mt. Everest `0xD0B9` reached after row 200 | 235,292 | `recordings/master_tape_clean.json` + `.npz` |
| Botwoon defeated and Draygon's Room `0xDA60` reached after row 209 | 256,172 | `recordings/master_tape_clean.json` + `.npz` |
| Draygon defeated and Space Jump collected in `0xD9AA` after row 211 | 261,300 | `recordings/master_tape_clean.json` + `.npz` |

The row-211 proof matches every planned crossing, every settled leave, and
the final core snapshot on the unmodified vanilla ROM with zero loads, RAM
writes, or deaths — verified by both `splice chain --replay` and the
zero-harness-import `scripts/tools/audit_master_tape.py`. It recorded 219
transitions, including 195 planned crossings, 16 farming transitions in
rows 41 and 46, the two Sloaters Refill transitions in row 127, the two
Kihunter runway transitions in row 135, and four refill and farming
transitions in row 158. Samus finishes row 211 in Space Jump Room (`0xD9AA`)
at `(85,155)` with 71/399 energy, 28/35 missiles, 1/5 Supers, and 3/5 Power
Bombs. The final state has the Crocomire flag in `$7E:D82A`, both Botwoon
and Draygon flags in `$7E:D82C` (`0x03`), the tube event in `$7E:D821`,
items `0x7325` (with Space Jump), and beams `0x1007`.
[Return-route handoff](plan.md#start-here-next-session)

On row 210 (`combat.draygon.play_clean_draygon_leave`, 4,334f), Samus drops
off the entry ledge, selects Grapple without spending ammo, and patrols the
floor to enter Draygon's left-facing grab with high energy. Because the
upper-left turret `(40,184)` is already sparking on room entry and
left-facing grabs shoot Grapple left, a 25-frame save/restore lookahead
latches onto `(40,184)` as Draygon climbs and electrocutes Draygon from
6000 HP to 0 without spending any missiles, Supers, or Power Bombs. After
`$7E:D82C` bit 0 sets, Samus jumps onto the left ledge, opens the blue door,
and settles in `0xD9AA` at `(216,139)` with 71 energy. Row 211 then clears
as recorded (`tape+idle:0`, 794f) to collect Space Jump.

On row 158, Samus recharges at Sloaters Refill and climbs Red Tower above
the Geegas. The Noob Bridge round trip resets the room, and shots on the
next descent yield a Power Bomb from the right Geega. Samus farms its
respawns from the ledge around `(187,1387)`, filling all five Power Bomb
slots, then spends one to enter the Fireflea Room with 399 energy.
The other four clear Red Tower on the return (160), break the Maridia Tube
(166), open the Norfair Map Room (169), and open the Ice Beam Gate floor
(178). Canceling Power Bomb selection on rows 164 and 170 preserves that
stock. Row 175 trims nine frames to climb back to the map room. Row 180
skips the first weapon-selection press to land on Supers for the Speedway
hatch.

On row 181, timed charged shots push Crocomire onto the breakable bridge.
His drops restore energy and Supers to capacity and supply two Power
Bombs. A running jump from `x=335` crosses the pit before the left door.
The Indiana Jones Room uses both Power Bombs. Samus collects Grapple on
row 185 and skips the empty Power Bomb slot when selecting it on row 186.
Row 194 rejoins the recorded Grapple swing from the right ledge, collects
the missile expansion, and uses a Super to open the green gate. Samus
climbs the broken Maridia Tube on row 198 and collects two Power Bomb
drops on the Main Street crossing.

The Mt. Everest crossing rejoins the recorded swings from the first ledge
and the right plateau after changing the Grapple release timing. Crab
Shaft waits 32 frames on entry and Aqueduct waits 25. Botwoon's head
takes three Supers and 21 missiles on row 205. Samus collects drops and
enters the Energy Tank Room with 272 energy, 17 missiles, three Supers,
and four Power Bombs.

On the return from Gravity Suit, a 36-frame release on row 149 avoids the
West Ocean Skultera hit. A 13-frame wait on entry to row 152 times the
first green Kihunter kill for a second Power Bomb at frame 180,990. The
shaft costs 30 energy. Samus spends one Power Bomb on the elevator hatch
at frame 181,701 and the other on the Caterpillar door into Hellway at
frame 183,426. The searched Hellway crossing collects a
small energy drop and enters Red Tower with 12 energy. A 61-frame wait
in the Red Tower descent syncs the Noob Bridge crossing.

Row 134 waits 15 frames before the Waver kills and collects an extra Super.
Row 136 opens the West Ocean door with one shot, leaving one Super for the
green hatch on row 138. Three missiles open the Basement eye door. Charged
beams and flame pickups clear Phantoon, then the loot sweep leaves the
Basement at 94 energy. The Basement return keeps all 94 energy, and the
Attic enemy clear raises 74 to 132 with drops. Retimed West Ocean and
Pancakes inputs reach Bowling Alley with 129 energy. The searched Bowling
jumps cost 90 energy before the Chozo ride and the natural Gravity pickup.

The Cathedral controller clears row 86 from the master entry. The row-87
inputs follow the recorded lava crossing into a searched jump sequence and
enter Bubble Mountain with 325 energy. Row 91 replaces a recording reset
with an actual doorway crossing into the save room. Row 92 uses a short
turn, beam shot and return through that door. Row 93 climbs with wall jumps
from the save-room runway and opens the Bat Cave door with a Super. The
Speed Booster Hall crossing on row 96 uses a Super to open its red door.
Row 97 starts one frame later to collect Speed Booster and return. The
retimed Double Chamber traversal on row 102 reaches the Wave Beam Room.

On the Ice return, the row-119 input sequence uses bombs in the morph tunnel
to clear the Ice Gate Room at 7 energy. A searched jump sequence clears the
Skree Boost Room on row 126 at 5 energy. Row 127 opens the Sloaters Refill
door with a Super and uses the station to restore energy to 399, then
returns to Red Tower and climbs to Hellway. The Alpha Power Bomb capacity is
collected on row 130. The searched Hellway crossing takes 32 energy and
508 frames. It saves 81 energy over the previous controller leave. Row 135
uses the Kihunter runway to charge the Moat spark and reaches West Ocean
with 172 energy. The West Ocean crossing uses 136 energy and one Super
to enter the ship on row 136.

Row 78 used 3,528 frames from `play_baby_to_kihunter_return`
(`recordings/row78_baby_to_kihunter_clean.npz`). The source tape stalled on
row 79, so the master used 1,626 frames from `play_kihunter_to_zeela_return`
(`recordings/row79_kihunter_to_zeela_clean.npz`). Both controllers ran from
the accepted predecessor leave. Recorded inputs then cleared rows 80–83,
including the natural energy capacity increase from 299 to 399 on row 81.

The first two came from the controller path before lookahead and still
stand. The master tape includes Bomb Torizo, natural Terminator tank
collection, a searched Green Pirates Shaft entry, the reconstructed Green
Brinstar Main Shaft missile farm (`recordings/row39_master_to_dachora_clean.npz`,
1,858f), a searched Dachora Room traversal (`entry-pulse:32`, 1,172f), the Big
Pink leave (row 41, 4,062f, below), and the Kihunter room as recorded (row 42,
832f, no hit).
Its independent power-on replay passed with zero loads, RAM writes, and
deaths. Every settled leave and every parsed final game-state field matched;
endpoint flags were read from bank `$7E`, not the low-WRAM open-bus view.

Development restores can shift audio/subframe snapshot fields. Full core
snapshot equality is reported separately from gameplay equality. Two
fresh loadless replays were also checked byte-exact against each other.

## Chain commands and handoff

```bash
QT_QPA_PLATFORM=offscreen SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m super_metroid.splice chain --replay

# Replay the accepted prefix, then search its next real room.
QT_QPA_PLATFORM=offscreen SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m super_metroid.splice chain
```

`splice.assemble.solve_tape_chain` runs candidates through `tips.play_hops`.
Only accepted buttons advance the master ledger. The replay blocks state
load and RAM write APIs, including `memory.assign` and writable bank views.
`--override ROW=PATH.npz` accepts a searched `actions` array; it must clear
from the real predecessor, never from its development pin alone.

The repaired candidate board is `recordings/chain_board_clean.json`:
340 rows from 27 tapes, through the ending. It retains pickup tails and
replaces stale tape slices. Coverage is a candidate inventory, not proof.

Row 41, **Big Pink → Spore Spawn Kihunter Room**, is accepted through the
room controller `routes.kpdr.brinstar.big_pink.play_clean_big_pink_leave`
(wired in `splice.assemble._chain_controller`). The master arrives at
`(551, 651)` with 109 energy and zero missiles; the pink door at `(987, 139)`
takes five. The leave, all by buttons against the ROM:

1. Farm Zeb drops on the pipe ledge. A shot delay is kept only if the ROM
   rolls a missile drop (+2) for it. The pipe re-arms when the room reloads,
   so the farm leaves by the west door and returns. Four drops: 8 missiles.
2. Follow the human climb (shots and item select masked) with a parallel
   beam. Near a Zeb, Sidehopper or Reo it branches (fire, wait, step, back
   off); every branch that loses energy is dropped. It killed a Sidehopper
   with a missile and picked up its drops.
3. Five missile hits on the pink door, then through it.

Zebs (`0xF193`) drop through projectile `0xF345`. Its settled loop is
`0xEDBD/0xEDC1` for a missile, `0xED91..0xED9D` for small energy, and
`0xEDA7..0xEDB3` for big energy. Shot delays of 2–12 frames gave a missile
about one delay in six.

Result: 4,062f, 114/199 energy (no hit taken), 3 missiles, settled `0x9D9C`.
The Kihunter room (row 42) then clears as recorded in 832f (`tape+idle:0`),
which spends the last 3 missiles: the master reaches `0x9DC7` at `(137, 713)`
with 114/199 energy and **0 missiles**.

Row 43 farms missiles from spores and reaches 60 boss HP without losing
energy. Every candidate at the stalled branch ran into a 12-energy hit.
The rollout retries from the last ledge checkpoint with an energy floor
24 below entry. From 114 energy and zero missiles, the accepted route
takes one hit and reaches settled `0x9B5B` with 102 energy and Brinstar boss
bit `0x02` in 10,989 frames. The following 745-frame pickup tail collects
5 Super Missiles and ends with 96 energy.

`recordings/row43_master_to_supers_clean.npz` holds the row 43 buttons.
The independent power-on replay matched every settled leave and the core
snapshot with zero state loads, RAM writes, and deaths.
The local ROM regression pin `scratch/spore_master_hp160_clean.state` under
the Super Metroid integration comes from frame 63,346 of that tape's
power-on run, with 160 boss HP remaining.

The older controller command remains available:

```bash
uv run python snes/super_metroid/scripts/record/continuous.py --to bombs --clean --no-video
```
