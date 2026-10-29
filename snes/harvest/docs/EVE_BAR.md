# Eve at the bar

Eve can be courted at night in the bar. The door is open from 17:00 (shut at
12:00, shut all Sunday), the outdoor hour caps at 18 and the indoor clock never
moves, so the courting loop costs no game time. Code: [`eve_loop_task.py`](../harvest/tasks/eve_loop_task.py)
(`EveTalkLoopTask`). Tests: `tests/test_eve_bar_loop.py`. Decoder:
`uv run python -m harvest.scripts.event_dis` (see the end).

Every run below is a what-if: the date, hearts or tool slot were edited in RAM
before the task started (the pin is a D22 evening, which pays nothing). The
tasks themselves send controller input only. No run starts from power-on.

```bash
# Night, anywhere in town (it walks to the door). Repeats enter, talk, leave.
HEADLESS=1 ./run_bot.sh play --autoplay --state <pin> --eve-loop \
  --eve-target-points 250
```

Stops with `SUCCESS` at the target, or `FAILURE` with a reason that starts
`eve dead_zone`, `eve none`, `eve paid nothing`, `eve bar door did not open`.
The autoplay day then walks home and sleeps.

## Traps for the day plan

- `run_bot.sh` uses the harvest `.venv`, which lacks PyYAML; a day plan reads
  `docs/perfect_rancher/milestones.yaml`. Use the root venv:
  `HEADLESS=1 ../../.venv/bin/python -m harvest.runtime.harvest_bot play --autoplay ...`.
- `build_day_phases_from_ram` schedules the bar, Spring D23, the Sunday feather
  buy, the shed fetch and the Saturday proposal (`apply_eve_phases`). The
  Saturday fetch and proposal replace farm work together. `GoToSleepTask` keeps
  the bar walk after the 18:00 clock cap.
- Each bar entry pays +8 once; leave and re-enter for the next. The first entry
  is the first-meeting scene, which the loop skips. Eve moves after that visit,
  so a recorded tape of it cannot be replayed.
- The Sunday seller (sprite `0x00DC` at (568,360), face (555,368)) spawns only
  after house bit `$7F1F64 & 0x40`. The sprite at (584,360) stays on text
  `0x0078`.
- First house (`planner/tasks/home.py`): fetch the axe, bank 250 wood, keep
  6,000G (the order plus the feather), request the cabin sign, pay the next
  morning, sleep four nights. The cabin's first visit is an introduction: leave
  and re-enter for the sign. After remodeling, finish the carpenter scene and
  walk off the steps to (136,376) before BFS. A marriage in Fall or Winter
  leaves a first-house order that does not arm the clock.

## What the ROM does

Regular bar script: event group 5 event 0, Eve's CC script at `$B3D213`
(`event_dis group 5`). The first entry ever runs group `0B` event 7 instead
(first meeting, Eve at (168,440), no hearts), then `$7F1F68` bit `0x0002`
selects group 5 for every later entry (`$83DBE6`). Group 5 seats Eve at
(56,448); stand at (78,446) facing left and press A.

One talk, by Eve's hearts (RAM points, `$7F1F27`):

| Hearts | Season / weather | Text | Pays |
|--------|------------------|------|------|
| 120–149 | any | `0x00EE` | **nothing** (dead zone) |
| 150–200 | any | `0x00EF` Sometimes / Not at all | Sometimes **+8** (`0x00F0`), Not at all +2 |
| else | snow flag `0x0008` (winter precipitation) | `0x039E` | nothing (Winter D30: +2) |
| else | Spring D22, or rain flag `0x0002` | `0x020A` / `0x00E0` | nothing |
| else | Spring | RNG 0 `0x00E5` / RNG 1 `0x00E6` juice | juice: Yes **+4**, No thank you **+8** |
| else | Summer, Fall | `0x00E9–0x00EC` | nothing |
| else | Winter (not D23) | `0x00ED` | **+2** every talk |

After a juice or Sometimes answer the script jumps to a loop (`$B3D234`,
`$B3D23D`) that only says "just kidding". Nothing pays again until the map
reloads, so each +8 costs one re-entry. Winter +2 and no-gain talks return to
the main loop, so talking again works inside one entry.

Gifts (`$B3D25E`, any time): items `0x01–0x05`, `0x10–0x17` +3; `0x06`,
`0x19`, `0x49` +8; poison mushroom `0x02` and fish `0x07` −8; `0x09–0x0C`
−2. Measured with a planted tomato-class item `0x06`: 124→132, text `0x0212`.
Real held items were not run. The arm left the planted `0x06` in hand, so
whether a gift consumes the item is still open.

Proposal: the feather check at the top of the bar loop (`$B3D246`) answers
hearts 0–249 `0x017C` "too early" and 250+ `0x0441` "talk to grandpa, go to
the mountain on a sunny Saturday". It sets nothing; the bar cannot marry. The
wedding is Eve's Saturday mountain script (`$B3B7DE`, map `0x10`, she stands at
(504,216) until 17:00): same feather check, hearts ≥ 250 then `0x017B`, wife
bit 4 (`$7F1F66` `0x10`), flag `$7F1F5E` bit 9, transition `$2D`. Both scripts
are skipped when `$7F1F6C` bit 4 is set.

## Proposal and wedding (run)

[`eve_proposal.py`](../harvest/tasks/eve_proposal.py) (`EveProposalTask`, kind
`propose_eve`, `--day-plan propose_eve`). The feather swing only reaches Eve
from one pixel: the check at `$83AE26` needs the point in front of the farmer
inside the 25-pixel box right of and below her anchor, so stand at (503,229)
facing up. (504,231) and (490,235) did nothing; talking (A) from (503,229)
gives the Saturday question `0x00DC`. The feather must be the selected tool
(`$0921` = `0x18`); the task swaps it in from the backpack slot with X and
fails with a named reason when it holds neither.

What-if run (Saturday D27, Eve planted at 250, feather planted in the tool
slot, farm 08:00): farm gate, path, the mountain switchback (22 waypoints,
probed from the entry at (328,728)), stand, Y. `0x017B`, `$7F1F66` = `0x10`,
map `0x2D`, the wedding in town, text `0x01BF` "We got married", then D28
06:00 in the house. 3,368 frames; the same from start offsets of 7, 29 and
111 frames. Blockers it checks first: already married, hearts < 250, not
Saturday, 17:00 or later, rain, snow or storm flags.

The feather was already in the tool slot on that research pin. Native buying,
shed pickup and marriage are received in the continuation below.

## Measured

Spring D10 night, `Y1_D22_Town_Eve` with the date edited (what-if pin
`LN_eve_whatif_d10_town`, local, gitignored; D22 itself pays nothing). Only
controller input from the task. Clock stayed 18:03 across every run.

| Run | Frames | Result |
|-----|--------|--------|
| far end of town → 120 | 19,393 | `SUCCESS` at 120: walk to the door, skip the first meeting, 15 × +8 |
| 0 → 250 target | 19,239 | stops `eve dead_zone at hearts=120/250` |
| 150 → 250 (hearts planted at 150) | 19,617 | 7 Sometimes, then 6 juice, 254 |

One entry costs about 1,100–1,400 frames: door 153, settle 190 (input is dead
while the scene fades), walk about 100, talk 350–900, leave about 100. The
juice question types for about 240 frames, then the arrow, then Down until
`$018F` reads 1, then A. Pressing A early picks Yes.

## Crossing 120–149: the Flower Festival

Talk cannot cross it (max +8 from 119 lands on 127). The Spring D23 Flower
Festival does, once a year. Code: [`flower_festival.py`](../harvest/tasks/flower_festival.py)
(`FlowerFestivalTask`, day-plan kind `flower_festival`, named sequence
`--day-plan flower_festival`; the Spring D23 `ATTEND_FESTIVAL` phase uses it
when `attend_festivals` is on).

What the ROM does (group `0E` events 0 and 1, `event_dis group 14`):

1. Before 15:00 press Left at the town's west edge (x≈10, y 128–150): map
   `0x08` loads at (264,312) and the clock is set to 07:00. The script stores
   `0xF7FF` in `$7F1F74`.
2. Each of eleven NPCs clears one bit of `$7F1F74` when talked to: six
   villagers bits 0–5, then Maria 6, Ann 7, Nina 8, Ellen 9, Eve 10. A girl
   pays +8 (hearts ≤100), +12 (101–300) or +15 (301+), once.
3. With all eleven clear the blue-haired woman at (120,248) leaves the alcove
   gate. The host in the alcove (sprite `0x293`, (88,232)) asks "Can we start
   dancing now?"; "Let's dance" sets bit 11. The next girl talked to is the
   partner (her bit, Eve's is 4). The end event (`$B3FE09`) gives her **+30**
   (`$B3F1B7`, `$B3F1E4` if she is the wife), the dance plays, and the game
   puts the farmer to bed: next is D24 06:00 in the house, so the festival
   takes the D23 night.

Run (what-if pins, local, gitignored: `LN_eve_whatif_d23_farm` is the farm at
10:00 D23, `LN_eve_whatif_d23_town_west` the town edge; Eve planted at 130):
130 → 142 on her talk, 172 after the dance, the other four girls +8, day
rolled to D24 06:00. About 10.6k frames from the town edge, 11.3k from the
farm; start-time offsets of 0, 13, 41, 97, 150 and 333 frames all pass.
The route is a lattice of 8-pixel hops probed with emulator save/restore
(`harvest/maps/flower_festival_walk.json`, 856 cells, regenerate with
`uv run python -m harvest.scripts.probe_walk_grid`); the map has no
walkability data. The alcove row is hand-walked.

Chained on what-if pins (dates edited between legs): bar D10 night 0 → 112
(18,325 frames, 15 entries), festival D23 112 → 154 (10,663 frames from the
bar door, out the south gate, up the path), bar D24 night 154 → 250 (18,728
frames, 13 entries). Eve reaches the proposal floor on D24 of Year 1.

The Spring chain that reaches 250 is a bar night until hearts are at least
108, the Flower Festival, then the D24 night. Stop the first night at the
first total of 108 or more. The festival adds 42. Summer and Fall pay
nothing. Winter pays +2.

## Native composition and evidence

Verified 2026-10-15 with controller input from natural predecessors. The
first house, Sunday sale and wedding are Clean continuations from power-on:
one initial checkpoint load per later segment, zero mid-run loads, zero RAM
writes and no assists. `STATUS.md` is unchanged.

| Segment | Native receipt | Artifact |
|---|---|---|
| Power-on → Summer D1 | Spring D23 retains 2,840G through the festival; the seed shelf bit stays clear. After buying corn, Summer D1 ends at 3,440G. | `recordings/house_marriage_power_on.json` (576,282 frames; deliberately stopped at the checkpoint) |
| Summer D1 → carpenter order on Summer D28 | Axe fetched normally, farm stumps and renewable mountain stump bank 252 wood. Corn income funds the order: 6,200→1,200G, 252→2 wood; construction bit `0x0080`. | `recordings/house_marriage_resources.json` (310,147 frames) |
| Four construction nights | Counter 0→1→2→3→4; first-house bit stays clear for three nights and sets on Fall D2. | `recordings/house_build_nights.json` |
| Fall D3 Sunday sale | Seller `0x00DC` at (568,360), text `0x01C3`; 1,680→680G and shed row 0→`0x80`, 2,209 frames. | `recordings/house_feather.json` |
| Shed fetch → sunny Saturday proposal → wedding | The native continuation proposes on Fall D9 and finishes Fall D10 06:00 in the remodeled house. Eve 250, wife bit `0x0010`, house bit `0x0040`; proposal leg 3,696 frames. Two more sleeps retain the wife bit on Fall D11 and D12. | `recordings/house_wedding.json` |

The automatic continuation `recordings/house_marriage_natural.json` also
buys on Fall D3, marries on Fall D10, and reaches Fall D18 06:00 with the
wife bit retained: 170,857 frames, zero bad days/rescues, zero RAM writes
and zero mid-run loads. Resume `LN_house_married_natural`. The compact
receipt and validated source hashes are `recordings/house_marriage_receipt.json`.

The state chain is `LN_house_marriage_power_on_FAIL` →
`LN_house_paid_natural` → `LN_house_built_natural` →
`LN_house_feather_natural` → `LN_house_wedding_natural`. `_FAIL` is the
runner's filename for the deliberately interrupted power-on checkpoint;
the recorded source hashes connect the later segments. State bytes, dates,
hearts, money and wood were earned in the ROM. The earlier
`eve_chain_power_on.json` (683,321 frames, seed bought, 1,340G and no wood)
remains a historical run.

`build_day_phases_from_ram` schedules the bar, Flower Festival, first house,
Sunday buy, shed fetch and Saturday proposal. `GatherHouseWoodTask` and
`HouseUpgradeTask` live in `planner/tasks/home.py`. First-house planning
still pays 6,000G (5,000G order plus the 1,000G feather) once a cow is
already owned. Pasture and the 5,300G cow buy stay on the calendar when
Eve is at 250. Marriage can land in Fall or Winter. Farm stumps pay six wood;
the mountain stump provides renewable fallback. The cabin closes on
weekends; its first visit is the introduction, so leave and re-enter for
the request sign. Native room navigation uses the ROM's 16-column stride.

After remodeling, `ExitToFarmTask` finishes the carpenter's completion and
clock dialogue, then walks south off the steps to (136,376). BFS starts on
that open stand. `ReturnHomeTask` approaches the same stand before pushing
through the door. Saturday shelf fetch and proposal replace crop work as
one plan. `EveProposalTask` waits through the date roll until the new house
is controllable; the multi-day planner lets the event finish and records
its overnight instead of starting another return/sleep route. The level-1 bed approach passes below Eve at y=136 and uses fixed interior coordinates during the entry warp.

From repo root:

```bash
QT_QPA_PLATFORM=offscreen HEADLESS=1 uv run --with pytest python -m pytest \
  snes/harvest/tests/test_house_upgrade.py -m rom -q
```

The six ROM checks cover lumber, the cabin request, corn planting, all four
construction nights, the natural Sunday sale, and the complete wedding.
The receipts depend on the local natural checkpoints above.

Second house, measured from `LN_house_married_natural` (Fall D18 06:00,
2 wood, 680G, Eve 234, wife bit still `0x0010`, `num_cows` 0,
`wife_pregnancy` 9). The planner's open quote is 500 wood and 10,000G once
that wife bit and house bit `0x0040` are set. One mountain chop banks the
renewable stump: `recordings/house2_wood.json` shows 2→8 wood in 3,757
frames, money stayed 680G, `upgrade_flags` stayed `0x41`, the wife bit
stayed set, ram writes 0, mid-run loads 0. Resume
`LN_house2_wood_natural`. The 10,000G order is not paid, and it now waits
until a cow exists. Fall has no seed, so the cow's 5,000G and the later
house quote both need a spring. The quote can stay open into Year 2.

The house-planner unit tests and the new lumber ROM test passed when that
chop was recorded. The original six house ROM tests were not re-run.

Pregnancy still needs Eve at 450, then 650, and house bit `0x0080`. The
counter is already 9, so the heart floor and the second house are the
gates, not another 20 nights from zero. The indoor wife script is loaded
on this morning (`$B59386`). Its positive arms are gifts only: +4 at
`$B593E9` for items `0x01`–`0x05` and `0x10`–`0x17`, +8 at `$B593FD` /
`$B5941B`. The pin is holding nothing. Two face-down presses at the first
grape stand (326, 409) on Fall D18 left the held item at 0, so the gift
was not entered (`recordings/eve_heart_floor.json`, 1,994 frames, hearts
234→234). Other ways across Eve's 120–149 bar dead zone remain unproved:
the wife-only +100 script, and the grandfather rescue +50. A forage item
that is actually on the map is still required before that gift. Cow cash
comes before another wood session or the 10,000G payment.

## Decoder

`event_dis at 0xB3D213`, `event_dis group 5 --event 0`,
`event_dis sites eve` (every add to Eve's hearts and the events that reach
it). Scripts are LoROM, `$B3:8000` is file offset `0x198000`. Opcodes from
`0x51` up are unknown and stop the walk. A few Eve sites (`$B4A724`, `$B4AFE5`)
are not reached from any group event yet.
