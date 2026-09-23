> Historical lane note. The live sitting is the gathering prefix in [PRE_L1.md](../PRE_L1.md). This file stays because the clean-tip ladder or a route doc still cites it. It is not the current plan.

# rr-d6v residual — Clean L6 Entrance→TF (no Survival)

**BLOCKED.** `route_eligible=false`. Do not close the bead. Not a STATUS
claim.

**The `Level6Entrance` pin is invalid** (`$066F=0x2F`: 15 whole hearts in 3
containers — a state normal play cannot reach). Everything measured from it,
including this page's own ledger, rests on a ~7x inflated heart budget. Read
"Invalid pin" first; it is this sitting's most important finding. The two
follow-on ideas (0x7a kill speed, skip 0x7a) are **parked pending a valid
pin** — tuning against a fake denominator cannot produce a trustworthy green.

Do not resume 0x59-chase or east-waist LEFT (v4–v9, blocked 3/3 each). Do
not re-tune 0x78 poses. Do not rebuild the pin or poke health from this
lane — that is a decision about what Clean L6 means and belongs upstream.

## 2026-09-14 — building the pin found two engine stalls first

The measured path is `scripts/fixtures/capture_level6_entrance_fixture.py`
(new): run `run_survival_spine` from power-on `--through level6-entry` in one
emulator session (`set_state` 0, `mid_run_state_load` false), check the
arrival pose and `health_byte_is_coherent`, then `save_state` +
`write_state_provenance`. It refuses to save a pin that fails either check.

It could not reach L6 on the first three attempts, and each stop was the same
shape — **a rule that holds one button with no escape** (the
`zelda_i` controller bug class) — so both fixes are on the engine, not on a
room:

**1. `ROUTE_ENTRY` had no escape.** Survival `clear45_key` spent **8999 of
9000 frames** pressing DOWN at L1 `0x44` `(168,141)`, `combat_frames=0`, one
`timeout` note. The leftover had drifted (upstream L1 timing changed; the
chain is frame-perfect), so `ROOM_45_SURVIVAL_SPEC`'s first leg `(192,165)`
aimed straight through the statue at `(176,160)`.

`dungeon/route_entry.py` (new, `EntryRouteWalker`, mixed into
`GenericDungeonRoomController`) watches for 24 identical-pose frames on a held
`entry_route` button, notes `entry_route_stall_f<n>_(x,y)_wp<i>(tx,ty)`, then
replans off the live `$6530` map, latching until the leg advances; with no
tile map it drops the leg (bounded by the waypoint count). **Below the
threshold the axis walk is byte-identical, so a green chain does not move** —
Clean M5 re-ran 2/2 `triforce=0x01` at **19416f**, the same frame count as
before.

The replan walker is `sticky=True`, and that is the interesting half: the
tile map says `(168,142)` is *free*. `blocked_link_cells` samples one point
under Link, but Link is 16px wide, so a statue his **right edge** clips reads
as open floor. A forgetful walker replans back into that cell every few
frames and yo-yos; keeping the miss is what converges. The single-point
occupancy model under-reporting wide-body collisions is a live limitation,
recorded here, not fixed.

Result: Survival L1 is **green** — `clear45_key` finishes, the run reaches the
Triforce room `0x36` mode 18, 30 `entry_route` frames + 11 `entry_route_replan`
+ 1 `entry_route_skip`.

**2. `COLLECT_REWARD` idled 2px off the pickup.** The next stop was L2 `0x7e`
`clear7e_key`: **6752 of 8000 frames** on `reward_wait` at `(138,141)` with
the key target at `(136,141)` — inside the 2px waypoint tolerance, so the walk
stopped, and nothing moved Link the last two pixels.
`GenericDungeonRoomController._reward_nudge` now closes them after 24
consecutive idle frames (`reward_wait` / `collect_wait` /
`collect_skip_unreachable`), and rocks one pixel when already exactly on the
tile. This is `Level6EastKeyController._go_key`'s local wiggle, promoted.

**3. A walker with no map, and a goal inside a wall.** With 1 and 2 fixed the
spine reached L2 `0x6e` `enter_6f_key`, which spent **3999 of 4000 frames** in
`band_wait` at `(41,189)` — not one pixel of movement. Two causes, both shared:

* `Level2Enter6fKeyController` built a bare `OccupancyWalker()`, which knows no
  walls at all; it learns each diamond by bumping it and, being non-sticky,
  forgets it again. `walk.physics.measured_walker(ram, bounds, sticky=True)`
  (new) is the one way to build a walker that starts with the live `$6530`
  geometry; `route_entry` and this controller both use it now.
* With the real map bound, BFS answered honestly: the band target
  `(120, DIAMOND_BAND_6E=113)` **is inside a diamond**, so there is no path
  and the walker stands. `(208,141)` and `(160,113)` were both reachable from
  the same pose. `OccupancyWalker.next_dir` now retargets a blocked goal to
  `grid.nearest_open` and counts it (`retargets`) — but **only when the
  walker opts in** (`retarget_blocked_goal`, which `measured_walker` turns on
  and the bare walker leaves off). Turning it on globally was measured and
  reverted: it took Clean M5 from 2/2 `triforce=0x01` at 19416f to 2/2 red at
  `aquamentus_heart`, 18830f, TF `0x00`. Retargeting changes the frame a walk
  arrives on, and the L1 chain is frame-perfect — the same lesson as
  `zelda-l1-chain-frame-perfect`, paid for again. A goal that is already
  walkable is untouched either way, so sound waypoints never move.

**Where the pin attempt stands.** Not captured. The spine went from dying at
L1 `clear45_key` (stage 13 of 13) to clearing **23 stages** and stopping at L2
`0x6e` `enter_6f_key`, 33,355 frames in, with `set_state=0` throughout. Link
is coherent there (`$066F=0x33`, 3/4, TF `0x01`, keys 4, bow 1) — so the
*capture* machinery is proven end to end; only the route is short.

The next stop is **not** another instance of the button-holding class. It is a
route question: the SW pocket of `0x6e` reaches `(160,113)` and `(208,141)`
but not `(120,113)` or `(120,141)`, and `Level2Enter6fKeyController` aims at
`(120, DIAMOND_BAND_6E)`. Retargeting to the nearest open cell moved Link to
`(40,181)`, the west boundary, and stopped — so the band target is wrong for
this pose, not merely 9px off. **Pick a band cell the pocket can actually
reach** (measure it: dump `0x6e`, BFS from the live leftover) rather than
nudging the existing one. Do not add another hand-written waypoint without
checking it against the tile map first — that is the mistake this whole
sitting kept finding.

**4. Every controller now reports what held the frames.** The records-only
`reason_counts` + 30-frame `tail` that lived on `Level6EastKeyController` is
on the engine `report()` (`_record_reason`), and the local copy is deleted.
Both stalls above were diagnosed from it in one run each rather than a
bespoke probe per sitting — the thing this page asked for after losing a trial
to a 12000-frame stage that reported one tile.

## 2026-09-14 — the pin problem is the whole ladder, not L6

`scripts/audit_pins.py` (new) reads every level-entrance pin in its own
emulator process and checks `$066F` against `ram.health_byte_is_coherent`
(`lo <= hi`; new, alongside the existing `full_health_byte`). **Seven of
fourteen entrance pins are incoherent, all with the same `lo = 0xF` mistake
this page found on `Level6Entrance`:**

| pin | `$066F` | hearts / containers | TF | coherent |
|-----|---------|---------------------|----|----------|
| `At4A` / `At4B` | `0x32` | 2 / 4 | `0x01` | yes |
| `At5B` | `0x31` | 1 / 4 | `0x01` | yes |
| `At78` | `0x22` | 2 / 3 | `0x00` | yes |
| `Level1Entrance` | `0x21` | 1 / 3 | `0x00` | yes |
| **`Level2Entrance`** | **`0x3f`** | **15 / 4** | `0x01` | **NO** |
| **`Level3Entrance`** | **`0x7f`** | **15 / 8** | `0x03` | **NO** |
| **`Level4Entrance`** | **`0x6f`** | **15 / 7** | `0x04` | **NO** |
| **`Level5Entrance`** | **`0x3f`** | **15 / 4** | **`0x00`** | **NO** |
| **`Level5EntranceFromL4`** | **`0x7f`** | **15 / 8** | `0x0c` | **NO** |
| **`Level6Entrance`** | **`0x2f`** | **15 / 3** | `0x00` | **NO** |
| `Level7Entrance` | `0x22` | 2 / 3 | `0x00` | yes |
| **`Level8EntranceReconFixture`** | **`0x2f`** | **15 / 3** | `0x7f` | **NO** |
| `Level9EntranceReconFixture` | `0xff` | 15 / 16 | `0xff` | yes |

So this page's finding was not an L6 accident. **Every Clean heart number on
the `l2_tf`–`l8_tf` rows is against a fake denominator**, and the ladder now
says so: `spine/clean_tip.py` gained a `Blocker.INVALID_PIN` class and a
`pin` field, and those five rows carry it. `Level5Entrance` is worse than
incoherent — it holds `TF 0x00`, which no L5 arrival can (L1–L4 is `0x0F`).

Nothing was repaired by hand. Repairing a pin by writing `full_health_byte`
into it would make the byte legal without making it *measured*, and the
measured path now exists: see below.

## Invalid pin — `Level6Entrance` `$066F = 0x2F`

`ram.py:63` documents `$066F` as `hi = containers−1, lo = whole hearts`, so a
coherent byte always has **`lo <= hi`**. Read directly from the `.state`
files through `read_snapshot`, one emulator per process:

| pin | `$066F` | `$0670` | hi | lo | containers | filled | full | coherent |
|-----|---------|---------|----|----|------------|--------|------|----------|
| **`Level6Entrance`** | **`0x2F`** | `0xFF` | **2** | **15** | 3 | **15** | False | **NO** |
| `Level1ExitOverworld` | `0x33` | `0xFF` | 3 | 3 | 4 | 3 | True | yes |
| `At4A` | `0x32` | `0x7E` | 3 | 2 | 4 | 2 | False | yes |
| `At4B` | `0x32` | `0x7E` | 3 | 2 | 4 | 2 | False | yes |
| `At5B` | `0x31` | `0x7D` | 3 | 1 | 4 | 1 | False | yes |
| `At78` | `0x22` | `0xFF` | 2 | 2 | 3 | 2 | True | yes |

`Level6Entrance` is the only incoherent pin of the six. `At78` is the
correctly-formed 3-container version of the same byte (`0x22`, `full=True`),
which is what this pin should have held.

The repo already knew this shape and guards it everywhere except here.
`ram.full_health_byte` (`ram.py:199`) exists precisely for it:

> "low nibble is whole hearts, **not** a `0xF` full flag. Writing `0xF` makes
> the triforce/potion fill `INC HeartValues` until the nibbles match, which
> grants extra containers."

and `tests/test_assist.py:66` asserts `full_health_byte(0x2F) == 0x22`.
`UnlimitedHealthAssist` writes `health_byte_for_containers(accepted)` — always
`n<<4|n` — and clamps containers it did not see granted. So the guard is
real; the pin predates it.

**Consequence.** The ROM decrements that low nibble happily: this sitting's
trace shows 15 → 3 → 0 with no clamp. So every Clean run from this pin has
been fighting L6 on ~15 hearts where a coherent 3-container Link has 3 —
**and it still dies.** The Clean L6 red is therefore *deeper* than this page
previously recorded, not shallower: from a coherent pin, 0x7a's measured
spend is fatal long before 0x78 is reached. The old residual noticed the
smell — "health `0x2F` (3 HC, lo nibble `0xF` display — not `0x22` lo==hi)"
— and did not follow it.

The container count is as suspect as the low nibble. Clean has never reached
L6 (Clean M5 is L1 only), so this pin is a fixture standing in for a leg that
does not exist. A genuine post-L1–L5 Link has far more than 3 containers —
see "Real L6 arrival" below.

## Real L6 arrival — what a valid pin should look like

`run_survival_spine.py --through level6-entry` **could not be re-derived live
this sitting**: it is currently red at `clear45_key` in L1 (`tag
l6_entry_real`, 0/1, room `0x44`). The shared L1 / `dungeon.engine` /
`walk.physics` files are mid-edit in another lane, so that red is not
evidence about L6 and was not investigated from here.

Two archived **continuous power-on** Survival tapes stop at exactly this
pose and were used instead. Both carry `continuous_emulator_session=True`,
`mid_run_state_load=False`, `seamed=False`, `stop=level6_entry_0x79`,
`ok=True`:

| tape | end_frame | `$066F` | hi/lo | containers | full | TF | keys | bombs | assist `accepted_containers` | `container_clamps` |
|------|-----------|---------|-------|------------|------|----|------|-------|------------------------------|--------------------|
| `l6_entry_continuous_v2` | 179355 | **`0x66`** | 6/6 | **7** | yes | **`0x1F`** | 5 | 8 | 7 | **0** |
| `l6_entry_recompose` | 164891 | **`0x66`** | 6/6 | **7** | yes | **`0x1F`** | 4 | 8 | 7 | **0** |

Both land at level **6**, room **121 (`0x79`)**, `(120,205)`, mode **5** —
the same pose the pin claims to hold.

Survival refills health, so the *filled* nibble there is the assist's write
(`health_byte_for_containers(accepted)`, always `n<<4|n`). **Containers are
real**: the assist never grants one, and `container_clamps = 0` on both runs
means it never even had to reject a glitched high nibble. `capacity_writes`
is 0. (Both tapes do carry Survival inventory pokes — `poke_bombs=16`,
`poke_keys=2` — which is why keys/bombs above are quoted as context, not as
a Clean target.)

**So the pin is wrong on three axes, not one:**

| | `Level6Entrance` pin | real continuous arrival |
|---|---|---|
| `$066F` | `0x2F` (**incoherent**, lo 15 > hi 2) | `0x66` (coherent, lo == hi) |
| containers | 3 | **7** |
| triforce | `0x00` | **`0x1F`** (L1–L5) |
| L5 inventory | none | raft, ladder, whistle, bow, map |

A valid Clean L6 entrance pin should be a **7-container, TF `0x1F`** Link
with L5 inventory. That is more than twice the containers the pin gives —
but it is also a *coherent* counter, so it is still far less total damage
capacity than the 15-heart nibble the runs have actually been using.

## Pin glance

`Level6Entrance` play **0x79** `(120,205)` mode **5**. Isolated pin: TF
`0x00`, keys **0**, bombs **0**, bow **0**, arrows **0**, rod **0**, raft
**0**, ladder **0**, health **`0x2F` (invalid, see above)**, deaths 0. No L5
inventory.

Runner: `scripts/run_level6_entrance_tf.py --from-state Level6Entrance
--no-infinite-life --no-video --trials 1 --tag <tag>`. `make_env` +
`reset_obs` + `resync_custom_state`. Assist None. `poke_arrows=False`.
`allow_pokes=False`. No `--infinite-life` in any row on this page.

## Per-stage health ledger — `l6_final_t0/t1/t2`, 3/3 byte-identical

Hearts are the `$066F` low nibble in → out. `hits_by_cause` is
`dungeon.postmortem` (object type + approach side).

**The `15` denominator here is the invalid pin, not a Clean budget.** The
per-stage *hits* and *causes* below stay valid — they are what the ROM did —
but "arrives at 0x78 with 3" describes a Link who began with five times a
coherent 3-container Link's health.

| stage | frames | ok | hearts in→out | hits | by cause |
|-------|--------|----|---------------|------|----------|
| `level6_right_0x7a` | 374 | yes | 15 → 15 | 0 | — |
| `level6_east_key_0x7a` | 1084 | yes | **15 → 3** | **6** | `0x24_E`×4, `0x59_E`×1, `0x59_W`×1 |
| `level6_return_0x79` | 305 | yes | 3 → 3 | 0 | — |
| `level6_west_key_0x78` | 381 | yes | 3 → 3 | 0 | — |
| `level6_west_clear_0x78` | 308 | **no** | 3 → 0 | 1 | `0x59_S`×1 |

Total 2452f, deaths 1, TF `0x00`. Arrival health at 0x78 is `0x23`
(**3 hearts**). 0x78 kills in 308 frames from there.

**The whole spend is one room and, inside it, one pose.** Four of the six
0x7a hits are the same frame-by-frame picture:

| frame | Link xy | cause | cause xy | d | ttc | dodgeable | in line | action |
|-------|---------|-------|----------|---|-----|-----------|---------|--------|
| f295 | (76,133) | `0x59_E` | — | 12 | 33 | yes | no | `combat_engage_slash` |
| **f460** | **(88,133)** | **`0x24_E`** | **(96,125)** | **8** | **0** | **no** | **yes** | `combat_engage` |
| **f522** | **(88,133)** | **`0x24_E`** | **(96,125)** | **8** | **0** | **no** | **yes** | `combat_engage_slash` |
| **f662** | **(88,133)** | **`0x24_E`** | **(96,125)** | **8** | **0** | **no** | **yes** | `combat_engage` |
| **f726** | **(88,133)** | **`0x24_E`** | **(96,125)** | **8** | **0** | **no** | **yes** | `combat_engage_slash` |
| f944 | (96,133) | `0x59_W` | — | 8 | 33 | yes | no | `combat_patrol` |

0x78's own hit is `0x59_S` at (186,141), d=6, ttc=0 — one hit on 3 hearts.

## Root cause of the 0x7a spend (measured)

`(88,133)` against a parked `0x24@(96,125)` is `dx=+8 dy=-8`: the hitboxes
overlap on **both** axes, so the body damages while a swing aimed along x
from y=133 misses a target sitting at y=125. Link is pinned there (x never
leaves 88) for 266 frames, alternating `combat_engage` / `combat_engage_slash`
and taking a contact hit roughly every 62 frames. `assess` reports `ttc=0
dodgeable=False` — **no reactive layer can answer this**; the contact has
already started.

Two structural facts behind it, both still true in the reverted tree:

- `Level6EastKeyController._combat` overrides the engine's `_combat`
  wholesale and **never calls `evader.decide`**. `ROOM_7A_SPEC.combat.evade`
  is also the `False` default. 0x7a has no reactive layer at all, unlike
  0x78.
- Its `stuck_close` escape is manhattan `dist < 16`. This pose is `8+8 = 16`
  **exactly**, so the backstep never fires.

## Heart budget — why cutting 0x7a hits did not help

Six variants, all Clean, all deterministic (trials on identical code are
byte-identical). `engage_slash` is the swing count from the new reason
histogram; `-` is a run from before it existed.

| variant | 0x7a f | 0x7a ok | hits | hearts | engage_slash | outcome |
|---------|--------|---------|------|--------|--------------|---------|
| **baseline** (reverted tree) | 1084 | yes | 6 `0x24_E`×4 | 15→3 | - | reaches 0x78 @3, dies f308 |
| diagonal-contact break | 12000 | no | 4 `0x24_E`×1 | 15→2 | - | 0x7a timeout; patrol looped (95,93)↔(96,93) |
| + off-floor return | 921 | no | 5 | 15→**0** | 113 | dies in 0x7a f921 |
| + evader (`avoid_firing_lines`) | 924 | no | 4 `0x59_E`×4 | 15→**0** | 0 | dies in 0x7a, evader pinned Link at (56,157) |
| + evade box east of the block | 1357 | no | 4 | 15→**0** | 25 | dies in 0x7a f1357 |
| distance-based patrol skip | 925 | no | 6 | 15→**0** | 138 | dies in 0x7a f925 |

The contact break **does** remove the measured damage it targets — `0x24_E`
goes 4 → 1 → 0 as more of the reactive stack is added. It does not help,
because **time in 0x7a, not hits per frame, sets the heart cost.** 0x7a is a
5-wizzrobe beam gallery; every rule that dodges instead of swinging starves
the kill (the engine already says this about `contact_backstep`, rr-gjey),
the fight runs 300–900 frames longer, and the extra `0x59` beams cost more
than the contact hits saved. No variant that reduced hits cleared the room.

Also measured while chasing that: the runs are deterministic, so a tie-break
inside a dodge rule is a *trajectory* choice, not tuning — changing which
axis a break prefers moved the whole fight from the west half of 0x7a to the
east half and back.

**Therefore the honest reading: 0x78 is not survivable at 3 hearts on any
geometry, and 0x7a cannot be made to cost less than ~12 hearts by dodging
better.** The next lever is kill speed or the budget itself, not either
room's stand line.

## Landed this sitting

- `scripts/run_level6_entrance_tf.py`: per-stage **health ledger**
  (`health_in` / `health_out` from a global-frame `HealthLedger` fed through
  `on_frame`), plus `phase` / `reason_counts` / `tail` on the failed-stage
  leftover. This is the table above; it did not exist before.
- `level6/wizzrobe.py`: **records-only** reason histogram and a 30-frame
  action tail on `Level6EastKeyController`. No action is chosen from them;
  a unit test asserts the traced controller returns the same action as an
  untraced one. This is what identified the 12000-frame stall as 11443
  frames of `combat_patrol` in a 1px loop rather than a key-collect hang.
- Nothing else. The six behaviour variants in "Heart budget" were all
  **reverted**; they are recorded so the next sitting does not re-derive
  them, but none is in the tree. No pin was rebuilt and no health was poked.

Tests: `QT_QPA_PLATFORM=offscreen uv run pytest nes/zelda_i/tests/test_level6*.py -q`
→ 126 passed. `uv run pytest nes/zelda_i/tests -q` → 1347 passed, 3
deselected.

## Leftover (stop)

room **0x78**, mode **17** (death / CONTINUE), xy **(186,141)**, TF
**0x00**, keys **0**, bombs **4**, health **0x20** (lo=0), deaths **1**,
bow **0**, arrows **0**, rod **0**. Total 2452f. Death cause f308
`0x59` from **S** `v=(0,-3)` d=6 while `wizzrobe_sidestep`.

## Class split (unchanged; v4–v9 kept)

| tag | xy | west_clear f | class |
|-----|----|--------------|-------|
| v4 | (189,141) | 300 | east-waist **blocked 3/3** |
| v5 | (192,140) | 301 | east-waist |
| v6 | (176,141) | 300 | east-waist |
| v7 | (120,125) | 302 | 0x59 chase |
| v8 | (160,149) | 92 | 0x59 chase |
| v9 | (144,141) | 300 | 0x59 chase **blocked 3/3** |
| `l6_final` | (186,141) | 308 | baseline, **3/3** |

## Next hop

**Blocked on the pin, not on a room.** The only open question worth a
sitting is: *what should the Clean L6 entrance pin be?* That is a decision
about what Clean L6 means — Clean has never reached L6, so this fixture
stands in for a leg that does not exist — and it belongs upstream, not in
this lane. Nothing here rebuilt the pin, poked health, or re-tuned anything
on the back of the finding.

**Parked pending a valid pin** (do not spend a sitting on either; the number
they would be judged by is not real):

- ~~Kill 0x7a faster.~~ The contact pose is a swing that cannot land, and
  aligning onto the target's row before swinging is the untested half of
  that finding. But "hearts out of 0x7a" against a 15-heart nibble is not a
  Clean measurement, so a green here would not be trustworthy.
- ~~Check whether 0x7a is required at all.~~ Its only product is the key
  `level6_west_key_0x78` spends. Worth knowing, but it is a route question
  whose payoff is measured in hearts — same fake denominator.

Still true regardless of the pin, and cheap to re-check once one exists:
nothing on this route banks hearts between 0x7a and 0x78
(`level6_return_0x79` and `level6_west_key_0x78` are both 0-hit, 0-gain).

Also outstanding, and **not** from this lane: `run_survival_spine.py
--through level6-entry` is red at `clear45_key` (L1) while the shared L1 /
`dungeon.engine` / `walk.physics` files are mid-edit elsewhere. Re-derive
the real L6 arrival live once that lane settles; until then the two
continuous tapes above are the evidence.

Do not poke arrows. Do not reopen Survival Gohma. Do not add health pokes or
`--infinite-life` to anything quoted here. Do not rebuild `Level6Entrance`
from this lane. `rr-d6v` stays OPEN. `route_eligible=false`.
