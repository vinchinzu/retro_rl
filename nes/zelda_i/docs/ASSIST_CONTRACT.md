# Assist contract — unlimited health (first-pass routing)

Zelda I uses a **disclosed Survival assist** for first-pass dungeon/overworld
mapping. The goal is reliable navigation, room transitions, keys, bosses, and
route graph evidence—not heart conservation.

Zelda I **segment scripts default to Survival** (`--infinite-life`).
Pass `--no-infinite-life` for Clean. STATUS M5 stays
`run_level1_complete` without the flag (`UnlimitedHealthAssist` off).

Any published assisted result must record this contract path and the assist
telemetry block in the run report.

## Allowed writes

### Unlimited health (hearts)

- During ordinary controllable gameplay (mode 5 overworld/dungeon play,
  underworld passage mode 9, or cave mode 11), restore **filled hearts** for the **accepted** container
  count.
- `$066F` (`HeartValues`): high nibble = containers−1, **low nibble =
  whole hearts**. Full is `lo == hi` (`0x22` = 3/3), plus `$0670`
  (`HeartPartial`) = `$FF`. Official fill (`World_FillHearts`) does
  `INC HeartValues` until `CompareHeartsToContainers` (lo==hi). Writing
  `0xF` in the low nibble makes that fill grant extra containers — that
  is how the L1/L2 tape jumped to 7 then 11 hearts.
- Track accepted containers from the first play frame (start = 3). Allow
  **+1 only** when the game just granted a heart container (`room_item_id`
  `0x1A`) or when leaving Triforce fanfare (mode 18). A fanfare return
  never grants more than one.
- Implementation writes `health_byte_for_containers(accepted)` and
  `heart_partial=$FF`. Do **not** write `(health & 0xF0) | 0x0F`.
- Do **not** increase heart containers except the legal +1 cases above.
  A high-nibble spike is clamped back (telemetry `container_clamps`).
- Observe natural damage before restoring; count writes / restored units.
- Do not revive a completed death transition (mode 17): suspend writes.
- Suspend during boot/menu modes, scroll transitions (6/7/16), and Triforce
  fanfare (mode 18).

The spine's optional `--observed-damage-guard` keeps a requested refill
threshold while raising its effective floor to the largest single-frame hit
Link has survived so far. A two-heart hit from four to two therefore teaches
the guard and refills on that same play frame. This is still a Survival health
write under the rules above; it never writes capacity or progression. Reports
separate `target_refills` (at the requested threshold) from `safety_refills`
(above it), and include `effective_floor`. The guard is reactive: a first hit
that immediately enters death mode cannot be undone. A zero-death run must be
verified on the full tape before lowering its threshold.

The implementation is `zelda_i.assist.UnlimitedHealthAssist`, applied from
`zelda_i.route.chain.run_controller_stage` / probe loops—not scattered policy writes.

### Owned inventory counts (bombs / keys / rupees) — Survival route-development shortcut

Opened 2026-08-15 so the continuous spine can open L2 bomb walls (power-on
entry is bombs=0) and attach Boom → Dodongo → TF without a farm pass. Expanded
by operator direction on 2026-08-23: bomb-count top-ups may be used at verified
bomb gates through the assisted full-game clear while route experience and
reusable skills are still being built and refactored.
The spine applies this at L2 entry, again before `SPINE_BOMB_RETOPUP`
stages, and at the natural L3 Raft boundary before the bomb-heavy boss suffix.
Rupee-count top-ups (L7 Bait 60R, pre-l1 coast pack 20R, gathering Blue
Ring 250R) were **retired on 2026-09-24**. The gather chain now opens the
hidden rupee caves on and beside its walk (`SECRET_RUPEE_CAVES` in
`overworld/locations.py`: 0x2D, 0x28, 0x48, 0x5B, 0x6B, 0x56 before the
ring; 0x62 after it), which pays the 250R ring and the 60R candle from play.
The wallet caps at 255, so the order matters. Nothing writes `$066D`; a
short wallet now fails the buy instead of being topped up.
**Not Clean.** Strip or replace with farms during the later resource pass; do
not treat a top-up tape as natural inventory.

Allowed fields only:

| Field | Address / data key | Rule |
|-------|--------------------|------|
| bombs | `$0658` / `bombs` | Count top-up at a verified route bomb gate, through the assisted full-game clear. Never write `max_bombs` (`$067C`). |
| keys | `$066E` / `keys` | Count top-up of the already-used key item. |
| selected_item | `$0656` / `selected_item` | B-slot select of an **already owned** item (bombs=`1`). |

Every write must be listed in the run report (`inventory_assist` / `poke_bombs`
/ `poke_keys`). `progression_writes` and `capacity_writes` stay 0.

### Bait / Food byte at the L7 shop (operator exception, 2026-09-02)

The natural L6 → bait-shop overworld route is a **mountain-locked desert
pocket** (`0x22/0x32/0x33/0x23/0x24/0x25`) with no southward outlet to the
row-4/5 band that holds pond `0x42` and shop `0x34`; the shop and pond are
only reached walking north out of the western forest band from *start*. Mapping
that route is its own campaign (bead `rr-8t4.4`). Until it lands, the Survival
spine's Bait stage (`level7_bait_purchase`) runs
`SurvivalBaitPurchaseController`: one disclosed write of the owned **Food**
byte in place of the natural 60R buy, so `level7-entry` and L7-B can run.
**Not Clean** — Clean keeps `NaturalBaitPurchaseController` fail-closed.

Food is a **consumable gate item** (spent feeding the Hungry Goriya in L7-B),
not an owned resource count, so it is disclosed here separately from the
bomb/key/rupee top-ups and is **not** written through `apply_owned_inventory`.

Allowed fields only:

| Field | Address / data key | Rule |
|-------|--------------------|------|
| food | `$065D` / `ADDR_FOOD` | Once, at `level7_bait_purchase` on the Survival spine, set to `1` if still 0. Never a rupee, Whistle, door, or TF write. |

The implementation is `zelda_i.dungeon.ops.poke_food`, wired into
`SurvivalBaitPurchaseController`. List the write in the stage
`inventory_assist` / `report()["writes"]`. `progression_writes` and
`capacity_writes` stay 0. The wallet is not topped up for it (retired
2026-09-24). A natural buy from a mapped route replaces the Food write.

### One-room Link position — L6 0x3A stairs (retired, superseded 2026-09-03)

**Retired.** `rr-tne2` (L6 Gohma) is closed. The natural south-band east-column
walk-on (`rr-17co`, `l6_stairs3a_southband` 1/1, `position_writes=0`)
superseded this poke; it is no longer used or authorized. Kept below for
historical record only.

Walk-on of the cleared 0x3A stairs was BLOCKED after six 3-red hops. One
disclosed **Link-position** write was allowed so the same power-on session
could leave 0x3A and continue L6. **Not Clean.** Dest was still RAM.

Allowed fields only:

| Field | Address / data key | Rule |
|-------|--------------------|------|
| link_x | `$0070` / `ADDR_LINK_X` | Once, in play 0x3A after the live center 0x68 push. |
| link_y | `$0084` / `ADDR_LINK_Y` | Same write. Target is the 0x09 analog `(208, 93)`. |

The pair counts as **one** position write (`position_writes=1`). List it in
`position_assist`. Do not write facing, mode, room, doors, inventory,
Triforce, or capacity. Do not load state. `continuous_emulator_session`
stays true. Do not repeat the write on later rooms.

Live follow-up dated 2026-08-27: this exact `(208,93)` target enters cellar
`0x08`. The first controllers climbed its A-side ladder and therefore returned
to play `0x3A` at `(96,157)`. Offline PRG1 ROM + disassembly decoding showed
the pairing is `AttrA[0x08]=0x3A`, `AttrB[0x08]=0x1D`; `InitMode9` spawns on
the left ladder for an A-side source, and `CheckSubroom` selects the B endpoint
when Link exits at `x>=0x80`. The corrected controller keeps this same
authorized target, crosses the tunnel floor to the right ladder at `x=192`,
and reached play `0x1D` 1/1. No second position target or write was added.

The implementation was `zelda_i.assist.poke_link_position`, **deleted
2026-09-07** (zero call sites since the natural walk-on landed). `rr-17co`
walked onto tile `0x71` at `(208,93)` via the south-band east column
(`l6_stairs3a_southband` 1/1, hop 290f) with no retry of occupancy at y=149
(ne71 v1–v3), and both the warp and cellar08 reported `position_writes=0`.
`--through level6-gohma` subsequently went green and `rr-tne2` closed, so
this position exception is retired — do not re-enable
`poke_link_position` on the spine.

### Wooden arrows at L6 Gohma (operator exception, 2026-08-28)

Bow is earned on the Survival L1 splice (`ADDR_BOW=1`). Wooden arrows are
an OW shop item (~80R) not yet on the tape. One disclosed write grants
**wooden arrows** so the same power-on session can kill Gohma. **Not Clean.**
Do not write `ADDR_BOW`. Do not grant silver arrows (`ADDR_ARROWS=2`).

Allowed fields only:

| Field | Address / data key | Rule |
|-------|--------------------|------|
| arrows | `$0659` / `ADDR_ARROWS` | Once, in play `0x1C`, set to `1` (wooden) if still 0. |
| selected_item | `$0656` / `ADDR_SELECTED_ITEM` | B-slot `2` (arrows) of the just-granted item. |

The implementation is `zelda_i.assist.poke_wooden_arrows`. List the write in
the Gohma controller `inventory_assist`. `progression_writes` and
`capacity_writes` stay 0. `bow_writes` stay 0. Natural 80R shop buy replaces
this (`ADDR_ARROWS` is type, not ammo; do not farm ownership). Bow is already
on the L1 Survival splice; splice the buy after `level1-bow-pickup`, before
Gohma. Live `0x5E` has no arrows. The helper already skips when arrows are
wooden.

This exception does not authorize speculative top-ups on every frame. Apply it
immediately before a known bomb-consuming stage, preserve all other inventory,
and record the before/after count and semantic stage name.

Do **not** grant an item Link has not found on this session: sword upgrade,
boomerang / magical boomerang, bow, candle, whistle, raft,
stepladder, book, ring, bracelet, letter, potion, rod, magic key, map,
compass, or triforce bits. Wooden arrows at Gohma `0x1C` and the Food byte at
`level7_bait_purchase` are the disclosed exceptions above; silver arrows stay
forbidden.

## Forbidden writes

- undiscovered inventory items (see table above)
- triforce / dungeon completion bits
- room, screen, door, object, or map state
- Link position, facing, or mode (the one 0x3A `ADDR_LINK_X`/`ADDR_LINK_Y` pair above is retired — see that section)
- heart **containers** (high nibble of `ADDR_HEALTH`)
- bomb **capacity** (`ADDR_MAX_BOMBS`)
- timers / dialog counters
- save-file completion state

If a new write is needed, update this contract before using it.

## Phase guard

| Phase | Mode cues | Assist |
|-------|-----------|--------|
| ordinary_gameplay | mode 5, passage 9, or cave 11, not dead | refill |
| transition | 6 / 7 / 16 | suspend |
| triforce_fanfare | 18 | suspend |
| death | 17 | suspend; count death entry |
| menu_or_boot | 0–4 | suspend |

## Required telemetry

Assisted reports include:

- `assist.enabled`, write count, hearts restored
- first frame assist became active
- **damage taken** (observed filled-heart units before each refill):
  - `total_damage` — cumulative units lost over the run
  - `damage_events` — count of frames with a loss
  - `maximum_single_frame_damage`
  - `damage_by_location` — heatmap keys `L{level}:0x{screen}` (hottest first)
  - `damage_samples` — up to 64 events with frame/xy (debug; totals unbounded)
- death entries
- suspended phase frame counts
- `progression_writes` / `capacity_writes` (must stay 0)
- `accepted_containers` / `container_clamps` (high-nibble spikes must
  clamp; after L1 TF accepted=5, after L2 TF accepted=7 on a first-quest
  start)

Use `damage_by_location` later for **Clean combat harden** priority. First-pass
work stays on pathfinding, doors, keys, bombs, and puzzles — not sword polish.

## Integrity assertions (assisted clear)

- continuous emulator session from declared start
- no state loads after power-on (for natural-entry claims)
- natural inventory / triforce acquisition
- no progression or capacity writes
- natural room and boss advancement (the one disclosed 0x3A Link-position
  write is retired, see that section; room/door/inventory/TF still natural)

## Dual track (Clean vs assisted)

| Track | Stem / flag | Claims |
|-------|-------------|--------|
| Survival-assisted | segment default; `--infinite-life`; `*_assisted` artifacts | first-pass geometry + route graph only |
| Clean | `--no-infinite-life`, or `run_level1_complete` without the flag | may promote STATUS Clean / M5+ |

Do not mix assisted greens into Clean STATUS rows. Prefer SM-style dual-track
stems when both exist.

## CLI

```bash
# First-pass: door path + Moon entry without heart farm
uv run python zelda_i/scripts/probe_level2_suffix.py \
  --infinite-life --enter-dungeon --tag l2_assist

uv run python nes/zelda_i/scripts/run_survival_spine.py \
  --through level2 --no-video --trials 1
```
