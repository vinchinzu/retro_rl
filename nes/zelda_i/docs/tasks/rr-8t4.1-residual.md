# Residual — rr-8t4.1 L7-A pond drain / enter 0x79

**Closed 2026-09-05.** Power-on `--through level7` 2/2. Living residual is
[`rr-6o7.1-residual.md`](rr-6o7.1-residual.md). Archive below.

---

**Spine bead:** `rr-8t4.1` (closed). Acceptance was
power-on `--through level7-entry` from `MEASURED_POST_L6_EXIT`. Do not
STATUS. Do not push unless asked.

## This sitting (2026-09-05) — the pocket is ESCAPED

**`rr-8t4.1` stays open** (drain stair-seek is the last blocker), but the
long-standing "Survival dies before pond `0x42`" wall is GONE. The spine now
reaches OW `0x42` **from power-on** (`recordings/l7entry_warp_t1.json`,
`set_state=0`, TF `0x3F`, 60R).

The route out of the mountain-locked `0x22` pocket is the **Recorder warp**
(H1), not an overland walk — every overland edge is mapped dead. New module
`level7/warp.py`:

```text
0x22 ↓0x32 →0x33 ↑0x23 →0x24        POST_L6_TO_WARP_HOPS (walk)
0x24  blow Recorder facing DOWN      RecorderWarpController -> 0x45 (L4 door)
0x45 ↓0x55 ↓0x65 ←0x64 ↑0x54 ←0x53 ←0x52 ↑0x42    WARP_JOIN_TO_POND_HOPS
```

- Entry chapter stages are now five: `level7_post_l6_overworld` (stops on the
  warp launch screen `0x24` via the new `dest_screen` field) →
  `level7_recorder_warp` → `level7_pond_approach` → `level7_bait_purchase` →
  `level7_pond_drain_entry`.
- **Blow count is screen-checked, never hardcoded.** Recon measured 8 blows
  from its launch position; the live spine needed **7** (landings
  `0x24, 0x24, 0x22, 0x22, 0x0b, 0x0b, 0x45`). A count-locked controller
  would have overshot past `0x45` to `0x74`.
- `0x45→0x55` and `0x55→0x65` both use `align_x=128` (the raft-dock column
  the whirlwind drops Link on). Reusing the stock `align_x=112` on the second
  hop drags Link into the mid-screen house/tree mass — a 30,000f stall.
- `0x74 → 0x64` UP is DEAD (north edge mountain, all 32 tile columns).
- Probe evidence: `scratch/pond/probe_recorder_warp_full_route.py`,
  tags `rw_full_route_t2`/`_t3`, pond `0x42` `(128,221)` at 4913f, 2/2
  byte-identical, `writes=0`. Warp determinism 3/3.
- No pokes: the Recorder is owned from L5 and the Raft from L3, so the warp
  and the dock crossing are natural play. The old recon `ADDR_WHISTLE` poke
  is no longer needed to reach or drain the pond.

## HANDOFF — next step (2026-09-05 end of sitting)

**Done and verified this sitting:** L6 power-on 2/2; L7 **entry** power-on 2/2
(Recorder warp → pond → drain → play `0x79`); `LEVEL7_ENTRY_STOP` promoted to
`spine-green`; drain stairs corrected to `(96,144)` tile `0x70`; new pin
`OW_L7PondNatural`. 771 tests green (level9 lane excluded — another session).
Nothing committed.

**Interior progress (partially verified — read this before trusting it):**
`level7/path.py` gained `Room69WestBombController` +
`L7_ROOM69_WEST_APPROACH`, wired via `make_room69_west_bomb_controller`.
Root cause it encodes: the `69_branch_v2/v3` fixture spawned Link near the
stand, so the naive `TO_STAND` walk never crossed the room; from the live
south-mouth arrival `(120,205)` the dominant-axis walk drives into the
centre-row diamond blocks at `x~96, y~141`. Fix rises the open `x=120`
column to `EAST_BAND_Y`, crosses west on that row, then drops to the stand.
It uses the **existing** `approach_waypoints` field — `dungeon/bomb_wall.py`
was not modified.

Live chain evidence `recordings/candle_chain_v1.json` (from
`OW_L7PondNatural`, pond drain 466f, then the red-candle stage list):

```text
level7_entry_first_door   ok    251f
level7_room69_west_bomb   ok   1851f   <- was the blocker
level7_room68_north       ok    305f
level7_room58_east        ok    349f
level7_room59_up          ok   1460f
level7_room49_up          ok   2111f
level7_room39_left        ok    330f
level7_room38_up          FAIL 8000f   <- new blocker
```

**Caveat:** that agent was stopped mid-task, so the `0x69` fix has this 1/1
chain run plus four diagnostic runs (`room69_west_diag_v6..v9`,
`bombframe_v*`) but **no dedicated 2/2**. Confirm 2/2 before treating it as
settled.

**Next step, in order:**

1. Re-confirm `level7_room69_west_bomb` 2/2 live from `OW_L7PondNatural`.
2. Fix `level7_room38_up`. Its failure tilemap (in `candle_chain_v1.json`,
   key `failed_tilemap`) is a regular block lattice — blocked columns at
   `x = 48, 80, 112, 128, 176` on rows `y = 112, 144, 176`, open rows at
   `y = 96, 128, 160, 192`. It timed out at the full 8000f, so this looks
   like the same fixture-vs-live naive-walk class as `0x69`: route along an
   open row rather than dominant-axis.
3. Re-run power-on `--through level7` and record the new stop point.
4. The complete chapter (`level7_complete_chapter_stages`) is still entirely
   unexercised live, and `LEVEL7_COMPLETE_STOP` / `MEASURED_POST_L7_EXIT`
   stay fail-closed and unfilled — do not fill them from a fixture leftover.

**Environment traps:** long emulator runs get killed by a faulty low-memory
guard (it reads `MemFree`, ~3 GB, not `MemAvailable`, ~50 GB) — launch them
`setsid nohup ... & disown` and poll the log. `--trials 2` is impossible
("Cannot create multiple emulator instances per process"); use `--trials 1`
in separate processes.

### Drain stair-seek — FIXED, and the entry gate is promoted

Stairs are `(96,144)` tile `0x70` (the drained 2x2 quad at `$6530` cols 12-13,
rows 10-11), **not** the old `(96,132)` tile-114 numbers, which came from the
poked-Whistle `OW_L7Pond` recon (drain_v2) via a `PostSwordStart` approach and
never matched a real drained arrival. `STAIR_CANDIDATES` is now the four
corners of the real quad; `BLOW_WAIT_FRAMES=240` is adequate (stairs become
walkable 225f after the 12th B press).

New iteration pin `OW_L7PondNatural` (whistle naturally owned, arrived on
`0x42` `(128,221)` before any blow, zero pokes). **Read its
`acceptance_warning`**: it descends from the `Level6ExitOverworld` fixture, so
its keys/rupees/health are that fixture's, not the measured spine leave. It is
an iteration checkpoint only — power-on acceptance is the spine run.

**Power-on `--through level7-entry` is live 2/2 byte-identical**
(`recordings/l7entry_warp_v4_t0.json`, `l7entry_warp_v5.json`, `set_state=0`):
warp 7 blows / 1982f, drain 466f on stair candidate 0, leftover L7 `0x79`
`(120,205)`, `writes=0`, TF 63, keys 2, bombs 8, 60R.

`LEVEL7_ENTRY_STOP` is therefore **promoted to `evidence="spine-green"`,
`route_eligible=True`** — the exact condition its own comment named ("until a
natural L6-leave drain"). **Survival scope only:** `food >= 1` is still the
disclosed `ADDR_FOOD` poke, so this is not a Clean claim and `docs/STATUS.md`
stays the planner's. Red Candle and complete stops stay fail-closed.

### Superseded blocker (kept for the record): drain stair-seek

`level7_pond_drain_entry` is the only failing stage. The blow **works** — the
pond visibly drains (`recordings/l7entry_warp_t1_final.png` shows the dry bed
and a staircase just LEFT of Link, at roughly his own y) — but all ten
`STAIR_CANDIDATES` miss, because `STAIRS_XY=(96,132)` and its neighbours came
from the poked-Whistle `OW_L7Pond` recon (drain_v2), not from the natural
arrival. Fix is the stair coordinates in `level7/pond.py`, not the route.

Also still true: L6 finishes from power-on **2/2** byte-identical
(`tf=63`, room `0x0c`, keys 2, bombs 8, 42R, `set_state=0`).

Do not STATUS. Do not fill `MEASURED_POST_L7_EXIT`.

## Prior sitting

---

Interior chapters were already composed (`9dac146a`). Survival still dies
before pond `0x42`. Three agents + parent compose:

### Pond drain module (wired)

`level7/pond.py` owns drain + `PostLevel6OverworldController`. Scratch
probes: `scratch/pond/`. Pause-select is `dungeon.pause_select` (shared with
Hungry / Digdogger / L7 bomb walls). Recorder slot 5, no `$0656` poke, 12×B,
stairs `(96,132)` tile 114, dest play `0x79` `(120,205)`. `writes=0`,
`route_eligible=False`. Live 2/2 drain skipped: no `0x42`+whistle=1 pin
(`OW_L7Pond` is whistle=0). Do not poke Whistle.

Composition contract (this sitting): leftover B-slot is a hop input. Pond
leaves recorder=5; first candle-chapter bomb wall (`0x69` west) pause-selects
bombs. Hungry leftover bait=6 → MAP north bomb selects bombs. Digdogger
leftover recorder=5 → `0x0C` east bomb selects bombs. Red Candle success is
the 0→2 rising edge; a pin that already has candle 2 fails closed. Cellar
`0x4A` drops south once then east/north — no y<180 DOWN vs climb oscillation.
`0x49` moat fails closed if `ADDR_LADDER=0`.

`make_pond_entry_controller` now returns that controller. Food poke +
rupee top-up stay.

Same leftover-B / unaccepted-RIGHT copies elsewhere in zelda_i were folded
onto `dungeon.pause_select` (L8 candle select, L8/L9 bomb walls, L9 fixture
and silver-arrow select). L6 rod pickup is a 0→1 rising edge. Next live
boundary is still OW `0x13`.

### Post-L6 overworld (partial)

`0x22` is a boxed mountain graveyard (PNG `l7_p22w7_final.png`). West →
`0x21` Magical Sword grave is **dead** (leftover `(90,165)` tile 38, 41
occupancy misses). North is the L6 cave (UP = mode 16). South corridor
is the only walk-off.

Greened `POST_L6_TO_POND_HOPS` (not the dead `0x25` pocket):

```
0x22 ↓0x32 →0x33 ↑0x23 →0x24 ↑0x14 ←0x13
```

`l7_p14w` 1/1 from `Level6ExitOverworld`, 1698f, leftover **OW `0x13`
`(240,189)`** mode 5, whistle 1, food 0. `_after_hops` succeeds only on
pond `0x42`; the prefix fails closed there (`post_l6_path_exhausted_unmeasured`).

`POST_L6_TO_BAIT_HOPS` `0x22→0x25` is a DEAD SPUR (kept for bait-micro
tests, not the spine default).

### path.py Gut

`path.py` 2813 → 1123. `west.py` 1077 → 864 via `_WestHop` (same file;
no sibling extract). `level9/overworld.py` 734 → 704 after dead pause
phases; still over HYGIENE ~600.
Merged `Room0DClear*` → `stairs0d.py`, `Room4A`/`Room1B`/`Room39` →
`digdogger.py`, `Room38Up` → `hungry.py`, `Room1ACandle` → `cellar.py`.
Public names still import from `zelda_i.level7.path`.

## Measured leave (do not invent)

`MEASURED_POST_L6_EXIT`: OW `0x22` `(112,125)` mode 5, TF `0x3F`, keys 2,
bombs 8, rupees 42, Rod/Bow/Whistle 1, Food 0, Candle 0, 8 HC full.

## Next live boundary

**Superseded 2026-09-05.** The old boundary (OW `0x13` west/south toward the
forest band) is retired: there is no overland outlet from the pocket, and the
Recorder warp replaces the whole search. `0x13` / `0x12` / `0x14` are no
longer on the spine at all — the walk stops at `0x24` and warps.

The live boundary is now the **drain stair-seek on `0x42`**: the pond drains,
the stairs are visible, `STAIR_CANDIDATES` looks in the wrong place.

Do not fill `MEASURED_POST_L7_EXIT` from the TF-0 fixture leftover.

## Integrity

deaths 0, progression_writes 0, capacity_writes 0, position_writes 0,
whistle poke 0. Food poke allowed (Survival, disclosed). The Recorder warp
and the `0x45→0x55` dock crossing use only naturally-owned items (Recorder
from L5, Raft from L3) — no new assist.
