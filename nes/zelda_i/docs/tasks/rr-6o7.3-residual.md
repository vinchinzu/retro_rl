# Residual — rr-6o7.3 L8-C four-head Gleeok → TF 0x80 → post-L8 OW leave

**Spine bead:** `rr-6o7.3`. Public target `--through level8`. Internal stages:
the ordered Gleeok suffix (`level8/suffix.py`). Do not STATUS. Do not push
unless asked.

## 2026-09-05 — `--through level8` power-on GREEN

`rr-6o7.2` (`--through level8-magic-key`) went power-on **2/2** this sitting;
the Gleeok suffix then composed with **almost no iteration** because every hop
was already fixture-live 2/2 and the power-on inventory (bombs 15, hc 9, MK 1)
was compatible.

### What landed

**`scripts/level8_clear_lab.py`** — the rr-6o7.3 iteration harness.  `--pin`
builds `Level8SuffixEntryLive` (a power-on `--through level8-magic-key` stop
at play `0x1F` `(96,157)`); the bare run drives the ordered
`LEVEL8_SUFFIX_GATES` stage-by-stage with a per-stage glance, then holds the
OW leave.

**`LEVEL8_SUFFIX_GATES` completed** (`level8/suffix.py`) — was 8 gates
starting at `0x3E`; prepended the 3 missing hops
(`level8_return_passage_{west_1f,south_1e,south_2e}` = `0x1F→0x1E→0x2E→0x3E`,
`make_{west_1f,south_1e,south_2e}_controller` from `level8/path.py`) and
appended a `level8_ow_leave_settle` epilogue.  Now 12 rows, `0x1F → 0x2C →
OW 0x6D`.

**`Level8OWLeaveController`** (`level8/triforce.py`, new) — the epilogue.
`Level8Shard2CController` stops at the shard fanfare (mode 18), but
`level8_clear_stop` needs the settled OW leave.  This controller presses
nothing and succeeds after 30 consecutive frames on OW `0x6D` play; fails
closed on any other OW screen.

**`MEASURED_LEVEL8_CLEAR`** (`level8/dungeon.py`) — `Level8ClearEndpoint(
level=0, screen=0x6D, mode=5, incoming_hc=9, outgoing_hc=10,
route_eligible=True)`.

**`NATURAL_LINEAGE_LEVEL8_SUFFIX`** (`level8/suffix.py`) — composable
(`route_eligible` + `natural_predecessor`).  `continue_level8_spine` defaults
flipped: `clear_endpoint=MEASURED_LEVEL8_CLEAR`,
`suffix=NATURAL_LINEAGE_LEVEL8_SUFFIX`.  Bare `l8_hops()` stays fail-closed
(`FIXTURE_LINEAGE` / `UNOBSERVED`), mirroring the topology / burn-target
split.

**`MEASURED_POST_L8_HANDOFF`** (`level9/dungeon.py`) — the L9 predecessor:
OW `0x6D` `(96,93)` mode 5, TF `0xFF`, MK 1, keys 1, bombs 14, hc 10,
B = bombs, bow+arrows 1.  `PostLevel8Handoff.mismatch()` only gates on
screen/level/mode/TF/bombs>0 (rupee & drop pickups vary).

**L9 seam wired** — `continue_level9_spine` (carrying
`MEASURED_POST_L8_HANDOFF`) attached in `spine/survival.py`;
`L9_THROUGH` added to `SPINE_THROUGH`.  Every natural L9 chapter is still a
fail-closed marker (see the frontier below).

### Power-on `--through level8` (`l8clr_poweron_v2`, `set_state=0`, first quest)

```
level8_return_passage_west_1f     ok  275f  -> 0x1e (208,141)
level8_return_passage_south_1e    ok  264f  -> 0x2e (120,77)
level8_return_passage_south_2e    ok  255f  -> 0x3e (120,93)
level8_return_passage_east_3e     ok  329f  -> 0x3f (32,141)
level8_return_passage_stairs_3f   ok  ~360f -> cellar 0x2f (208,141) m9
level8_return_passage_cellar_2f_settle  ok  600f -> (192,93)
level8_return_passage_cross_2f    ok  ~436f -> 0x4c (112,125)
level8_return_passage_bomb_north_4c ok ~848f -> 0x3c (120,189)  (1 bomb)
level8_four_head_gleeok           ok  6953f -> 0x3c (32,181)  hc 9->10, 0x46 seen
level8_heart_shard_leave_2c       ok  43f   -> 0x2c (120,149) m18 TF 0xFF
level8_ow_leave_settle            ok  567f  -> OW 0x6d (96,93) m5 TF 0xFF
final: OW 0x6D (96,93) m5  keys 1  bombs 14  rupees 56  hc 10  TF 0xFF  MK 1
```

deaths 0, progression_writes 0, capacity_writes 0.  `inventory_assist` is
`SPINE_L8_RETOPUP` only (bombs->16 / keys->2 count top-up, ASSIST_CONTRACT).
`level8_clear_lab.py` was 2/2 byte-identical from the pin (the pose settled
`(192,157)` there under a raw 2000f idle vs `(96,93)` from the
`level8_ow_leave_settle` 30-consecutive-frame stop -- the stage's value is
canonical; `(96,93)` matches the old `Level8PostShardOWReconFixture`).

Suite **856 passed**.

## Frontier — L9 natural entry + interior (the big remaining lift)

`--through level8` is green; `--through level9-*` all fail on their first
frame.  Every natural L9 chapter is a `NaturalRouteUnavailableController`:

- **`level9_post_l8_overworld`** — needs a real controller to walk OW `0x6D`
  `(96,93)` → Spectacle Rock `0x05`.  `LEVEL9_ROCK_HOPS` (`level9/overworld.py`)
  is the walk from `0x78` (one east of start `0x77`); the `0x6D → 0x78`
  connector is unmapped.  From `0x6D` Link goes UP to `0x5D`, then W/N
  through the L2/L8 maze area to `0x77`/`0x78`.
- **`level9_spectacle_rock_bomb`** — the bomb-the-left-rock entry is
  fixture-live (`Level9EntranceReconFixture` was built this way;
  `level9/overworld.py` `FixtureEntryPhase`), but never from a walked pose.
- **`level9_old_man_tf_gate`** — `0x76` UP through the full-TF gate → `0x66`.
  Fixture-live dest hop exists (`level9/prefix.py` `Level9North76Controller`,
  P1/P2 2/2) but the Old Man gate acceptance is unobserved.
- **Interior** — all 16 prefix hops `0x76 → 0x10` are fixture-live 2/2
  (`level9/prefix.py`, `docs/LEVEL9_ROUTE.md` `rr-sz8.6`) but on *composed*
  fixture inventory; the natural Patra join `0x10 → 0x52` + Ganon + Zelda +
  credits runs 26,109 continuous frames from `Level9EntranceReconFixture`
  (`rr-sz8.7`).  None of it is wired into `l9_hops` (which uses the
  fail-closed `Natural*` controllers).

Ordered next steps:
1. Recon the `0x6D → 0x05` overworld walk from the measured
   `MEASURED_POST_L8_HANDOFF` pose; build `Level9PostL8OverworldController`.
2. Promote the Spectacle Rock bomb entry from a walked pose → live L9 `0x76`.
3. Old Man `0x66` full-TF gate acceptance.
4. Compose the 16 prefix hop controllers on natural resources; wire `l9_hops`.
5. Natural Patra join + the existing `level9_credits_chapter`.

### 2026-09-06 Recon Progress — Post-L8 OW Walk & Screen 0x05 Decoded

- **Pinned Post-L8 Leave**: `Level8OWLeaveLive.state` captured and verified:
  OW `0x6D (96,93)` mode 5, TF `0xFF`, 14 bombs, Magic Key 1, 10 HC, B=bombs.
- **Reverse OW Route Decoded (6D → 58)**:
  - `0x6D`: Walk LEFT to `x=48`, then UP to `0x5D` (lands at `(48, 221)`).
  - `0x5D`: Align `y=132`, walk LEFT into `0x5C` (lands at `(240, 133)`).
  - `0x5C`: Reverse maze waypoints (`(240,133) → (192,132) → (192,92) → (16,92)`) into `0x5B` (lands at `(240, 93)`).
  - `0x5B`: Highway row 4/5 (`y=93`) is open across all cols; walk LEFT into `0x5A` (lands at `(240, 93)`).
  - `0x5A`: Walk DOWN to `y=140`, walk LEFT into `0x59` (lands at `(240, 141)`).
  - `0x59`: Walk DOWN to `y=155`, walk LEFT into `0x58` (lands at `(240, 141)`).
  - `0x58`: Seamlessly joins the verified `LEVEL9_ROCK_HOPS` corridor (`0x58 → 0x48 → 0x38 → 0x28 → 0x27 → 0x17 → 0x07 → 0x06 → 0x05`).
- **Screen 0x05 (Spectacle Rock) WRAM Tilemap Measured**:
  - Col 30 (`x=240`) has a rock wall at `y=128..136` (`d4 d6`), so Link cannot walk straight up upon entering `0x05`.
  - Cols 26-27 (`x=216`) are clear (`tile 0x26`) from `y=141` to `y=96`.
  - Path: Step LEFT to `x=216`, UP to `y=93`, LEFT through gap to `(120, 93)`, DOWN to `(120, 173)`, LEFT to `(72, 173)`.
  - Place bomb on B (consumes 1 bomb), wait blast, walk UP into cave mouth at `x=72` → live Level 9 room `0x76`!

## Integrity

deaths 0, progression_writes 0, capacity_writes 0, position_writes 0,
tf_poke 0.  Inventory assist: `SPINE_L8_RETOPUP` count top-up only
(ASSIST_CONTRACT), listed in every run report's `inventory_assist`.
