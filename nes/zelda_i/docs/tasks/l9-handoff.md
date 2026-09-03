# L9 fixture-live handoff — Death Mountain

## Poked-start overworld continuation — 2026-09-03

No STATUS promotion and no Clean/natural-entry claim.  The source
`Level9OverworldReconFixture` has a disclosed full-loadout poke and is
`fixture_only=true`, `natural_entry=false`, `route_eligible=false`.
`Level9RockHopsPartialReconFixture` is the live-walked checkpoint at `0x27`.

### Live route to Spectacle Rock

Luna sub-agents verified the following route under Survival health refill:

```text
0x77 →R→ 0x78
  →U x=48→ 0x68 →U x=48→ 0x58
  →0x58 bush waypoints, U x=112→ 0x48
  →U x=128→ 0x38 →U x=48→ 0x28
  →L y≈102→ 0x27
  →DOWN y≈133, LEFT x≈144, UP→ 0x17
  →UP y≈133, LEFT x=64, UP with Raft→ 0x07
  →UP y≈141, LEFT→ 0x06
  →realign y≈141, LEFT→ 0x05
```

The final run settled overworld screen `0x05` at `(240,141)` with TF
`0xFF`, Magic Key 1, bombs 16, and Raft 1 unchanged.  Deaths were 0;
runtime inventory, progression, and capacity writes were all 0.  Evidence:

- `recordings/l9_start_luna_27_geom_v2.json` (`0x27→0x17`)
- `recordings/l9_start_luna_07_v3.json` (`0x17→0x07`)
- `recordings/l9_start_luna_06_v1.json` (`0x07→0x06`)
- `recordings/l9_start_luna_05_v2.json` (settled `0x05`)

The old “full-width `0x27` north edge is solid” conclusion is dead.  That
probe was trapped in the east pocket behind the mountain.  Moving down
around the mountain tip exposes the central north mouth at `x≈144`.
Likewise, the first `0x06` west attempt was displaced to blocked `y=173`;
realigning to the open `y≈141` corridor completed `0x06→0x05`.

### Exact resume point

The bomb-and-enter trial did not run: the Luna lane encountered repeated
service 404s, not an emulator or geometry failure.  Resume by replaying the
verified path to `0x05`, inspecting the rock screenshot, selecting bombs by
normal pause input, placing exactly one bomb below the **left** Spectacle
Rock, and entering the mouth.  Stop at the first settled dungeon room,
expected L9 `0x76`; do not move north into the Old Man room in the same
trial.  Record bomb/selection deltas and require deaths 0,
`progression_writes=0`, `capacity_writes=0`.

No packaged Level 9 waypoint controller was landed in this sitting.  The
natural-entry controllers remain correctly fail-closed, and the existing
fixture-live interior/credits suffix must not be presented as a continuous
run from this poked overworld start.

## Earlier route design handoff

chapter id and evidence label
  rr-sz8.5 L9-A topology + natural entry — hypothesis (route_eligible=false)
  public credits adapter — fixture-live policies, still route_eligible=false
  Did not STATUS-promote. Did not close rr-sz8.3/.6/.7.

selected natural route (room sequence) vs fixture suffix join
  Magical Key minimum; Red Ring 0x07 excluded.
  prefix (hypothesis): 0x76 → 0x66 Old Man TF → 0x65 bomb-N → 0x55 Lanmola
    → cellar 0x60 → 0x14 → 0x15 → 0x16 skip Patra → 0x06 bomb-W → 0x05
    → cellar 0x70 → 0x63 → 0x62 (8 Keese corridor) → 0x61 stairs
    → cellar 0x75 → 0x20 bomb-N → 0x10 Silver Arrows (ADDR_ARROWS==2)
  join into proven fixture suffix at 0x41:
    0x10 → 0x20 → 0x61 → 0x51 → 0x41 → 0x31 bomb-W → 0x30
    → cellar 0x67 → 0x04 bomb-W → 0x03 → cellar 0x77 left → 0x52
  requires_51_to_41=True (selected). Dest walk 0x51 north still unverified
  (statue diamond). Do not spend a sitting on rr-yxy6 until this prefix is live.

predecessor / required inventory
  Incoming (when L8 finishes, not now): TF exactly 0xFF, Magic Key owned,
  bombs natural/declared, Bow owned. Post-L8 OW leftover UNMEASURED
  (do not use start-0x77 LEVEL9_ROCK_HOPS as the cumulative leave).
  Controllers refuse without TF 0xFF / bombs; never write TF, bomb capacity,
  rooms, or doors.

internal stages
  level9-entry: level9_post_l8_overworld, level9_spectacle_rock_bomb,
    level9_old_man_tf_gate (one-frame fail-closed)
  level9-silver-arrows: level9_natural_silver_arrows
    (missing: silver_arrow_room_0x10_unobserved)
  level9-patra: level9_natural_patra_join
    (missing: 0x51_north_dest_walk_unverified_statue_diamond)
  level9-credits: select arrows (pause cursor only) → Patra → Ganon →
    Power TF → Zelda → credits; load no fixture

endpoint predicates
  level9-entry: play L9 room 0x76, TF 0xFF, Magic Key, bombs > 0,
    no room/door/TF write
  level9-silver-arrows: ADDR_ARROWS==2, Bow owned, TF 0xFF, room 0x10 hyp
  level9-patra: live uncleared Patra 0x52 (body 0x47, 8 eyes 0x25, north closed)
    from natural prefix (not fixture)
  level9-credits: credits/final-page, deaths 0, progression_writes=0,
    capacity_writes=0, inventory_writes=0

dead beliefs (0x62, 0x51 if excluded)
  0x62 is NOT south neighbor of Patra 0x52 (ROM N/S walls; live 8 Keese W/E).
    It remains the Magical Key 8-Keese corridor 0x63↔0x62↔0x61 only.
    No 0x62↔0x52 edge on the natural graph. Not on the suffix join.
  0x51 is NOT excluded: selected route requires 0x51→0x41. Live dest walk is
    still NO (statue diamond). Fail-closed until that walk is earned.

fixture provenance route_eligible=false
  Every *ReconFixture stays route_eligible=false. Natural graph name
  level_9_natural_hypothesis. SELECTED_NATURAL_ROUTE.evidence=hypothesis.
  Start-based LEVEL9_ROCK_HOPS labeled fixture-live, not the L8 leftover.

files changed
  nes/zelda_i/level9/{dungeon,natural_path,hops,overworld}.py
  nes/zelda_i/door_graph/{level9_exits.py,__init__.py}
  nes/zelda_i/docs/LEVEL9_ROUTE.md
  nes/zelda_i/tests/{test_level9.py,test_door_graph.py}
  nes/zelda_i/docs/tasks/l9-handoff.md
  Did not edit STATUS.md, .beads, spine/survival.py, level6/7/8, stair_*.

stitch notes: needs L8 TF 0xFF + Magic Key + measured post-L8 OW leftover
  level9-credits is callable only after live uncleared Patra 0x52 with the
  exact census; it loads no fixture and performs zero inventory/room/door
  writes. Ganon fails closed unless arrows are selected via pause-menu cursor,
  never ADDR_SELECTED_ITEM assign. Public through names unchanged:
  level9-entry, level9-silver-arrows, level9-patra, level9-credits.
