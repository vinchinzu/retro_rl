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
