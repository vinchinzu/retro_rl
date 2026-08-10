# Agent Instructions — zelda_i

NES Legend of Zelda (graph nav; **M5** Clean power-on → Level 1 Triforce).
Shared: `retro_harness.adventure`, `retro_harness.nes`. Docs: `docs/STATUS.md`,
`docs/plan.md`, `docs/LEVEL1_ROUTE.md`, `docs/LEVEL2_ROUTE.md`,
`docs/DUNGEON_LAB.md`, `docs/ASSIST_CONTRACT.md`, `docs/tasks/PROCESS.md`.

Work tracker: **`bd ready -l zelda_i`** (prefix `rr-`).

## Commands

```bash
uv run python zelda_i/scripts/setup_rom.py
uv run python zelda_i/scripts/boot_probe.py

# Natural-entry chain (power-on → sword → L1 complete) — Clean default
uv run python zelda_i/scripts/run_sword_cave.py --natural-entry
uv run python zelda_i/scripts/run_to_level1.py --natural-entry
uv run python zelda_i/scripts/run_level1_complete.py --natural-entry --trials 2
uv run python zelda_i/scripts/run_to_level2_prefix.py --from-heart --trials 2

# Clean heart-safe door path 0x4A→0x3C (no assist; farm + 0x5A clear)
uv run python zelda_i/scripts/probe_level2_suffix.py --from-state At4A --tag l2_clean_at4a_t0

# First-pass Survival assist (infinite life; not Clean STATUS)
uv run python zelda_i/scripts/probe_level2_suffix.py --infinite-life --enter-dungeon
uv run python zelda_i/scripts/run_to_level2_prefix.py --infinite-life --trials 1

# Room timing / lab
uv run python zelda_i/scripts/probe_room_timer.py self-check
uv run python zelda_i/scripts/dungeon_lab.py --help
uv run pytest zelda_i/tests retro_harness/adventure/tests -q

# TAS button movies (non-glitch all-items default) — docs/TAS_ADAPT.md
uv run python -m zelda_i.tas.fetch_refs
uv run python -m zelda_i.tas.import_fm2 --summary-only
# uv run python -m zelda_i.tas.fetch_refs --include-glitched  # any% etc.

# L4 entry (assisted; not Clean STATUS)
uv run python zelda_i/scripts/run_level4_entry.py --infinite-life --trials 2 --save-state

# L4 interior pure room segments (live IDs; not Clean STATUS promote)
uv run python zelda_i/scripts/run_level4_rooms.py --segment entry_up --trials 2
uv run python zelda_i/scripts/run_level4_rooms.py --segment clear_61 --trials 2 --save-state
uv run python zelda_i/scripts/run_level4_rooms.py --segment chain_to_key --trials 2 --save-state
uv run python zelda_i/scripts/run_level4_rooms.py --segment clear_50 --trials 2 --save-state
uv run python zelda_i/scripts/run_level4_rooms.py --segment key_right_62 --trials 2 --save-state
uv run python zelda_i/scripts/run_level4_rooms.py --segment clear_62 --trials 2 --save-state
uv run python zelda_i/scripts/run_level4_rooms.py --segment compass_62 --trials 2 --save-state
```

## Layout (pointers)

| Path | Role |
|------|------|
| `anchors.py` | Canonical L3–L9 door/entry/TF constants |
| `ram.py`, `overworld.py`, `overworld_nav.py` | Snapshots + OW graph / L1 path |
| `ow_path.py` | Shared `OverworldPathController` (L2–L8 hop engine) |
| `level1.py`, `level1_finish.py`, `level1_dungeon.py` | L1 combat / finish / rooms |
| `dungeon.py` | Shared dungeon combat engine + registry |
| `level*_dungeon.py` | **Room specs + stop predicates only** |
| `bomb_wall_path.py`, `level2_bomb_path.py` | Parameterized bomb-wall (`make_*` factories) |
| `level3_path.py` | L3 door micros / west-key / north chain (no raft re-export) |
| `level3_raft_path.py` | L3 Raft path (canonical; shim via `level3_dungeon`) |
| `level*_boss_*.py`, `dungeon_ops.py` | Boss chains + shared door/clear ops |
| `level2_puzzles.py` | BombWall / KeyDoor geometry catalog |
| `door_graph/` | Door topology (stands from puzzles) |
| `level*_overworld.py` | Hop tables + thin `ow_path` subclasses |
| `runner.py` | Shared script env/assist/report helpers |
| `chain.py`, `routes.py` | Post-Triforce + named routes |
| `assist.py` | Survival infinite-life (opt-in) |
| `docs/HYGIENE.md` | Architecture rules (do not re-expand phase machines) |
| `docs/tasks/PROCESS.md` | Dual-track + bead grain |

## Dual track

- **Clean** (default): STATUS-eligible; no health writes.
- **Assisted first pass** (`--infinite-life`): infinite hearts + damage heatmap
  (`total_damage`, `damage_by_location`); map path / puzzles / doors first.
  Do not promote as Clean. Contract: `docs/ASSIST_CONTRACT.md`.

**Agent order:** pathfinding + puzzles → assisted full clear → Clean harden
from damage heatmaps. Do not block tip progress on combat polish.

## Traps (burned once)

- Sword cave door is **NW** of spawn on 0x77 (not north edge 0x67). Cave = mode **11**.
- Sword pickup: x≈120 then UP; idle ~280f through dialog.
- After cave exit ~(64,77): **DOWN first**. Level 1 path:
  `0x77→E→78→N→68→N→58→N@x112→48→N→38→W→37`, door UP @x≈112.
- L1 entry room **0x73** north locked → east **0x74** first key. Keese HP=0 while
  alive (type-only liveness). Room 0x53 key fixed at **(128,109)**.
- After triforce: idle mode 18 (~704f) → OW 0x37; do not reload
  `Level1Complete` mid-fanfare. L2 prefix: `37→38→48→58→59→49→4A`; never 0x79.
- Lab checkpoints are dev fixtures until natural-entry runner passes same spec.

## Next

```bash
bd ready -l zelda_i   # tip leaf: rr-o0nn post-Compass→ladder; parallel: rr-38p
```

| Order | Bead | Work |
|------:|------|------|
| ✓ | **rr-n5i** / **rr-5dk** | L2 Dodongo + TF `0x02` assisted LIVE |
| ✓ | **rr-rnx** / **rr-ci7** | Post-L2 OW → L3 enter **2/2 assisted** |
| ✓ | **L3 Raft runner** | `run_level3_raft.py` **2/2 assisted** → `Level3Raft` |
| ✓ | **rr-vpl** / **rr-wmv** | Manhandla + TF `0x04` **2/2 assisted** from `Level3Raft` |
| ✓ | **rr-k0w** | L4 planning scaffold (`level4_overworld`, plan-only probe) |
| ✓ | **`rr-0fx`** | L4 live entry: dock **0x55** → island **0x45** → room **0x71** **2/2 assist** |
| **1 TIP** | **`rr-o0nn`** / **`rr-5lu`** | L4 post-Compass: component closed; ADDR_LADDER residual |
| free | **`rr-38p`** | Early OW white sword / candle / bombs (parallel) |
| later | **`rr-4oz`** | Clean residual after full-game assist pass |

Full board: `docs/tasks/QUEUE.md`. Routes: `LEVEL2_ROUTE.md`, `LEVEL3_ROUTE.md`.

**L2 complete (assisted LIVE):** `Level2Boom` → … → Dodongo → LEFT `0x0d`
south-band TF → `tf&0x02`. Evidence: `l2_complete_assisted.json`;
checkpoints **`Level2Complete`**, **`Level2ExitOverworld`**.

**Post-L2 → L3 enter (assisted LIVE, rr-rnx):** settle OW **0x3C** → reverse
door corridor + reverse 0x5C maze → west forest → door **0x74** → room **0x7c**.
Evidence: `l2_to_l3_assisted.json`. Runner:
`run_l2_to_l3.py --infinite-life --from-state Level2ExitOverworld`.

**L3 Darknuts → Raft (assisted LIVE 2/2):** from `Level3Darknuts`
LEFT→0x5a → LEFT KEY y≈141 → clear 0x59 DOWN → clear 0x69 RIGHT@y141 →
0x0f channel x≈176 → Raft (`ADDR_RAFT`). Evidence:
`level3_raft_assisted.json` (~6448f); checkpoint **`Level3Raft`**. Runner:
`run_level3_raft.py --infinite-life --trials 2 --save-state`.

**L3 Raft → Manhandla → TF (assisted LIVE 2/2, rr-vpl):** exit 0x0f→0x69 →
UP 0x59 → **BOMB_RIGHT** 0x5a → 0x5b → BOMB_RIGHT 0x5c → clear raw=3 →
RIGHT@y141 → 0x5d clear Zol/Gel/Keese (slots **1–12**, ignore 0x2b) → doors
raw=10 → UP **0x4d** Manhandla **`0x3c`** bomb kill → HC → UP **0x3d** TF
`0x04`. Evidence: `level3_to_boss_assisted.json` (~21653f). Runner:
`run_level3_to_boss.py --infinite-life --trials 2 --save-state`. Checkpoints
**`Level3Boss`**, **`Level3Complete`**. **Not Clean STATUS.**

**L4 entry (assisted LIVE 2/2, rr-0fx):** `Level3Complete` settle OW **0x74**
→ `0x73→0x63 E@y≈149→0x64→0x65→dock 0x55` Raft N → island **0x45** door UP
→ room **0x71**. Evidence: `l4_entry_recon.json` (~2173f). Runner:
`run_level4_entry.py --infinite-life --trials 2 --save-state`. Checkpoints
**`Level3ExitOverworld`**, **`OW_L4Dock`**, **`Level4Entrance`**. **Not Clean STATUS.**

**L4 interior first rooms (pure LIVE 2/2, rr-5lu children 2026-08-10):** from
`Level4Entrance` room **0x71** (empty) UP → **0x61** 3× Vire `0x12` (split
`0x1c` slots 10–12) → **BOMB_UP** @(120,105) → **0x51** 8× Keese `0x1b` +
key `0x19` (pickup ~136,149) → LEFT @y141 → **0x50** 5× Vire **dead-end**
pocket; progress is **KEY-RIGHT** @y141 from 0x61 → **0x62** 5× Vire +
Compass `0x16` dark maze pure (~471f) + return. Module: `level4_dungeon.py`.
Runner: `run_level4_rooms.py`. Evidence includes
`l4_compass62_pure_compass_62.json`. Closed: `rr-zchy` / `rr-yr77` /
`rr-h278` / `rr-wqdu` / `rr-2ysf` / **`rr-9so0`**. Post-compass component
closed (rr-o0nn recon). **Not Clean STATUS.**

**Next tip:** **`rr-o0nn`** post-Compass → `ADDR_LADDER` (parent **`rr-5lu`**).
Live: closed component `{0x71,0x61,0x51,0x50,0x62}`; 0x51 UP/RIGHT sealed;
need room id **outside** that set. Epic `rr-q3n`; parallel OW `rr-38p`. Clean
residual deferred.

**Traps (L4 OW entry):** 0x63 east only **y∈[145,155]** (y=141 bush stick);
dock 0x55 raft only **x≈128**; free 0x73 east edge before UP.

**Traps (L2→L3 OW):** 0x4c east only **y∈[133,145]** (y=149 solid); 0x5c maze
reverse needs denser channel waypoints (no y_band on 0x5b hop); 0x64 west
band **y≈125–150**.

**Traps (L3 Raft / boss):** 0x5a key door long y=141 push; 0x59 DOWN lag after
clear; DOWN push = x≈120 hold; 0x69 stairs **only y≈141**; passage channel
**x≈176** then LEFT on y≈141 to **x≈136**; **0x59 walk-RIGHT sealed post-Raft**
(bomb reopen); 0x5c need **raw=3** (raw=1 false-clear seals RIGHT); 0x5d gel
in **slot 11** seals UP until clear slots 1–12; type **0x2b** invuln ≠ boss;
TF is **0x3d UP of boss** (not east).

Checkpoints: `Level2Boom` / `Level2Complete` / `Level2ExitOverworld` /
`Level3Entrance` / `Level3WestKey` / `Level3Darknuts` / `Level3Raft` /
`Level3Boss` / `Level3Complete` / `Level3ExitOverworld` / `OW_L4Dock` /
`Level4Entrance`.
Use `--infinite-life` for first-pass; Clean STATUS only after full-game assist.
