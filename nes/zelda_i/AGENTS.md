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
```

## Layout (pointers)

| Path | Role |
|------|------|
| `ram.py`, `overworld.py`, `overworld_nav.py` | Snapshots + OW graph / L1 path |
| `ow_path.py` | Shared `OverworldPathController` (L2–L8 hop engine) |
| `level1.py`, `level1_finish.py`, `level1_dungeon.py` | L1 combat / finish / rooms |
| `dungeon.py` | Shared dungeon combat engine + registry |
| `level2_dungeon.py`, `level2_overworld.py` | L2 rooms + OW approach |
| `level3_dungeon.py`–`level6_dungeon.py` | Later-level room specs |
| `level*_overworld.py` | Path tables + thin subclasses of `ow_path` |
| `chain.py`, `routes.py` | Post-Triforce + named routes |
| `nav_common.py`, `room_timer.py`, `dungeon_lab.py` | Shared nav + lab |
| `assist.py` | Survival infinite-life (opt-in) |
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
bd ready -l zelda_i   # tip: L3 Raft→Manhandla→TF (assist); Clean deferred
```

| Order | Bead | Work |
|------:|------|------|
| ✓ | **rr-n5i** / **rr-5dk** | L2 Dodongo + TF `0x02` assisted LIVE |
| ✓ | **rr-rnx** / **rr-ci7** | Post-L2 OW → L3 enter **2/2 assisted** |
| ✓ | **L3 Raft runner** | `run_level3_raft.py` **2/2 assisted** → `Level3Raft` |
| 1 | **L3 tip residual** | Manhandla → TF `0x04` from `Level3Raft` (`rr-vpl`) |
| later | **rr-4oz** | Clean residual after full-game assist pass |

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

**L3 Raft → Manhandla path (assisted recon):** exit 0x0f→0x69 → UP 0x59 →
**BOMB_RIGHT** 0x5a (walk sealed) → 0x5b → BOMB_RIGHT 0x5c → clear →
RIGHT@y141 → 0x5d → UP **0x4d** Manhandla type **`0x3c`** candidate. TF `0x04`
not yet. Evidence: `l3_manhandla_recon.json`. Probe:
`probe_level3_manhandla.py --infinite-life --tag l3_manhandla`.

**Next tip:** Stabilize 0x5d→0x4d gate + bomb Manhandla + TF `0x04`. Not Clean.

**Traps (L2→L3 OW):** 0x4c east only **y∈[133,145]** (y=149 solid); 0x5c maze
reverse needs denser channel waypoints (no y_band on 0x5b hop); 0x64 west
band **y≈125–150**.

**Traps (L3 Raft / boss):** 0x5a key door long y=141 push; 0x59 DOWN lag after
clear; DOWN push = x≈120 hold; 0x69 stairs **only y≈141**; passage channel
**x≈176** then LEFT on y≈141 to **x≈136**; **0x59 walk-RIGHT sealed post-Raft**
(bomb reopen); 0x5c RIGHT **y≈141 after clear**; type **0x2b** invuln ≠ boss.

Checkpoints: `Level2Boom` / `Level2Complete` / `Level2ExitOverworld` /
`Level3Entrance` / `Level3WestKey` / `Level3Darknuts` / `Level3Raft`.
Use `--infinite-life` for first-pass; Clean STATUS only after full-game assist.
