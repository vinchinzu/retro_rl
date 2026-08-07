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
bd ready -l zelda_i   # tip: rr-n5i TF residual after Dodongo → rr-5dk
```

| Order | Bead | Work |
|------:|------|------|
| ✓ | **rr-lzk** | 0x6f bomb N @(120,101) → 0x5f — **2/2 Clean** |
| ✓ | **rr-etl** | 0x5e Goriya pure — **2/2 Clean** |
| ✓ | **rr-fvt** / **rr-cjf** | 0x5f policy + bomb-UP → **0x4f** boom path LIVE |
| ✓ | **rr-bsq** / **rr-ebe** | 0x4f Magical Boomerang — **2/2 Clean** |
| ◐ | **rr-n5i** | path → Dodongo 0x0e + HC LIVE; **TF 0x02 residual** |
| 3 | **rr-5dk** | Natural-entry assisted power-on → TF 0x02 |

Full board: `docs/tasks/QUEUE.md`. Routes: `docs/LEVEL2_ROUTE.md`.

**Dodongo path (assisted LIVE):** `Level2Boom` → bomb-N 0x3f → LEFT Moldorm →
UP 0x2e clear → UP 0x1e Goriya clear → **bomb-N @(120,101)** → **0x0e** type
`0x32` bomb-mouth → HC. **Trap:** walk-UP on 0x1e after clear is solid
(doors=12 red herring). Post-kill doors LEFT-only; RIGHT/TF residual.
Runner: `run_level2_dodongo.py --infinite-life`. Checkpoint: `Level2_0E`.

Checkpoints: `Level2Entrance` / `WestKey` / `EastKey` / `Compass` /
`Level2_5F` / `Level2_5E` / `Level2Boom` / `Level2_0E`. Runners:
`run_level2_clear{6d,6c,7e,6f,5e}.py`, `run_level2_bomb_north.py`,
`run_level2_magic_boomerang.py`, `run_level2_bomb_north_4f.py`,
`run_level2_dodongo.py`. **Boom path:** 0x5f bomb N @(120,101) → **0x4f**.
Use `--infinite-life` for first-pass; Clean STATUS only after natural 2/2.
