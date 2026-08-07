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
| `ram.py`, `overworld.py`, `overworld_nav.py` | Snapshots + OW graph |
| `level1.py`, `level1_finish.py`, `dungeon.py` | L1 combat / finish |
| `level2_overworld.py`, `chain.py`, `routes.py` | Post-Triforce + named routes |
| `nav_common.py`, `room_timer.py`, `dungeon_lab.py` | Shared nav + lab |
| `assist.py` | Survival infinite-life (opt-in) |
| `docs/tasks/PROCESS.md` | Dual-track + bead grain |

## Dual track

- **Clean** (default): STATUS-eligible; no health writes.
- **Assisted first pass** (`--infinite-life`): map door geometry / dungeon
  rooms; do not promote as Clean. Contract: `docs/ASSIST_CONTRACT.md`.

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

1. **Past 0x6f** residual (0x6e RIGHT **open** → 0x6f gels+compass) → boom /
   Dodongo (`rr-ebe` / `rr-n5i` shared).
2. Magical Boomerang pure on `ADDR_MAGIC_BOOMERANG`; Dodongo → `triforce & 0x02`.
3. Clean heart-safe door path 0x4A→0x3C→entry (parallel, not tip-blocking).
Detail: `docs/LEVEL2_ROUTE.md`, `docs/tasks/PROCESS.md`.
Key branch (Clean checkpoint): `run_level2_clear6d.py` / `run_level2_clear6c.py` /
`run_level2_clear7e.py` (east key; `Level2EastKey.state`).
Recon: `probe_level2_rooms.py` / `probe_level2_boomerang_path.py --infinite-life`.
**Diamond-east:** `nav_common.diamond_east_phase` — band→wall→S2→pure push.
0x7d band≈157 → **0x7e**; 0x6e band≈113 (WEST entry + key) → **0x6f**.
Bombs: `ADDR_BOMBS=0x0658`, B when selected; selected pos `0x0656`.
Boomerang RAM: `ADDR_BOOMERANG=0x0674`, `ADDR_MAGIC_BOOMERANG=0x0675`.
