# Agent Instructions — zelda_i

NES Legend of Zelda (graph nav; **M5** Clean power-on → Level 1 Triforce).
Shared: `retro_harness.adventure`, `retro_harness.nes`. Docs: `docs/STATUS.md`,
`docs/plan.md`, `docs/LEVEL1_ROUTE.md`, `docs/LEVEL2_ROUTE.md`,
`docs/DUNGEON_LAB.md`.

## Commands

```bash
uv run python zelda_i/scripts/setup_rom.py
uv run python zelda_i/scripts/boot_probe.py

# Natural-entry chain (power-on → sword → L1 complete)
uv run python zelda_i/scripts/run_sword_cave.py --natural-entry
uv run python zelda_i/scripts/run_to_level1.py --natural-entry
uv run python zelda_i/scripts/run_level1_complete.py --natural-entry --trials 2
uv run python zelda_i/scripts/run_to_level2_prefix.py --from-heart --trials 2

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

Heart-safe door path 0x37→0x3C → Moon 0x7d → L2 clear (`triforce & 0x02`).
Detail: `docs/LEVEL2_ROUTE.md`, `docs/research/DUNGEON_WALKTHROUGHS.md`.
