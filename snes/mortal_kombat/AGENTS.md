# Agent Instructions — Mortal Kombat (SNES)

North star: **Bronze/Clean Liu Kang** power-on → Goro → Shang Tsung → credits.
Docs: `docs/STATUS.md`, `docs/plan.md`, `docs/ram_map.md`.
Tracker: `bd ready -l mortal_kombat`.

## Commands

```bash
bd ready -l mortal_kombat

uv run python snes/mortal_kombat/scripts/setup_rom.py
uv run python snes/mortal_kombat/scripts/boot_probe.py
uv run python snes/mortal_kombat/scripts/ram_probe.py

# Overnight: retrain all 12 fights (RAM+hitbox v3, all cores)
uv run python snes/mortal_kombat/scripts/train_overnight.py --dry-run
uv run --extra ml python snes/mortal_kombat/scripts/train_overnight.py --steps 4000000 --jobs 12 --n-envs 2

uv run --extra ml python snes/mortal_kombat/scripts/eval_roster.py --attempts 5
uv run --extra ml python snes/mortal_kombat/scripts/run_tournament.py
```

## Traps

- v3 obs is 20-dim RAM+hitbox. **Do not** `--load` pixel CNN or v1/v2 MLP zips.
- Pixel models may stay as roster fallbacks until the v3 zip exists.
- Win = `rounds_won >= 2 AND rounds_won > rounds_lost`. Health max **161**.
- Liu Kang id **3**. D-pad vs shoulders: `LEFT`/`RIGHT` walk; `X` is block.
- Tournament: `M1–M6 → M7 → E1 → E1B → E2 → Goro → Shang` (12 fights).
- Dual-track save-state eval is not a continuous credits claim.
