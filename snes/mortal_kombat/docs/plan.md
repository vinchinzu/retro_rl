# MK1 plan — overnight RAM specialists

## Now

Gear the Bronze Liu Kang path, then **retrain all 12 fights** overnight on
all cores. Pixel CNNs stay on disk as fallbacks only; they are the wrong
observation for v3 and are not loaded.

```bash
uv run python snes/mortal_kombat/scripts/setup_rom.py
uv run python snes/mortal_kombat/scripts/boot_probe.py
uv run python snes/mortal_kombat/scripts/ram_probe.py
uv run python snes/mortal_kombat/scripts/train_overnight.py --dry-run
uv run --extra ml python snes/mortal_kombat/scripts/train_overnight.py --steps 4000000 --jobs 12 --n-envs 2
```

16c/32t: 12 jobs × 2 envs. Logs: `logs/overnight_v3/<Stage>.log`.
Outputs: `models/mk1_v3_<Stage>_ppo_final.zip` + `models/roster.json`.

After training:

```bash
uv run python snes/mortal_kombat/scripts/eval_roster.py --attempts 5
uv run python snes/mortal_kombat/scripts/run_tournament.py
```

## Next

- Tighten attack-state / facing bytes from `ram_probe.py` diffs
- Weak-stage extra steps (Goro, Shang, endurance) if eval < 30%
- Credits detect if `match_counter>=12` is not enough
- Do not STATUS-promote a full clear until a continuous power-on run
  reaches credits
