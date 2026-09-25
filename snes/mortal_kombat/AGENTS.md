# Agent instructions: Mortal Kombat (SNES)

Facts: `docs/STATUS.md`. Future: `SPEEDRUN_PLAN.md`. RAM: `docs/ram_map.md`.
Tracker: `bd ready -l mortal_kombat`.

## Commands

```bash
uv run python snes/mortal_kombat/scripts/setup_rom.py
uv run python snes/mortal_kombat/scripts/boot_probe.py
uv run python snes/mortal_kombat/scripts/ram_probe.py
uv run --extra ml python snes/mortal_kombat/scripts/eval_roster.py --attempts 5
uv run --extra ml python snes/mortal_kombat/scripts/run_tournament.py
uv run python snes/mortal_kombat/scripts/replay_natural_fight7.py --repeat 5
uv run --extra ml python snes/mortal_kombat/scripts/capture_natural_endurance1.py --identify-only
```

## Traps

- v3 observation is 20-dim. Do not `--load` a pixel CNN or a v1/v2 MLP zip.
- v3 x/y is `0x00DA` / `0x0174` (animation noise). Live pose is `0x1966` / `0x030F`. Do not retarget v3 observation without a new train.
- Win means `rounds_won >= 2` and `rounds_won > rounds_lost`. Health max is 161. Liu Kang id is 3.
- `LEFT` and `RIGHT` walk. `X` is block. Do not mash START between rounds.
- `p2_rounds` at `0x04B7` is noisy on VS and timer cheats. Count a round only on BETWEEN_ROUNDS.
- Pin HUD after a win is the previous opponent, not the live one. Identify on a visible frame, not the black fade.
- `--promote` only at N>=20. A wall-cutoff zip is not the incumbent.
- Replay tapes load no model. They are not a reactive policy and not credits.
- A Clean tournament stops at Continue. The furthest roster slot is not a clear.
