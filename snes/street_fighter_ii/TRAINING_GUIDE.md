# Street Fighter II Turbo

Commands and traps: `AGENTS.md`.

This file is the manifest status note. A training checkpoint is not a
published match win.

## Commands

```bash
./run_bot.sh play --state Fight_StreetFighterIITurbo
uv run python retro_harness/fighters/train_ppo.py \
  --game sf2 --state Fight_StreetFighterIITurbo --eval \
  --load models/sf2_ppo_final.zip
./validate_states.sh
```

## RAM

| Variable | Address | Hex |
|----------|---------|-----|
| health (P1) | 1590 | `0x0636` |
| enemy_health (P2) | 1840 | `0x0730` |
| timer | 6387 | `0x18F3` |
| rounds_won (P1) | 6300 | `0x189C` |
| enemy_rounds_won (P2) | 6299 | `0x189B` |
| matches_won | 3280 | `0x0CD0` |
| match_state | 6392 | `0x18F8` |

Addresses are below `0x2000`, so they fit `data.json`. Health 176/176 at
`Fight_StreetFighterIITurbo` is a boot check, not a win.
