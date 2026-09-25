# Super Street Fighter II

Commands and traps: `AGENTS.md`.

This file is the manifest status note. Training steps are not a match win.
`data.json` has no `rounds_won`, so win rate is not tracked.

## Commands

```bash
./run_bot.sh play --state Fight_SuperStreetFighterII
uv run python retro_harness/fighters/train_ppo.py \
  --game ssf2 --state Fight_SuperStreetFighterII --eval \
  --load models/ssf2_ppo_final.zip
```

## RAM

Health P1 `0x0636`, enemy health `0x0876`, timer `0x1929`, current battle
`0x18BC`, P1 position `0x0617` / `0x061A`, P2 position `0x0857` / `0x085A`,
hit flags `0x0594` / `0x07D4`. The health offset matches Street Fighter II
Turbo. Health 176/176 at `Fight_SuperStreetFighterII` is a boot check, not
a win.
