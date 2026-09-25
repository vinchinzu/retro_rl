# Agent instructions: Super Street Fighter II

RAM and the manifest note: `TRAINING_GUIDE.md`.

## Commands

```bash
./run_bot.sh play --state Fight_SuperStreetFighterII
uv run python retro_harness/fighters/train_ppo.py \
  --game ssf2 --state Fight_SuperStreetFighterII --steps 500000
uv run python retro_harness/fighters/train_ppo.py \
  --game ssf2 --state Fight_SuperStreetFighterII --eval \
  --load models/ssf2_ppo_final.zip
./validate_states.sh
./watch.sh
```

## Traps

- `data.json` has no `rounds_won`. A reward log is not a win rate.
- Fight-ready health 176/176 only checks the state. It is not a win.
- There is no `train_multi_opponent.sh` here. Do not add a phase the tree
  cannot run.
