# Agent instructions: Street Fighter II Turbo

RAM and the manifest note: `TRAINING_GUIDE.md`.

## Commands

```bash
./run_bot.sh play --state Fight_StreetFighterIITurbo
uv run python retro_harness/fighters/train_ppo.py \
  --game sf2 --state Fight_StreetFighterIITurbo --steps 500000
uv run python retro_harness/fighters/train_ppo.py \
  --game sf2 --state Fight_StreetFighterIITurbo --eval \
  --load models/sf2_ppo_final.zip
./validate_states.sh
./watch.sh
```

## Traps

- Fight-ready health 176/176 only checks the state. It is not a win.
- `rounds_won` is in `data.json`. Still do not publish a win rate from an
  unfinished eval.
- There is no `train_multi_opponent.sh` in this tree. Do not document a
  second training phase that the scripts do not run.
