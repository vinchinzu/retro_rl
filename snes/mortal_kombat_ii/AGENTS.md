# Agent Instructions — Mortal Kombat II

SNES MK2 (12 characters; longer tournament). Shared fighters stack:
`retro_harness.fighters`. Docs: `docs/STATUS.md` (create/extend as needed).

## Commands

```bash
./run_bot.sh play --state Fight_LiuKang

# Isolated Fight_LiuKang (RAM-gated; CNN zip is un-eval'd / lost)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy uv run python snes/mortal_kombat_ii/eval_match.py --probe-health
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy uv run python snes/mortal_kombat_ii/eval_match.py --scripted
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy uv run python retro_harness/fighters/train_ppo.py \
  --game mk2 --state Fight_LiuKang --eval \
  --load snes/mortal_kombat_ii/models/mk2_ppo_final.zip

# Train (do not overnight-train until RAM-gated isolated win exists)
uv run python retro_harness/fighters/train_ppo.py \
  --game mk2 --state Fight_LiuKang --steps 500000
./train_multichar.sh

# State extraction (RAM-hack wins; preferred)
./extract_all_states.sh
python cheat_extractor.py --char LiuKang
python cheat_extractor.py --char LiuKang --start-from Match6

# Manual turbo creator / validate / watch
./create_character_states.sh
./validate_states.sh
./watch.sh
```

## Traps

- Boot is long (~15s logos). Menu: Title → CharSelect → Battle Plan → story/bio
  → VS → Fight. START mashes story screens.
- Max health **161** (same as MK1). Health is **high WRAM**
  (`get_ram` index = WRAM + 0x2001); 0x020A/0x020E are **not** health.
- Tournament: `Match 1–8 → Shang Tsung → Kintaro → Shao Kahn`. Opponent order
  varies by character.
- States: 134 total (CharSelect + Fight_* + tournament stages) — already
  extracted; do not re-run full extraction unless corrupted.
- ``rom.sfc`` must point at repo ``roms/Mortal Kombat II.smc`` (SHA1
  f6aa5291759e982ea249c4b76f729ca2f4ab1cf4). A stale absolute symlink
  from another checkout will raise ``No romfiles found``.

## Next

- Isolated Fight_LiuKang RAM-gated match-win (scripted fireball / hitbox). CNN
  ``mk2_ppo_final.zip`` lost 0-2; do not overnight-train.
- More RAM: P1/P2 X, rounds won, timer, character ID
