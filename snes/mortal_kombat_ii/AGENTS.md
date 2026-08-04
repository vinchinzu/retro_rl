# Agent Instructions — Mortal Kombat II

SNES MK2 (12 characters; longer tournament). Shared fighters stack:
`retro_harness.fighters`. Docs: `docs/STATUS.md` (create/extend as needed).

## Commands

```bash
./run_bot.sh play --state Fight_LiuKang

# Train
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

## Next

- Train multi-character tournament policies
- More RAM: rounds won, timer, character ID
