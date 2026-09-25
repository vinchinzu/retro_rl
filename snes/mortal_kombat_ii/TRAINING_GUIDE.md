# Mortal Kombat II

Commands and traps: `AGENTS.md`. State inventory: `docs/STATUS.md`.

This file is the manifest status note. It is not a match-win claim and not
an arcade-clear claim.

## Commands

```bash
./run_bot.sh play --state Fight_LiuKang
./validate_states.sh
./watch.sh
```

## Traps

- Do not start a long train until a RAM-gated isolated `Fight_LiuKang` win
  exists. `models/mk2_ppo_final.zip` lost 0-2.
- Health is high WRAM, max 161. `0x020A` and `0x020E` are not health.
- `rom.sfc` must point at repo `roms/Mortal Kombat II.smc`. A stale absolute
  symlink raises `No romfiles found`.
- 134 states are already extracted. Do not re-run full extraction unless a
  state is corrupt.
