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
uv run --extra ml python snes/mortal_kombat/scripts/round_probe.py
uv run --extra ml python snes/mortal_kombat/scripts/round_probe.py --play

# Overnight: retrain all 12 fights (RAM+hitbox v3, all cores)
uv run python snes/mortal_kombat/scripts/train_overnight.py --dry-run
uv run --extra ml python snes/mortal_kombat/scripts/train_overnight.py --steps 4000000 --jobs 12 --n-envs 2

uv run --extra ml python snes/mortal_kombat/scripts/eval_roster.py --attempts 5
uv run --extra ml python snes/mortal_kombat/scripts/eval_roster.py --stages Match7 --checkpoints --attempts 20
uv run --extra ml python snes/mortal_kombat/scripts/eval_roster.py --kind script --stages Fight --attempts 5
uv run --extra ml python snes/mortal_kombat/scripts/eval_roster.py --compare --stages Fight,Match5,Match7 --attempts 5
uv run --extra ml python snes/mortal_kombat/scripts/run_tournament.py
uv run --extra ml python snes/mortal_kombat/scripts/run_tournament.py --ladder-model mk1_v3_Match5_ppo_final.zip
uv run --extra ml python snes/mortal_kombat/scripts/run_tournament.py --scripted
uv run python snes/mortal_kombat/scripts/replay_natural_fight1.py
uv run python snes/mortal_kombat/scripts/replay_natural_fight2.py
uv run python snes/mortal_kombat/scripts/replay_natural_fight2.py --repeat 5
```

## Traps

- v3 obs is 20-dim RAM+hitbox. **Do not** `--load` pixel CNN or v1/v2 MLP zips.
- Continuations require a distinct `--output-prefix` (enforced). Lower
  `--learning-rate` and explicit entropy bounds; never `--promote` before N>=20.
- Pixel models may stay as roster fallbacks until the v3 zip exists.
- Old pixel zips pickle `fighters_common` (now `retro_harness.fighters`); `compat.py` aliases it.
- Win = `rounds_won >= 2 AND rounds_won > rounds_lost`. Health max **161**.
- Liu Kang id **3**. D-pad vs shoulders: `LEFT`/`RIGHT` walk; `X` is block.
- Tournament: `M1–M6 → M7 → E1 → E1B → E2 → Goro → Shang` (12 fights).
- Dual-track save-state eval is not a continuous credits claim.
- Do not mash START between rounds (pauses). VS/continue only, after ~900f quiet.
- Clean tournament stops at Continue (no START). Furthest is roster `match_id`.
- Wall cutoff writes `*_ppo_{timesteps}_steps.zip` and exits 1; not an incumbent.
- `p2_rounds` @ `0x04B7` is noisy on timeout-cheat / VS; trust +1 ticks on BETWEEN_ROUNDS only. Continue still counts the match loss.
- v3 x/y obs is still `0x00DA`/`0x0174` (noise). Live pose is `0x1966`/`0x030F`. Do not retarget v3 obs without retraining — Fight zip went 0/5 when pose leaked into the vector.
- Scripted fireball is F,F,HP after ~90f intro (~25 dmg). `play_buttons_match` must use `rounds_settled`, not raw HUD.
- Current ladder candidate: `mk1_v3_Match5_ppo_final.zip` for M1–M7 (`--ladder-model`). Do not `--promote` it at N<20.
- `replay_natural_fight1.py` is an exact 7,863-frame power-on tape through the
  Match 2 transition. It loads no model, but it is state-exact, not reactive.
- `replay_natural_fight2.py` concatenates that tape with a 5,055-frame Match 2
  continuation (12,918 frames total) through the Match 3 transition. Natural
  Match 2 is Sonya (id 6). The Scorpion byte at the Fight 1 pin is leftover
  HUD, not the live opponent. Cold-boot each replay (`make_env`); `env.reset()`
  after a long `NONE` run is not a power-on pin.
