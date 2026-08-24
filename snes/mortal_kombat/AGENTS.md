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
uv run python snes/mortal_kombat/scripts/replay_natural_fight3.py
uv run python snes/mortal_kombat/scripts/replay_natural_fight3.py --repeat 5
uv run python snes/mortal_kombat/scripts/replay_natural_fight4.py
uv run python snes/mortal_kombat/scripts/replay_natural_fight4.py --repeat 5
uv run python snes/mortal_kombat/scripts/replay_natural_fight5.py
uv run python snes/mortal_kombat/scripts/replay_natural_fight5.py --repeat 5
uv run python snes/mortal_kombat/scripts/replay_natural_fight6.py
uv run python snes/mortal_kombat/scripts/replay_natural_fight6.py --repeat 5
uv run python snes/mortal_kombat/scripts/replay_natural_fight7.py
uv run python snes/mortal_kombat/scripts/replay_natural_fight7.py --repeat 5
uv run --extra ml python snes/mortal_kombat/scripts/capture_natural_endurance1.py --identify-only
uv run --extra ml python snes/mortal_kombat/scripts/capture_natural_endurance1.py --oracles match5-v3
uv run --extra ml python snes/mortal_kombat/scripts/capture_natural_endurance1.py --stochastic --repeats 20 --oracles match5-v3
uv run --extra ml python snes/mortal_kombat/scripts/capture_natural_endurance1.py --oracles scripted-courtyard --win-at 8
uv run --extra ml python snes/mortal_kombat/scripts/capture_natural_endurance1.py --stochastic --repeats 20 --oracles match5-v3 --round2-kano --win-at 8
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
- `replay_natural_fight3.py` concatenates through Match 3 (18,077 frames:
  12,918 + 5,159) to the Match 4 transition. Live Match 3 is Sub-Zero (id 5).
  The Sonya byte at the Fight 2 pin is leftover HUD.
- `replay_natural_fight4.py` concatenates through Match 4 (25,164 frames:
  18,077 + 7,087) to the Match 5 transition. Live Match 4 is Raiden (id 2)
  2–1. The Sub-Zero byte at the Fight 3 pin is leftover HUD. RAM oracles
  lost this match; the offline capture used the pixel ladder-ft zip.
- `replay_natural_fight5.py` concatenates through Match 5 (29,783 frames:
  25,164 + 4,619) to the Match 6 transition. Live Match 5 is Kano (id 1)
  2–0. The Raiden byte at the Fight 4 pin is leftover HUD. Deterministic
  RAM and pixel oracles lost 0–2; offline capture used stochastic
  Match5 v3. Runtime still loads no models.
- `replay_natural_fight6.py` concatenates through Match 6 (36,752 frames:
  29,783 + 6,969) to the Match 7 transition. Live Match 6 is Johnny Cage
  again (id 0) 2–1, not Scorpion. The Kano byte at the Fight 5 pin is
  leftover HUD. First fight-ready can be a black fade with p2=0; wait
  for a visible frame. Offline oracle was deterministic Match5 v3 after
  Match6 v3 lost 0–2. Runtime still loads no models.
- `replay_natural_fight7.py` concatenates through Match 7 (41,503 frames:
  36,752 + 4,751) to the Endurance 1 transition. Live Match 7 is the Liu
  Kang mirror (id 3) 2–0. The Cage byte at the Fight 6 pin is leftover
  HUD. Deterministic RAM and pixel oracles lost 0–2 / 1–2; offline
  capture used stochastic Match5 v3 (attempt 1). Runtime still loads no
  models. First seven matches as one exact tape is not a reactive policy
  or a credits claim.
- Natural Endurance 1 from the Fight 7 pin is Kano (id 1) on the
  courtyard, not leftover Liu Kang (id 3). Identify on a visible frame
  (first-ready 906f is a black fade; visible 957f). Isolated
  `Endurance1_LiuKang` is Sub-Zero on the same courtyard;
  `Endurance1B_LiuKang` is Kano on the warrior shrine. Match5 v3 is 5/5
  det on throne-room Kano (`Match5_LiuKang`) and has not closed
  courtyard Kano. `ladder_model` only rewrites M1–M7; capture must force
  the oracle onto E1/E1B. Scorpion still has not appeared. E1 vs Kano is
  still best-of-3 with health refill; the second fighter appears only
  after two round wins (`match_counter` 7→8). Do not wait for E1B after
  one KO. `--round2-kano` must ignore leftover pin HUD (`hp=59/0`
  `rounds=2-0`) and stick on keepaway after the first live KO. `p2.state`
  stays 0 for the knife; duck when sprite `0x1B36` leaves Kano. Stale
  `0x1B36=180` while Kano walks is not a knife. Constant duck makes
  Kano rush. Jumping the opener: idle until 296 start-pose frames
  (visible+240; fade is 51f with pose already 68/144), tap UP+forward
  10f, then *wait* — y drops ~20-30f later and walking/flying-kick
  during that startup cancels the jump. Land HK is 40 dmg but Kano
  walks under so we land crossed. Air HK on y-drop hits 25 and stays
  same-side from a leftover-pin probe (~151/183); standing HK then
  10 at ~182/214. `--oracles scripted-courtyard --win-at 8` still
  0-2: the TournamentRunner path lands crossed (~177/54, Kano 136).
  Do not treat y=143 as a land (standing is 144) or we walk during
  the hop. Keepaway still has not closed a round; rolling Match5 v3
  for 2–0 is the wrong next step.
