# Status — Mortal Kombat (SNES)

## Program gate

| Field | Value |
|-------|-------|
| Goal | Bronze/Clean Liu Kang arcade: power-on → Goro → Shang Tsung → credits |
| Approach | RAM-gated boot + per-fight v3 RAM/hitbox specialists (pixel CNN fallback) |
| Runtime class | Bronze (read-only RAM) |
| Maturity | **M3** isolated match wins; boot verified; M1→M2 swap probed; v3 4M roster eval'd |
| Bead | `rr-qpug` |

## Known results (do not re-discover)

Pixel CNN (save-state eval, historical):

| Run | Eval | Notes |
|-----|------|-------|
| 8M `train_speedrun.py --fresh` | ~8% overall | M1 30%, M4 10%, M7 50%; M2–M6/E/G/S 0% |
| `mk1_shangtsung_ppo_final` | ~30–60% Shang | Boss specialist; keep as pixel fallback |
| `mk1_goro_ppo_final` | weak | Retrain with v3 |
| `mk1_fresh_ppo_final` | ~40% M1 | Ladder base, not a full-clear model |
| `mk1_ladder_ft_ppo_final` | 0/100 clears | M1 85%, M3 0%, E1/Goro 0% |
| RAM v1 9-dim | M1 det 0%, stoch 36% | No spacing |
| RAM v2 13-dim | 0/240 det | Spacing still not enough |

v3 (20-dim hitbox RAM) overnight **finished 2026-08-23 01:15 CDT**: 12/12
`mk1_v3_<stage>_ppo_final.zip`, 4,001,792 steps, MlpPolicy, no pixel load.
Save-state eval N=5 (`eval_roster.py`, kind=`ram_v3`):

| Stage | Win% | W-L |
|-------|-----:|----:|
| Fight (M1) | 60% | 3-2 |
| Match2 | 20% | 1-4 |
| Match3 | 60% | 3-2 |
| Match4 | 40% | 2-3 |
| Match5 | 80% | 4-1 |
| Match6 | 20% | 1-4 |
| Match7 | 0% | 0-5 |
| Endurance1 | 0% | 0-5 |
| Endurance1B | 20% | 1-4 |
| Endurance2 | 0% | 0-5 |
| Goro | 0% | 0-5 |
| Shang Tsung | 20% | 1-4 |

N=5 is noisy. Isolated save-state wins are not a continuous credits claim.
Pixel zips remain round-loss fallbacks. Do not STATUS-promote a clear until a
continuous power-on run reaches credits.

Follow-up checkpoint audit: Match7 earlier checkpoints peaked at 2/20 (10%);
the 4M final was 1/20 (5%). Endurance1, Endurance2, and Goro were each 0/34
across every 250k–4M checkpoint (N=2/checkpoint). Episode traces confirmed
correct states/opponents, real damage, and valid termination; policies collapsed
onto a small action subset. Do not blindly resume these finals.

Verified this session (do not re-discover):

- Live pose (Fight_LiuKang): P1 X/Y `0x1966`/`0x1968` start 68/144; P2 X `0x030F` starts 180 and walks in. `0x00DA`/`0x0174` are animation noise. Overnight v3 zips were trained on the noise; retargeting obs dropped Fight zip from ~60% to 0/5. Restored. Scripted policy reads the live pose bytes only.
- Liu Kang F,F,HP fireball: after ~90 intro frames, 25 dmg on Sub-Zero (three repeats). F-holds walk into range; scripted Fight save-state 0/3. Pixel Match7 9.5M zip 0/5.
- Deterministic no-model replay now clears the exact `Fight_LiuKang` save state
  3/3 (two enemy health zero-crossings versus one player zero-crossing). The
  4,062-frame input is RLE-compressed in `fight1_tape.py`; the v3 Fight model
  was used only once as an offline trace oracle. This is not a natural-entry
  Match 1 or first-seven-matches claim.
- Natural-entry power-on replay now reaches the Match 2 transition at frame
  7,863, 5/5 with identical RAM trajectories. Runtime uses only an RLE SNES
  input tape; the offline oracle used deterministic Match5 v3 plus the existing
  pixel round-loss fallback. This proves one state-exact natural Match 1, not a
  reactive policy or a first-seven clear.
- Natural-entry continuation now reaches the Match 3 transition at frame
  12,918 (7,863 + 5,055), 5/5 identical HUD/pose signatures on cold boots.
  Match 1 vs Cage 2–0, Match 2 vs Sonya 2–0. RAM: `match_counter` 0→1→2 and
  Liu Kang id 3. Runtime still loads no models. The Scorpion id at the Fight 1
  pin is leftover HUD; the live Match 2 opponent is Sonya. Not a reactive
  policy or a first-seven clear.
- Natural-entry continuation now reaches the Match 4 transition at frame
  18,077 (12,918 + 5,159), 5/5 identical HUD/pose signatures on cold boots.
  Match 3 vs Sub-Zero (id 5) 2–0 (Cage 2–0, Sonya 2–0). RAM: `match_counter`
  0→1→2→3 and Liu Kang id 3. The Sonya byte at the Fight 2 pin is leftover
  HUD. Runtime still loads no models. Not a reactive policy or a first-seven
  clear.
- Natural-entry continuation now reaches the Match 5 transition at frame
  25,164 (18,077 + 7,087), 5/5 identical HUD/pose signatures on cold boots.
  Match 4 vs Raiden (id 2) 2–1. RAM: `match_counter` 0→1→2→3→4 and Liu Kang
  id 3. The Sub-Zero byte at the Fight 3 pin is leftover HUD. RAM oracles
  lost Raiden; offline capture used pixel `mk1_ladder_ft_ppo_final.zip`.
  Runtime still loads no models. Not a reactive policy or a first-seven clear.
- Natural-entry continuation now reaches the Match 6 transition at frame
  29,783 (25,164 + 4,619), 5/5 identical HUD/pose signatures on cold boots.
  Match 5 vs Kano (id 1) 2–0. RAM: `match_counter` 0→1→2→3→4→5 and Liu Kang
  id 3. The Raiden byte at the Fight 4 pin is leftover HUD. Deterministic
  RAM and pixel oracles lost 0–2; offline capture used stochastic
  `mk1_v3_Match5_ppo_final.zip`. Runtime still loads no models. Not a
  reactive policy or a first-seven clear.
- Natural-entry continuation now reaches the Match 7 transition at frame
  36,752 (29,783 + 6,969), 5/5 identical HUD/pose signatures on cold boots.
  Match 6 vs Johnny Cage again (id 0) 2–1. RAM: `match_counter`
  0→1→2→3→4→5→6 and Liu Kang id 3. The Kano byte at the Fight 5 pin is
  leftover HUD. First fight-ready can be a black fade with p2=0; identify
  on a visible frame. Match6 v3 lost 0–2; offline capture used
  deterministic `mk1_v3_Match5_ppo_final.zip`. Runtime still loads no
  models. Not a reactive policy or a first-seven clear.
- Natural-entry continuation now reaches the Endurance 1 transition at
  frame 41,503 (36,752 + 4,751), 5/5 identical HUD/pose signatures on
  cold boots. Match 7 vs the Liu Kang mirror (id 3) 2–0. RAM:
  `match_counter` 0→1→2→3→4→5→6→7 and Liu Kang id 3. The Cage byte at
  the Fight 6 pin is leftover HUD. Deterministic RAM and pixel oracles
  lost 0–2 / 1–2; offline capture used stochastic
  `mk1_v3_Match5_ppo_final.zip` (attempt 1). Runtime still loads no
  models. This is the first seven matches as one state-exact tape, not a
  reactive policy or a credits claim.
- Match5 v3 zip, restored obs, save-state N=5: own 3/5; Fight 4/5; Match2 4/5; Match7 2/5 (first v3 Match7 wins this session).
- Clean tournament N=5 with per-stage v3: 4/5 died Match 1; 1/5 furthest Match 3. Same N=5 with `--ladder-model mk1_v3_Match5_ppo_final.zip`: furthest Match 3 (2) / Match 4 (3). Win counter stayed 0 (missed +1 ticks / HUD). Not a 7-match claim.
- `KIND_SCRIPT` / `--scripted` / `--ladder-model` / `--kind script` / `--compare` are wired. `--promote` still N>=20 v3 checkpoints only.

Verified earlier (do not re-discover):

- Power-on boot: `char=LiuKang vs=JohnnyCage hp=161/161 timer=153`
- Corrected Clean tournament probe stops at first Continue: Match 1 loss,
  frame 6,882. An earlier probe that appeared to reach Match 4 had silently
  accepted continues; the runner now terminates and preserves high-water stage.
- `Fight_LiuKang` is vs **Sub-Zero**, not Cage
- Timeout-KO through Match 1 then roster swap into Match 2 (Scorpion):
  `round_probe.py` → `between_rounds=True model_swap=True next_match_fight=True`
  `wins=1 losses=0` `fight:Fight:mk1_multichar_…` → `fight:Match2:mk1_ladder_ft_…`
- Old pixel zips need `fighters_common` → `retro_harness.fighters` alias (`compat.py`)
- START during KO pauses; VS mash only after ~900f. `p2_rounds` HUD is noisy
  during timer=1 cheats and VS — tournament scores +1 ticks on BETWEEN_ROUNDS only.

## RAM (get_ram)

See [`ram_map.md`](ram_map.md). Max health **161**. Liu Kang id **3**.
Win = `rounds_won >= 2 AND rounds_won > rounds_lost`.

## Scripts

| Script | Role |
|--------|------|
| `scripts/boot_probe.py` | Power-on → Liu Kang fight-ready |
| `scripts/ram_probe.py` | Fighter-object / punch diffs |
| `scripts/round_probe.py` | Match 1→2 RAM events + roster swap (timeout-KO or `--play`) |
| `scripts/eval_roster.py` | Per-fight eval → `models/roster.json`; `--checkpoints` ranks v3 zips (`--promote` only at N>=20); `--kind script` / `--compare` print-only |
| `scripts/train_overnight.py` | 12 parallel v3 specialists; wall cutoff is not `_final` / not recorded |
| `scripts/run_tournament.py` | Continuous attempt; stops at Continue; furthest is slot `match_id`; `--ladder-model` / `--scripted` |
| `scripts/replay_natural_fight1.py` | Model-free exact power-on tape through the Match 2 transition |
| `scripts/replay_natural_fight2.py` | Model-free exact power-on tape through the Match 3 transition |
| `scripts/replay_natural_fight3.py` | Model-free exact power-on tape through the Match 4 transition |
| `scripts/replay_natural_fight4.py` | Model-free exact power-on tape through the Match 5 transition |
| `scripts/replay_natural_fight5.py` | Model-free exact power-on tape through the Match 6 transition |
| `scripts/replay_natural_fight6.py` | Model-free exact power-on tape through the Match 7 transition |
| `scripts/replay_natural_fight7.py` | Model-free exact power-on tape through the Endurance 1 transition |
