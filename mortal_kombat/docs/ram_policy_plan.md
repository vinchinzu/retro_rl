# MK1 RAM Policy Plan

Concrete stack for replacing CNN pixel PPO with RAM-vector MLP PPO on the
existing speedrun pipeline. Pixel experiments (E003 ladder resume) can continue
as a baseline in parallel.

## 1. Recommended Stack

| Layer | Pixel (current) | RAM (new) |
|-------|-----------------|-----------|
| Observation | `(4, 84, 84)` uint8 CNN | `Box(9,)` float32 MLP |
| Policy | `CnnPolicy` + `FighterCNN` | `MlpPolicy` `[256, 128]` |
| Actions | 32 discrete (`MK_FIGHTING_ACTIONS`) | Same |
| Reward | `FightingEnv` (damage, round win, timeout) | Same |
| Frame skip | 4 | 4 |
| Wrappers dropped | — | `GrayscaleResize`, `FrameStack` |

**Implementation:** `fighters_common/ram_observation.py`

- `RamObservation` — normalized vector from `data.json` info keys
- `make_ram_fighting_env()` — factory parallel to `make_fighting_env()`
- `build_eval_env(..., ram=True)` — shared eval builder for `speedrun_test.py`

### v1 observation vector (9 dims)

| # | Feature | Source | Normalize |
|---|---------|--------|-----------|
| 0 | p1_health | `health` @ 0x04B9 | / 161 |
| 1 | p2_health | `enemy_health` @ 0x04BB | / 161 |
| 2 | p1_health_delta | Δ health | / 161, clip ±1 |
| 3 | p2_health_delta | Δ enemy_health | / 161, clip ±1 |
| 4 | timer | `timer` @ 0x0122 | / 154 |
| 5 | p2_char_id | `p2_character` @ 0x0024 | / 6 |
| 6 | p1_rounds | `p1_rounds` @ 0x196E | / 2 |
| 7 | p2_rounds | `p2_rounds` @ 0x04B7 | / 2 |
| 8 | match_counter | `match_counter` @ 0x000A | / 11 |

Reserved for discovery sprint: `distance`, `block_flag`, `hitstun`, `last_action`.

## 2. RAM Discovery Sprint (1–2 sessions)

**Between stages:** `cheat_extractor.py --scan` — match counter candidates.

**In-fight:** `ram_scan_fight.py` — walk/jump/block diffs on `Fight_LiuKang`.

Targets:

- Player/enemy X (or screen-position proxies) → spacing policy
- Sub-pixel state (blocking, hitstun, attack id) → anti-air / block features
- Confirm `match_counter` / difficulty don’t drift in eval save states

Unknown addresses in `CLAUDE.md` (difficulty, position) are the blocker for
smart spacing — not more PPO steps on pixels.

## 3. Input Hacking

| Level | Meaning | Status |
|-------|---------|--------|
| Harness | `env.step(buttons)` drives the same path as human input | ✅ today |
| Combo macros | `ComboFrameSkip` in `fighters_common/combo_wrapper.py` | E016 |
| ROM-level | Controller read address, null opponent, scripted combos | Research |

Eval integrity: **no RAM hacks during benchmark** — save states use full
health; no mid-fight god mode.

## 4. Multimodel (still mandatory for ~90%)

RAM does not change the math: `P(clear) = ∏ p_stage`. It makes each specialist
cheaper to train (minutes per stage, not hours).

Proposed `STAGE_MODELS` after RAM pivot:

```
M1–M7     → mk1_ladder_ram_ppo_final
E1–E2     → mk1_endurance_ram_ppo_final
Goro/Shang → existing or RAM specialists
```

Per-stage opponents still need separate models or curriculum — M2 ≠ M4 in RAM.

## 5. Autoresearch Loop

| Piece | RAM version |
|-------|-------------|
| `prepare.py` / states | Fixed RAM obs + eval states (unchanged `.state` files) |
| `train.py` | Agent edits obs vector / reward / algorithm, not CNN |
| Metric | Still `full_clear_rate` (N=100) |
| Ratchet | Keep if M2 win% ↑ at same step budget |

**P0 experiment:** [E017](experiments.md#e017--ram-mlp-on-fight_liukang-1m) —
RAM MLP on `Fight_LiuKang` only. If M1 >80% in <1M steps, pivot the backlog.

## 6. Commands

```bash
cd mortal_kombat

# Train RAM baseline (E017)
uv run python train_ram_ppo.py --state Fight_LiuKang --steps 1000000

# Per-stage eval (same API as pixel models)
uv run python speedrun_test.py \
  --model models/mk1_ram_ppo_final.zip \
  --ram \
  --attempts 20

# RAM discovery
uv run python ram_scan_fight.py --state Fight_LiuKang
uv run python cheat_extractor.py --char LiuKang --scan
```

## 7. What Won’t Solve Itself

- Per-stage opponents need separate data or models
- Continuous play (no save states) is extra; per-stage 90% is enough for the
  north-star metric in `autoresearch_meta.md`
- Pixel PPO at 8M steps without ladder curriculum will keep starving M2–M6
