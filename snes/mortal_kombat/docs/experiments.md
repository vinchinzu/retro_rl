# MK1 Speedrun Experiment Backlog

Numbered experiments for the autoresearch-style loop:
**propose → run → measure → keep/discard**.

Primary metric: `full_clear_rate` from `speedrun_multimodel.py --tournament 100`.
Log all results in `experiments/results.tsv` (untracked).

Meta doc: [autoresearch_meta.md](autoresearch_meta.md)

---

## How to Run the Loop

```bash
cd mortal_kombat

# 1. Baseline (before any change)
./experiments/run_experiment.sh baseline

# 2. After training a candidate model
./experiments/run_experiment.sh eval mk1_ladder_ppo_final.zip "E003 ladder curriculum"

# 3. Record in results.tsv — script appends a row
```

Advance rule: new `full_clear_rate` > previous best **and** no stage regressed
>5% absolute vs baseline (unless experiment explicitly trades off stages).

---

## Status Legend

| Status | Meaning |
|--------|---------|
| `open` | Not started |
| `running` | Training or eval in progress |
| `keep` | Improved metric — update STAGE_MODELS / registry |
| `discard` | No improvement — document why |
| `blocked` | Needs code or states first |

---

## P0 — Critical Path to Multimodel 90%

### E001 — Multimodel baseline (N=100)

| Field | Value |
|-------|-------|
| **Hypothesis** | Current best general + boss specialists yield measurable full-clear rate |
| **Priority** | P0 |
| **Status** | open |
| **Command** | |
```bash
cd mortal_kombat
uv run python speedrun_multimodel.py \
  --general mk1_fresh_ppo_final.zip \
  --attempts 20 \
  --tournament 100
```
| **Success** | Establish baseline `full_clear_rate` + per-stage table in results.tsv |
| **Discard if** | Script errors or no model files — fix paths first |

---

### E002 — Per-stage bottleneck map (single model)

| Field | Value |
|-------|-------|
| **Hypothesis** | 8M fresh model shows which stages need specialists vs ladder training |
| **Priority** | P0 |
| **Status** | open (partial data: M1 30%, M4 10%, M7 50%, rest 0%) |
| **Command** | |
```bash
cd mortal_kombat
uv run python speedrun_test.py \
  --model models/mk1_speedrun_ppo_final.zip \
  --char LiuKang \
  --attempts 20
```
| **Success** | All 12 stages measured; bottleneck list updated in results.tsv |
| **Discard if** | Model missing — run E003 first |

---

### E003 — Ladder curriculum training (8M)

| Field | Value |
|-------|-------|
| **Hypothesis** | `--curriculum ladder` fixes M2–M6 starvation; mid-ladder >50% eval |
| **Priority** | P0 |
| **Status** | open |
| **Command** | |
```bash
cd mortal_kombat
uv run python train_speedrun.py \
  --curriculum ladder \
  --steps 8000000 \
  --fresh
```
| **Eval** | |
```bash
uv run python speedrun_test.py \
  --model models/mk1_speedrun_ppo_final.zip \
  --attempts 20
uv run python speedrun_multimodel.py \
  --general mk1_speedrun_ppo_final.zip \
  --tournament 100
```
| **Success** | M2–M6 each ≥50%; `full_clear_rate` > E001 baseline |
| **Discard if** | M2–M6 still <20% — try E004 (load fresh base) |

---

### E004 — Ladder fine-tune from fresh base (8M)

| Field | Value |
|-------|-------|
| **Hypothesis** | Fine-tuning `mk1_fresh_ppo_final` with ladder curriculum beats fresh scratch |
| **Priority** | P0 |
| **Status** | discard |
| **Command** | |
```bash
cd mortal_kombat
uv run python train_speedrun.py \
  --curriculum ladder \
  --lr 5e-5 \
  --prefix mk1_ladder_ft_ppo \
  --load mk1_fresh_ppo_final.zip \
  --steps 8000000 \
  --n-envs 2
```
| **Eval** | `uv run python speedrun_multimodel.py --general mk1_ladder_ft_ppo_final.zip --attempts 20 --tournament 100` |
| **Result** | 0/100 full clears. Per-stage eval: M1 85%, M2 10%, M3 5%, M4 10%, M5 40%, M6 65%, Mirror 5%, E1 opp1 0%, E1 opp2 10%, E2 opp1 20%, Goro 0%, Shang 40%. Tournament reached M3 and died there: M1 86%, M2 16%, M3 0%. |
| **Logs** | `experiments/logs/e004_ladder_finetune_20260619_221159.log`; `experiments/logs/e004_ladder_eval_20260620_063033.log` |
| **Success** | M1 ≥60%, M2–M6 average ≥45%, beats E003 on ladder stages |
| **Discard if** | Worse than E003 on ≥4 ladder stages |
| **Decision** | Discard. Ladder fine-tune improved M1/M6 but did not fix early ladder reliability; E1 opp1 and Goro remain 0%, making full clear impossible. Do not wire into `STAGE_MODELS`. |

---

### E005 — Wire full STAGE_MODELS + tournament eval

| Field | Value |
|-------|-------|
| **Hypothesis** | Optimal model per stage group maximizes `full_clear_rate` vs single general |
| **Priority** | P0 |
| **Status** | open |
| **Precondition** | E003 or E004 produces `mk1_speedrun_ppo_final` ladder model |
| **Action** | Edit `speedrun_multimodel.py` `STAGE_MODELS`: |
| | M1–M7 + Endurance → ladder model |
| | Goro → `mk1_goro_ppo_final.zip` |
| | Shang → `mk1_shangtsung_ppo_final.zip` |
| **Command** | |
```bash
cd mortal_kombat
uv run python speedrun_multimodel.py \
  --general mk1_speedrun_ppo_final.zip \
  --tournament 100 \
  --attempts 20
```
| **Success** | `full_clear_rate` ≥ max(E001, single-model E002) |
| **Discard if** | No improvement — keep simpler STAGE_MODELS |

---

### E006 — Shang specialist push (10M)

| Field | Value |
|-------|-------|
| **Hypothesis** | 10M more Shang-only steps reaches 80%+ Shang eval |
| **Priority** | P0 |
| **Status** | open (current ~60%) |
| **Command** | |
```bash
cd mortal_kombat
uv run python retro_harness/fighters/train_ppo.py \
  --game mk1 \
  --state ShangTsung_LiuKang \
  --steps 10000000 \
  --load models/mk1_shangtsung_ppo_final.zip \
  --prefix mk1_shangtsung_ppo
```
| **Eval** | |
```bash
uv run python speedrun_test.py \
  --model models/mk1_shangtsung_ppo_final.zip \
  --attempts 30
```
| **Success** | Shang ≥80%; no regression on other stages when used only via STAGE_MODELS |
| **Discard if** | Shang ≤60% after 10M — try reward shaping (E012) |

---

## P1 — High Value Follow-Ups

### E007 — Endurance specialist (8M)

| Field | Value |
|-------|-------|
| **Hypothesis** | Dedicated endurance training fixes 0% on E1/E1B/E2 |
| **Priority** | P1 |
| **Status** | open |
| **Command** | |
```bash
cd mortal_kombat
uv run python train_speedrun.py \
  --curriculum endurance \
  --steps 8000000 \
  --fresh
```
| **Success** | E1, E1B, E2 each ≥55% |
| **Discard if** | All endurance <25% |

---

### E008 — Goro specialist push (12M)

| Field | Value |
|-------|-------|
| **Hypothesis** | Extended Goro-focused training reaches 50%+ |
| **Priority** | P1 |
| **Status** | running |
| **Command** | |
```bash
cd mortal_kombat
uv run python train_speedrun.py \
  --curriculum boss \
  --lr 5e-5 \
  --prefix mk1_goro_ppo \
  --load mk1_goro_ppo_final.zip \
  --steps 12000000 \
  --n-envs 2
```
| **Backup** | `models/backups/mk1_goro_ppo_final_pre_E008_20260620_081304.zip` |
| **Eval** | Goro stage ≥50% in speedrun_test |
| **Success** | Goro ≥50%; multimodel `full_clear_rate` +5% absolute vs E005 |
| **Discard if** | Goro <30% after 12M |

---

### E009 — Deterministic eval protocol

| Field | Value |
|-------|-------|
| **Hypothesis** | Deterministic policy reduces eval variance; rankings stabilize |
| **Priority** | P1 |
| **Status** | open |
| **Action** | Add `--deterministic` to `speedrun_test.py` and `speedrun_multimodel.py` |
| **Command** | |
```bash
uv run python speedrun_multimodel.py \
  --general mk1_speedrun_ppo_final.zip \
  --tournament 100 \
  --deterministic
```
| **Success** | Std dev of 3×100-run eval drops vs stochastic |
| **Discard if** | Deterministic worse on all stages — keep stochastic for eval |

---

### E010 — Training/eval gap audit

| Field | Value |
|-------|-------|
| **Hypothesis** | Disabling state randomization at eval explains train/eval gap |
| **Priority** | P1 |
| **Status** | open |
| **Command** | Compare `speedrun_test.py --attempts 20` vs training log win rate |
| **Success** | Document gap % in results.tsv; gap <10% after E009 |
| **Discard if** | Gap is policy stochasticity only |

---

### E011 — Imitation bootstrap (LiuKang M1)

| Field | Value |
|-------|-------|
| **Hypothesis** | BC from human demo + PPO fine-tune beats pure PPO on M1 |
| **Priority** | P1 |
| **Status** | open |
| **Command** | Record demo via `watch.py`, BC train (TBD script), then E003 |
| **Success** | M1 ≥70% in ≤4M steps |
| **Discard if** | BC pipeline cost >8M PPO equivalent with no M1 gain |

---

### E012 — Reward shaping for timeout-heavy stages

| Field | Value |
|-------|-------|
| **Hypothesis** | Lower timeout penalty on boss stages increases KO aggression |
| **Priority** | P1 |
| **Status** | open |
| **Action** | Tune `REWARD_TIMEOUT_ROUND` in FightingEnv for boss-only training |
| **Success** | Boss stage win rate +10% absolute |
| **Discard if** | Win rate drops |

---

## P2 — Exploratory / Long Horizon

### E013 — Mirror match specialist (M7)

| Field | Value |
|-------|-------|
| **Hypothesis** | M7 needs dedicated Match7_LiuKang training (mirror confusion) |
| **Priority** | P2 |
| **Command** | `train_speedrun.py --curriculum ladder` with M7 weight 40% |
| **Success** | M7 ≥70% |

---

### E014 — Alternative algorithm (SAC) on Shang

| Field | Value |
|-------|-------|
| **Hypothesis** | SAC beats PPO on single-stage Shang given same step budget |
| **Priority** | P2 |
| **Action** | External autoresearch port or new `train_sac_boss.py` |
| **Success** | Shang > best PPO at same steps |

---

### E015 — Continuous tournament play

| Field | Value |
|-------|-------|
| **Hypothesis** | Continuous runner matches save-state win rates within 5% |
| **Priority** | P2 |
| **Command** | `speedrun_test.py --continuous` |
| **Success** | Full clear video captured; rates within 5% of N=100 sim |

---

### E016 — Combo macro actions + curriculum

| Field | Value |
|-------|-------|
| **Hypothesis** | ComboFrameSkip + ladder curriculum discovers fireball/flying kick |
| **Priority** | P2 |
| **Reference** | `retro_harness/fighters/combo_wrapper.py`, SPEEDRUN_PLAN.md |
| **Success** | M4+ win rate +15% with combos enabled |

---

### E017 — RAM MLP on Fight_LiuKang (1M)

| Field | Value |
|-------|-------|
| **Hypothesis** | RAM-vector MLP PPO reaches ≥80% M1 win in <1M steps vs hours for CNN |
| **Priority** | P0 |
| **Status** | **discard** |
| **Result (2026-05-23)** | Trained 1M steps ×2 (v2 after reset-obs fix). Train WR ~22%. **M1 det 0/20 (0%)**. **M1 stoch 18/50 (36%)**. Full ladder det **0/240**. Tournament N=100 skipped (det would be 0%). |
| **Why discard** | M1 stoch 36% <40% threshold; det 0%. 9-dim obs has no distance/spacing — agent can't learn approach. Large det/stoch gap (see E009). |
| **Next** | RAM discovery sprint (`ram_scan_fight.py`); add X/distance to obs; retry E017 or reward shaping before ladder RAM curriculum. |
| **Reference** | [ram_policy_plan.md](ram_policy_plan.md), `retro_harness/fighters/ram_observation.py` |
| **Train** | |
```bash
cd mortal_kombat
uv run python train_ram_ppo.py \
  --state Fight_LiuKang \
  --steps 1000000 \
  --prefix mk1_ram_ppo
```
| **Eval** | |
```bash
uv run python speedrun_test.py \
  --model models/mk1_ram_ppo_final.zip \
  --ram \
  --attempts 20
uv run python speedrun_multimodel.py \
  --general mk1_ram_ppo_final.zip \
  --ram \
  --tournament 100
```
| **Success** | M1 ≥80% at ≤1M steps; training wall time ≪ E003 CNN run |
| **Pivot if** | M1 ≥80% — repoint STAGE_MODELS to `mk1_*_ram_ppo_final`, prioritize RAM discovery for distance/block |
| **Discard if** | M1 <40% at 1M — fix obs/reward before scaling to ladder curriculum |

---

### E018 — RAM MLP v2 with spacing (1M)

| Field | Value |
|-------|-------|
| **Hypothesis** | 13-dim obs (p1_x, p2_x, p1_y, distance_x @ 0xDA/0x174) unlocks ≥80% M1 at ≤1M steps |
| **Priority** | P0 |
| **Status** | running |
| **Reference** | E017 discard; `data.json` position fields; `MK1_RAM_FEATURES` v2 |
| **Train** | |
```bash
cd mortal_kombat
uv run --extra ml python train_ram_ppo.py \
  --state Fight_LiuKang \
  --steps 1000000 \
  --features v2 \
  --prefix mk1_ram_v2_ppo
```
| **Eval** | |
```bash
uv run --extra ml python speedrun_test.py \
  --model models/mk1_ram_v2_ppo_final.zip \
  --ram \
  --attempts 20
```
| **Success** | M1 ≥80% deterministic at ≤1M steps |
| **Discard if** | M1 det <40% — need block/hitstun RAM or reward shaping |

---

## Experiment Index

| ID | Priority | Status | One-line |
|----|----------|--------|----------|
| E001 | P0 | **running** | Multimodel baseline N=100 (`run_experiment.sh baseline`) |
| E002 | P0 | open | Single-model bottleneck map |
| E003 | P0 | **running** | Ladder curriculum 8M, resume speedrun, `mk1_ladder_ppo` |
| E004 | P0 | open | Ladder fine-tune from fresh base |
| E005 | P0 | open | Full STAGE_MODELS wiring |
| E006 | P0 | open | Shang 80% push |
| E007 | P1 | open | Endurance specialist |
| E008 | P1 | open | Goro 50% push |
| E009 | P1 | open | Deterministic eval |
| E010 | P1 | open | Train/eval gap audit |
| E011 | P1 | open | Imitation bootstrap |
| E012 | P1 | open | Boss reward shaping |
| E013 | P2 | open | M7 mirror specialist |
| E014 | P2 | open | SAC vs PPO on Shang |
| E015 | P2 | open | Continuous tournament |
| E016 | P2 | open | Combo macros |
| E017 | P0 | **discard** | RAM v1 M1 36% stoch / 0% det at 1M — no distance |
| E018 | P0 | **running** | RAM v2 spacing 13-dim, mk1_ram_v2_ppo 1M |

---

## results.tsv Schema

Tab-separated, append-only, **do not commit**:

```
experiment_id  date  hypothesis  command  full_clear_rate  m1  m2  ...  shang  notes  decision
```

Example row:

```
E001  2026-05-22  multimodel baseline  speedrun_multimodel --tournament 100  0.02  0.40  ...  keep
```
