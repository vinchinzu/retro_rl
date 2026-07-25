# MK1 Speedrun Research Meta — Path to ~90% Full Clear

Research synthesis for LiuKang **normal-mode** SNES MK1 tournament completion.
Algorithm-agnostic: PPO is the current default, but multimodel chaining, curriculum,
imitation, and other RL approaches are all in scope.

See also: [experiments.md](experiments.md) (numbered backlog), [SPEEDRUN_PLAN.md](../SPEEDRUN_PLAN.md)
(stage inventory and multimodel design), [AGENTS.md](../AGENTS.md) (agent runbook).

---

## North-Star Goal

**~90% full tournament clear** on **normal mode**, LiuKang ladder, start-to-finish:

```
M1 → M2 → M3 → M4 → M5 → M6 → M7 (mirror)
  → Endurance1 → Endurance1B → Endurance2 → Goro → Shang Tsung
```

12 fights, 10 unique save-state prefixes. Each fight is best-of-3 rounds.
Win = `rounds_won >= 2 AND rounds_won > rounds_lost`.

### Normal Mode Definition

| Aspect | Definition |
|--------|------------|
| Platform | SNES MK1 via `stable-retro`, custom integration `MortalKombat-Snes` |
| Character | LiuKang (player 1) |
| Difficulty | Default tournament difficulty encoded in save states (not turbo/debug) |
| Evaluation | Per-stage save states (`Fight_LiuKang` … `ShangTsung_LiuKang`), not RAM hacks mid-fight |
| Stochasticity | Game RNG + agent policy; report both stochastic and deterministic eval |

Save states were extracted at round 1, full health, standard timer — matching a
fair normal-mode fight start. Do **not** tune on turbo (`TAB`) or cheat-modified
health/timer during benchmark runs.

---

## Why Single-Model PPO Failed (Known Results)

### 8M fresh run (`train_speedrun.py --fresh`, Feb–May 2026)

| Metric | Training log | Eval (`speedrun_test.py`) |
|--------|--------------|---------------------------|
| Overall win rate | ~15% | ~8% |
| M1 | — | ~30% |
| M4 | — | ~10% |
| M7 | — | ~50% |
| M2–M6, Endurance, Goro, Shang | — | **0%** |

Training win rate systematically **overstates** eval because:

1. **State randomization during training** — same prefix, jittered positions.
2. **Stochastic policy** — exploration helps training, hurts reproducible eval.
3. **Tier sampling mismatch** — bosses/endurance over-represented vs ladder midgame.

### Root cause: tier mix

Default `LIUKANG_TIERS` in `train_speedrun.py` allocates **~65%** of LiuKang
samples to Endurance + Goro + Shang (38% bosses alone). Mid-ladder stages
(M2–M6) share only **~15%**. A single policy cannot learn 12 distinct matchups
when hard bosses dominate gradient signal and mid-ladder never sees enough steps.

### Specialist results (partial success)

| Model | Focus | Best stage | Notes |
|-------|-------|------------|-------|
| `mk1_shangtsung_ppo_final` | Shang only | ~60% Shang | Strong boss specialist; weak transfer elsewhere |
| `mk1_goro_ppo_final` | Goro only | weak | Sub-boss mechanics (4 arms, damage) need more steps |
| `mk1_fresh_ppo_final` | M1–M7 broad | ~40% M1 | Good base for ladder fine-tune, not full tourney |

**Conclusion:** One monolithic PPO policy is the wrong default for 90% clear.
Multimodel chaining + per-stage training is the highest-confidence path.

---

## Probability Math — What 90% Actually Requires

Full-clear probability is the **product** of per-stage win rates (stages are
sequential; one loss ends the run):

```
P(clear) = p_M1 × p_M2 × … × p_M6 × p_M7 × p_E1 × p_E1B × p_E2 × p_Goro × p_Shng
```

### Equal per-stage rate

If all 12 stages share win rate `p`:

```
P(clear) = p^12 = 0.90  →  p ≈ 99.1%
```

**Every stage must win ~99% of attempts** for 90% full clear. This is the
fundamental difficulty of a 12-stage chain.

### Realistic multimodel targets (milestones)

| Milestone | Target P(clear) | Example per-stage mix | Product |
|-----------|-----------------|----------------------|---------|
| Baseline today | ~0.08% | 8% uniform (current eval) | 0.08%^… ≈ negligible |
| First clear | ~1% | 60% ladder, 40% bosses | ~1 in 100 |
| Reliable | ~50% | 95% ×6, 93% M7, 90% ×3 E, 85% Goro, 80% Shang | ~34% |
| **North star** | **~90%** | **~99% on 8 ladder stages, ~97% endurance, ~94% bosses** | **~90%** |

Worked example for **~90% clear** (multimodel, per-stage specialists):

```
p = 0.99^8  (M1–M6 + M7 + one endurance slot grouped)
  × 0.97^2  (remaining endurance + Goro)
  × 0.94    (Shang)
≈ 0.923 × 0.941 × 0.94 ≈ 0.82–0.90
```

Tighten any stage below ~95% and full-clear drops fast. **Bottleneck stages**
(Shang, Goro, M6) dominate — optimize them first in the experiment loop.

### Multimodel does not bypass the math

Swapping models per stage removes **catastrophic forgetting** but not the
multiplicative chain. Multimodel lets us **train each stage to its own ceiling**;
the product of those ceilings is still the score.

---

## Paths to 90% (Ranked by Confidence)

### 1. Multimodel chaining (P0 — primary)

`speedrun_multimodel.py` already maps `STAGE_MODELS` for Goro and Shang.
Extend to full stage groups:

```
GROUP A (ladder):  mk1_ladder_ppo_final     → M1–M7
GROUP B (endurance): mk1_endurance_ppo_final → E1, E1B, E2
GROUP C (Goro):      mk1_goro_ppo_final      → Goro
GROUP D (Shang):     mk1_shangtsung_ppo_final → Shang
```

Train specialists with **curriculum** (`--curriculum ladder|boss`) so each group
gets appropriate state mix. Benchmark with `--tournament 100`.

### 2. Curriculum learning (P0 — enabler)

Progressive difficulty instead of flat tier mix:

1. **Ladder phase** — M1→M7 until each ≥70% eval
2. **Endurance phase** — add E1/E1B/E2 with carry-over states
3. **Boss phase** — Goro then Shang specialists

Avoid training all 12 stages simultaneously until ladder group is solid.

### 3. Imitation / behavioral cloning (P1)

Human or scripted demos for LiuKang fundamentals (fireball, uppercut, block).
PPO fine-tune from BC policy bootstraps combos that random exploration never finds
(combo wrapper exists but was shelved — see SPEEDRUN_PLAN.md).

### 4. Reward / env shaping (P1)

- Timeout penalty already in `FightingEnv`; tune KO vs timeout balance
- Stage-specific reward scaling (boss damage tolerance)
- Frame-stack / action repeat tuning per boss

### 5. Alternative algorithms (P2)

Not locked to PPO:

| Algorithm | Use case |
|-----------|----------|
| SAC / TD3 | Continuous action variants (if action space expanded) |
| DQN + PER | Simpler policies for single-stage specialists |
| Population-based training | Hyperparam search across stage groups |
| World models / Dreamer | Sample-efficient boss learning (heavy lift) |

Start with PPO specialists; switch algorithm only when a stage plateaus after
≥16M steps and curriculum/reward changes fail.

### 6. Continuous play (P2 — stretch)

Current eval uses per-stage save states (valid for win-rate measurement).
Continuous tournament (no reload) needed for video proof and credit roll, but
not required for the 90% metric.

---

## Success Metrics

### Primary metric (ground truth)

```
full_clear_rate = clears / N   where N = 100 tournament simulations
```

Command:

```bash
cd mortal_kombat
uv run python speedrun_multimodel.py \
  --general models/mk1_ladder_ppo_final.zip \
  --tournament 100 \
  --attempts 20
```

**Advance criterion:** `full_clear_rate` improves vs previous best on same eval seed
protocol. Report 95% Wilson confidence interval for N=100.

### Secondary metrics

| Metric | Command | Use |
|--------|---------|-----|
| Per-stage win % | `speedrun_test.py --attempts 20` | Find bottleneck stage |
| Deterministic eval | `--deterministic` (when added) | Reduce eval variance |
| Training vs eval gap | Compare `SpeedrunMetrics` vs test | Detect overfitting |
| Stage reached distribution | tournament mode furthest_counts | Where chains die |

### Eval protocol (fixed — do not modify between experiments)

1. **Character:** LiuKang
2. **Attempts per stage (per-stage mode):** 20
3. **Tournament simulations:** N=100
4. **Policy:** `deterministic=True` for eval (stochastic only for training)
5. **Model set:** Document exact `--general` and `STAGE_MODELS` in experiment log
6. **Hardware:** Same GPU/CPU class when comparing runs
7. **Log format:** `experiments/results.tsv` — one row per experiment

---

## Autoresearch Pattern — Adapted for MK1 RL

We **do not** vendor-clone [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
into this repo. We adapt its **separation of concerns**:

| Autoresearch | MK1 analogue |
|--------------|--------------|
| `prepare.py` (fixed) | `speedrun_test.py`, `speedrun_multimodel.py`, save states |
| `train.py` (agent edits) | `train_speedrun.py` flags / tier weights / curriculum |
| `program.md` (human loop) | `AGENTS.md` + this doc + `experiments.md` |
| `val_bpb` metric | `full_clear_rate` (N=100 tournaments) |
| 5-minute time budget | ~2–8 hour training budget per experiment |
| git ratchet keep/discard | Model registry + `results.tsv` advance/revert |

### Experiment loop (propose → run → measure → keep/discard)

```
LOOP:
  1. Read experiments.md — pick highest-priority open experiment
  2. Record hypothesis + baseline full_clear_rate in results.tsv
  3. Run training command (fixed step budget unless experiment says otherwise)
  4. Register model in model_registry.py
  5. Eval: speedrun_multimodel.py --tournament 100
  6. If full_clear_rate improved → KEEP (update STAGE_MODELS / registry)
     Else → DISCARD (note failure mode in results.tsv)
  7. Update experiments.md status; propose follow-up from bottleneck stage
```

### External autoresearch clone (optional)

For pure RL-algorithm search (replace PPO loop entirely), clone externally:

```bash
git clone https://github.com/karpathy/autoresearch.git ~/autoresearch-rl-port
# Port prepare.py → MK env + speedrun_multimodel eval harness
# Port train.py → SB3 PPO/SAC training step
```

Keep the port **outside** `retro_rl` until an algorithm change beats PPO on a
single stage (e.g. Shang ≥65% with same step budget).

---

## Do-Not-Repeat List

| Failed approach | Evidence | Why |
|-----------------|----------|-----|
| Single model, default tier mix, 8M fresh | 8% eval, M2–M6 at 0% | Boss oversampling starves ladder |
| Narrow fine-tune from boss specialist | Shang model 10% M1 | Catastrophic forgetting |
| Trust training win rate alone | 15% train vs 8% eval | Randomization + stochastic policy |
| `mk1_ppo_*` naming / old scripts | — | Deprecated; use `train_speedrun.py` |
| One-off training scripts | AGENTS discipline | Add flags to existing scripts |
| Eval with `deterministic=False` only | High variance | Use deterministic for comparisons |

---

## Recommended Next Focus

1. **Establish baseline** — multimodel tournament N=100 with current best models
2. **Ladder specialist** — `--curriculum ladder`, 8M steps, target M2–M6 >50%
3. **Wire STAGE_MODELS** — ladder model for M1–M7, keep boss specialists
4. **Deterministic eval flag** — add to test/multimodel scripts for stable metrics
5. **Iterate bosses** — push Shang 60%→80%, Goro from weak→50%+

See [experiments.md](experiments.md) for numbered commands and success criteria.
