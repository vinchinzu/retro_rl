# Structured boss combat (full knowledge → RL)

## Decision

**Do not use vision-only networks for bosses until gold.** Pixel BC/PPO from
`../snes_editor/super_metroid_rl` can win on their training save
(`BossTorizo`) but fail natural Flyway entry (statue spritemap freeze /
distribution shift). Continuous acceptance already uses hash-pinned replay
for Bomb Torizo; keep that as the default continuous edge.

Hard spots should instead:

1. **Know the fight** — RAM positions, HP, spritemap, invuln timers, catalog
   hitbox sizes (`sm-json-data` dims).
2. **Code a strategy controller** — Spore Spawn is the template
   (`routes/spore_spawn_controller.py`: vulnerable spritemaps + aim + fire).
3. **Optionally RL-refine** on the structured feature vector (not pixels) to
   cut frames and damage while staying deterministic enough for continuous
   promotion.

## Bomb Torizo facts

| Field | Value | Source |
|-------|------:|--------|
| Room | `0x9804` | continuous / ram_map |
| HP | 800 | sm-json-data bosses |
| Hitbox | 73×90 px | sm-json-data `dims` |
| Contact damage | 8 | sm-json-data |
| Primary weapon | Missiles | early-game loadout |
| Idle spritemap | `0x87D0` | live probe (incomplete saves) |

## Pipeline

```text
natural entry (replay / door play)
        │
        ▼
  activation (spritemap leaves 0x87D0, AI moves)
        │
        ▼
  strategy controller  ←── full-knowledge features
        │                   (combat/features.py)
        ▼
  optional structured RL  (same obs, discrete actions)
        │
        ▼
  promote only after natural-entry continuous evidence
```

## Code

| Module | Role |
|--------|------|
| `combat/features.py` | AABB hitboxes, `CombatFeatures`, RL float vector (14) |
| `combat/actions.py` | Shared discrete action table (13) for RL + distillation |
| `combat/bomb_torizo.py` | Range-kite + missile strategy |
| `combat/kraid.py` | Left-lane Super-spray strategy (policy only) |
| `combat/natural_entry.py` | Capture activation mid continuous bombs prefix |
| `combat/env.py` | Gymnasium env: `feature_vector` obs, reward shaping |
| `scripts/probe/bomb_torizo_combat.py` | strategy / capture / prove / eval / train |
| `scripts/probe/kraid_combat.py` | Kraid strategy from room entry |

### Probe commands

```bash
# Active fight save (strategy baseline)
uv run python super_metroid/scripts/probe/bomb_torizo_combat.py --state BossTorizo
# or explicit:
uv run python super_metroid/scripts/probe/bomb_torizo_combat.py strategy --state BossTorizo

# Natural activation from continuous power-on prefix (~40k frames)
uv run python super_metroid/scripts/probe/bomb_torizo_combat.py capture-natural
uv run python super_metroid/scripts/probe/bomb_torizo_combat.py prove-natural

# Structured Gym: strategy projected onto discrete actions
uv run python super_metroid/scripts/probe/bomb_torizo_combat.py eval --episodes 1
uv run python super_metroid/scripts/probe/bomb_torizo_combat.py eval --state natural

# Short PPO smoke on feature_vector (ml extras)
uv run python super_metroid/scripts/probe/bomb_torizo_combat.py train --timesteps 4096
```

### Measured baseline (strategy, unlimited energy/ammo assist)

| Metric | `BossTorizo` | Natural active (continuous prefix) | Continuous room dwell (replay) |
|--------|-------------:|-----------------------------------:|-------------------------------:|
| Fight frames | **~792** | **~910** | ~3993 (fight + statue + exit) |
| Energy restored | **~50** (3 hits) | **~69** (5 hits) | n/a |
| Max single hit | **30** | **30** | — |
| Deaths | **0** | **0** | 0 |

Some damage is fine under the assist contract; no death. Natural activation is
captured at continuous prefix frame **~42242** (spritemap `0xAA12`, HP 800)
after rejecting room-load garbage and chozo spawn `0x804F`.

Incomplete integration states (`Bomb Torizo Room`, `… [from Flyway]`) freeze on
spritemap `0x87D0` and return `torizo_inactive_statue` — expected. Use
`BossTorizo` or a natural activation capture.

Natural-entry capture writes
`custom_integrations/SuperMetroid-Snes/scratch/natural_bomb_torizo_active.state`
(gitignored). `prove-natural` captures if missing, then runs the strategy from
that distribution.

Inactive spritemaps (not combat AI): `0x87D0` (statue), `0x804F` (spawn).

### RL status

| Step | Status |
|------|--------|
| Gym obs = `feature_vector` (14 floats) | **done** (`combat/env.py`) |
| Discrete action table (face/move/jump/fire) | **done** (`combat/actions.py`, 13 ids) |
| Reward: boss damage − λ·samus damage − ε·frames + win | **done** |
| Natural-entry capture + strategy prove | **done** (`prove-natural`, ~910f win) |
| 4h PPO from natural entry | **done** (2026-07-30) |
| Distill back to deterministic controller | not started |
| Continuous hybrid promotion | not started (keep hash-pinned replay) |

Reward defaults: `+Δboss_hp` − `0.5·Δsamus_damage` − `0.001·frame` + `50` win /
−`10` timeout.

#### 4h PPO run (natural active state)

```bash
uv run python super_metroid/scripts/probe/bomb_torizo_combat.py train \
  --state natural --hours 4 --checkpoint-freq 50000
```

| Field | Value |
|-------|------:|
| Wall clock | **4.0 h** |
| Timesteps | **7,295,220** (~507 steps/s) |
| Model | `models/bomb_torizo_feature_4h/bomb_torizo_feature_ppo.zip` |
| Report | `models/bomb_torizo_feature_4h/train_report.json` |
| Final eval (3 eps, natural, deterministic) | **3/3 wins** |

| Policy | Fight frames | Damage taken | Deaths | Wins |
|--------|-------------:|-------------:|-------:|-----:|
| Free strategy (`prove-natural`) | ~910 | ~69 energy restored | 0 | 1/1 |
| Discrete strategy projection (`eval --policy strategy`) | ~924 | ~60 | 0 | 1/1 |
| **PPO after 4h** (final zip) | **1264** | **80** | **0** | **3/3** |

PPO wins reliably on the natural distribution but is **~350 frames slower** than the
scripted strategy and takes more contact damage. Next: distill or constrain toward
the strategy action prior, or train with a stronger time penalty / imitation term,
before continuous hybrid promotion.

Note: stable-retro allows only one emulator per process — in-process
`EvalCallback` with a second env is disabled; final eval runs after train closes.

## Continuous integration policy

- **Default:** keep `pit_to_post_torizo` hash-pinned replay (accepted).
- **Optional hybrid (future):** replay until Torizo activation, strategy
  (or RL) for the fight, then scripted bombs pickup + exit.
- **Never** write boss/event/item RAM to claim a win.

## Kraid (policy only) + Varia closeout

| Field | Value | Source |
|-------|------:|--------|
| Room | `0xA59F` | KPDR / ram_map |
| Body HP | 1000 | live probe (enemy0) |
| Primary weapon | Super Missiles | early KPDR loadout |
| Entry | doorway-natural (`samus_x≈39`) | `eye_hj_kraid_entry` / composed |
| Closeout | rear blue door → Varia Room `0xA6E2` → Chozo shot + PLM | `combat/kraid.py` |

Deterministic Super-spray lane policy in `combat/kraid.py` (left-mid lane,
face right, pulse Supers + jumps), then rear-door exit and real Varia PLM
(shoot Chozo hand, touch orb). Probe:

```bash
uv run python super_metroid/scripts/probe/kraid_combat.py strategy --state entry
uv run python super_metroid/scripts/probe/kraid_combat.py varia --state entry
```

Measured baseline (policy only, unlimited energy/ammo assist):

| Metric | `eye_hj_kraid_entry` | `dev_kpdr_kraid_entry` | composed warehouse |
|--------|---------------------:|-----------------------:|-------------------:|
| Body zero | **~1321** | **~8333** | **~11685** |
| Boss bit 0 | **~1520** | **~8693** | **~12045** |
| Varia room | **~1756** | — | — |
| Varia collect | **~1908** | — | — |
| Energy restored | **0** | **0** | **0** |
| Deaths | **0** | **0** | **0** |

Prefer `entry` / `eye_hj_kraid_entry` for iteration. Report:
`debug/kraid_varia_run.json`. Not continuous evidence until composed on the
power-on KPDR prefix after `play_eye_to_kraid`. No RL for Kraid — strategy only.
KPDR segment: `kraid_entry_to_varia`.
