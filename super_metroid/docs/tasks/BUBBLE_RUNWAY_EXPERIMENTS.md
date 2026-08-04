# Bubble runway / wall-jump experiments (R18)

Living log of physics-informed probes on the save-door fire seat. Product
defaults live in `bubble_mountain_params` + `bubble_mountain_primitives`; this
file is the experiment board — not hop GREEN.

Related: [`BUBBLE_TECHNIQUES.md`](BUBBLE_TECHNIQUES.md) ·
[`SM-K4.4-PURE-R18.md`](SM-K4.4-PURE-R18.md).

## Physics (pre-Speed, Hi-Jump)

| Fact | Implication |
|------|-------------|
| Horizontal dash does **not** strongly raise jump/WJ **vertical** | WJ vy0 ≈ **5.33** (Hi-Jump) / **4.41** (regular) |
| Dash saturates ~**32f** at value 2 (no Speed) | Longer dash only helps if the runway has room |
| Arm-pump (L/R) ≈ **+1 px/pose on ground only** | Short runway reach / built momentum; not air speed |
| Spin hold preserves hx | Breaking spin early kills approach |
| Delayed WJ (late into window) | Needs matching approach; naive delay on R15 RED |
| Enemy KB (~**5.25** hx) + A+dir | Strongest fake “transfer”; **non-repeatable** |
| Crouch-jump | +~8 px start when geometry allows |

## Pin

`scratch/bubble_human_runway.state` ~(27, 395) p2 — **dev isolation** only.

## Probe matrix (human pin)

| Recipe | top | min_y | mx200 | Notes |
|--------|-----|-------|-------|-------|
| **Product: Y8 + run21 + spin83 + open double WJ** | **True** | 134 | 300 | Control / ship |
| Y8 + **arm-pump run21** + spin83 + double WJ | **True** | 134 | 303 | Human pin only; **RED pure seat** |
| run24 / 28 / 32 + double WJ | False | 395 | 0 | Walks off short seat |
| arm-pump run28/32 | False | 395 | 0 | Same — no room for max dash |
| crouch1–2 then spin | False | 388 | 0 | Desyncs fire-seat arc |
| delayed WJ1 into 2–8f | False | 142–228 | ≤251 | Burns R15 window |
| **single WJ only** | False | 156 | **251** | Ceiling pocket; needs bug clip to “win” |
| LEFT+A seek before WJ1 (false closed-loop) | False | 156 | 251 | **Burns first WJ** → single-class |

## Skills API (code)

| Skill | Product? | Role |
|-------|----------|------|
| `bubble_prepare_fire_run` | yes | Y-clear; no multi-frame bare RIGHT; optional crouch |
| `bubble_runway_dash` | yes | RIGHT+B ± arm-pump; default 21f |
| `bubble_spin_glide` | yes | RIGHT+B+A preserve spin |
| `bubble_walljump_once` | yes | One pulse; `delay_into_frames` experiment |
| `bubble_consecutive_walljumps` | yes | **N≥2** product; open-loop R15 |
| `bubble_double_walljump_r15` | yes | Double + follow spin |
| `bubble_wait_wall_ready` | optional | RIGHT+B+A only if short of wall |
| `bubble_period_walljump_climb` | climb | Right-structure farm; upgrades to double on latch |
| `bubble_damage_boost_hold` | **no** | KB → A+dir experiment |
| `bubble_save_runway_fire_recipe` | yes | Full compose |

## Product defaults (R18 pure)

```text
seat max-left x∈[25,32] via stationary X+L clear + walk-brake (no LEFT+X)
prepare(y_clear=True, crouch=False)
runway_dash(frames=21, arm_pump=False)
spin_glide(frames=83)
coast + double WJ (L20 a4 R8 + L14 a2 R6) + RIGHT+B+A×40
```

Human pin isolation still greens with arm-pump + WJ2 L24/R14/follow56 (R15).

## Do / do-not

**Do**

* Ship **double** WJ open-loop on the human-matched fire arc
* Arm-pump on short runway (21f)
* Extend approach with **RIGHT+B+A only** if short of wall
* Chain consecutive WJ before vy decays

**Do not (without new pure pin)**

* Gate Phase D on Geruta/Waver damage clip
* Multi-frame bare RIGHT face on max-left seat
* LEFT+A “seek” before WJ1 (burns the first jump)
* Max-dash 32f from x~27 human seat
* Ship single WJ as product

## Pure path (R18 → R19 GREEN)

R18: pure earns p132 + pose 84 (min_y≈159); Phase D caps mx200≈251 without
enemy-phase match. **R19:** idle phase wait for Geruta class A/B → product
fire tops; sticky Super door → ordinary Bat (**2012f** full pure GREEN).

## Reproduce control (human pin)

```bash
uv run python - <<'PY'
# see scripts/probe patterns; or call bubble_save_runway_fire_recipe
# from bubble_human_runway.state → top_reached True
PY
```
