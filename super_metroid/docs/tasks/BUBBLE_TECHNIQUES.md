# Bubble Mountain techniques map

Living map from community / TAS / pure RECON → code predicates and item caps.
**Caps (K4.4 first Bubble):** Morph + Bombs + Missiles + Supers + Hi-Jump +
Varia — **no Speed**.

Related: phase ladder [`SM-K4.4-PHASE-LADDER.md`](SM-K4.4-PHASE-LADDER.md) ·
R19 residual [`SM-K4.4-PURE-R19-residual.md`](SM-K4.4-PURE-R19-residual.md) ·
runway experiments [`BUBBLE_RUNWAY_EXPERIMENTS.md`](BUBBLE_RUNWAY_EXPERIMENTS.md).

**Physics (pre-Speed):** runway hx does **not** strongly raise WJ vertical
(~5.33 Hi-Jump / ~4.41 regular). Dash buys positioning, spin hold, and
contact timing. Real transfer = consecutive WJ chain, delayed window, crouch
(+8 px), or enemy KB clip (experiment only).

## Primitives (code)

| Technique | Source | Entry predicates | Module / symbol | Success metric | Failure modes |
|-----------|--------|------------------|-----------------|----------------|---------------|
| Max-left fire seat | maprando room 97 left climb; human `bubble_jump_try` | x∈[25,60] y∈[380,430] stand_pin; human band x∈[25,32] | `bubble_on_save_runway`, `SAVE_RUNWAY_FIRE_X`, `SAVE_HUMAN_SEAT_X` | seated solid before run | pure left-blocker ~x37; Save door x≲20 |
| Stationary missile clear | R17 pure RECON | grounded on runway; **X without LEFT** | `bubble_stationary_missile_clear` | can walk past ~x37 after clear | LEFT+X → KB p138; face-LEFT near door → Save 0xB0DD |
| Walk-brake seat | R17 | target x, true_ground | `bubble_walk_brake_to_x` | |x−target|≤band grounded | overshoot into Save if no brake |
| Run + arm-pump dash | human R15 / TAS arm pump | fire seat; RIGHT+B ± L/R | `bubble_runway_dash` (21f product; 32f = max dash, longer runway only) | post_run ~ (54,395) p9 human | 24–32f from x27 walks off seat |
| Spin-glide | spin preserve | after dash | `bubble_spin_glide` RIGHT+B+A×83 | approach wall with hx | early unspin kills hx |
| R15 consecutive double WJ | crocomi / TAS consecutive WJ | after spin; human **p132** ~(264,297) | `bubble_consecutive_walljumps` / `bubble_double_walljump_r15` (**always 2**) | Phase D x≥300 y≤200 | **single WJ** mx200~251 + lucky Geruta clip only |
| Wall-ready extend | R18 | short of wall | `bubble_wait_wall_ready` (**RIGHT+B+A only**) | ready for open-loop WJ | LEFT+A seek burns WJ1 |
| One WJ pulse | R18 skill | timed / latched | `bubble_walljump_once(WallJumpTiming)` (+ optional delay_into) | height / flip | naive delay on R15 RED |
| Period right WJ climb | place / mid | right structure air | `bubble_period_walljump_climb` | height farm; → double on latch | not Phase D alone from runway |
| Fire-run prepare | R18 | fire seat | `bubble_prepare_fire_run` | Y-clear; optional crouch | face-right×6 walks off max-left |
| Damage boost hold | experiment | pose 137/138 | `bubble_damage_boost_hold` | height from KB | **non-repeatable**; never product |
| Full fire recipe | R15+R19 | fire window seat | `bubble_save_runway_fire_recipe` (phase_wait) | top on pure | fire without phase wait → mx200≈251 |
| Enemy-phase wait | R19 | max-left seat + env RAM | `bubble_wait_fire_phase` / `bubble_fire_phase_geometry` | Geruta 4/6 class A or B | wide live box false-positive; HP=0 alone no |
| Sticky Super door | R19 Phase E | Phase D pin ~(305,141) | `bubble_top_super_door` | ordinary `0xB07A` | walk-right falls; beam swap regress |
| Charged Hi-Jump from lip | R6 pure | solid lip x∈[65,100] y∈[410,450] | mid lip branch / `LIP_*` | height min_y≤280 (pure 260) | lip run-up regress |
| Floor reclimb | R13 | deep y≥480, runway x∈[270,310] | mid floor-reclimb | Phase C usable right contact | marginal y≈429 contact |
| Right shelf LEFT HJ | R9 place | right shelf grounded | mid shelf branch | top band | outer-wall trap if RIGHT |
| Period-8 right WJ | place air | air x≥250 y≤450 | mid climb open-loop | climb progress | not Phase D alone; mx200 pocket |

## Community / external map

| Source | Room / trick | Code mapping | Items |
|--------|--------------|--------------|-------|
| maprando logic room 97 | Running Jump → Right Side Walljump Climb | fire seat + R15 double WJ | Hi-Jump; no Speed |
| crocomi.re / consecutive WJ | frame windows for double WJ | `SAVE_WJ_*` / `SAVE_WJ2_*` | spin pose |
| TASVideos any%/low% Bubble climbs | left runway + wall jumps | human pin states under `scratch/bubble_human_*.state` | route-dependent |
| Speedrun wiki Bubble Mountain | save-door runway climb | `SAVE_RUNWAY_*` | Hi-Jump |

## Phase status (pure, CATH-04 source)

| Phase | Gate | Status |
|-------|------|--------|
| A Mid pin | standing mid | green R5 |
| B Height | min_y≤280 | green R6 (fire path 132 R19) |
| C Right contact | x∈[300,395] y∈[200,430] | green R13 |
| D Top | x≥300 y≤200 | **green pure R19** (enemy-phase fire; min_y≈132) |
| E Bat door | ordinary 0xB07A | **green pure R19** (2012f ×2) |

## Anti-thrash (R17+)

Do **not** without new pin evidence:

1. Another open-loop period / y-band / charge / run-frame tweak on the same arc
2. Place-at-rest (x,y) as proof of natural contact velocity
3. Lip walk-left + dash before HJ (Phase C regress)
4. Left-column top hunt
5. Prefer-save-over-lip always (launched=False regress)
6. Early closed-loop WJ at “wall band” **before** p132 (breaks human open-loop)
7. Shipping **single WJ + enemy damage clip** as product (non-repeatable)
8. Multi-frame bare RIGHT face on max-left fire seat (walks off runway)

**Do** after R19 GREEN:

1. Graph compose Bubble→Bat; pure Bat→Speed Hall from successor
2. Preserve R18/R19 seat + phase wait + sticky door (no enemy RAM patch)

## Debug / pin paths

| Artifact | Role |
|----------|------|
| `scratch/bubble_human_runway.state` | human Phase D isolation |
| `scratch/post_rising_tide_to_bubble_pure.state` | Bubble hop source |
| `scratch/post_bubble_to_bat_pure.state` | Bat successor (R19 GREEN) |
| `scratch/post_bubble_phase_d_pure_r19.state` | Phase D pin for door recon |
| `debug/bubble_to_bat_pure_pin_r19.json` | R19 full pure pin |

## Cross-room reuse

- **Promoted** to `controller_common`: `WallJumpTiming`, `is_wall_latch`,
  `walljump_once`, `consecutive_walljumps` (canonical skill surface)
- **Second consumer:** post-Torizo Parlor Alcatraz left climb
  (`spore_spawn_controller.play_parlor_to_main_shaft` →
  `consecutive_walljumps` / `_PARLOR_CHIMNEY_WJ`)
- Bubble wrappers (`bubble_walljump_once`, …) call shared `walljump_once`
  with track `stop_when`
- Spazer early wall-jump still reuses Bubble consecutive-WJ lessons / timings

## Skill API (shared + Bubble)

| Symbol | Role |
|--------|------|
| `controller_common.WallJumpTiming` | into/amid/flip (+ optional delay_into) |
| `controller_common.walljump_once` | one pulse (room-agnostic) |
| `controller_common.consecutive_walljumps` | N-pulse chain + optional gap |
| `controller_common.is_wall_latch` | pose 132 |
| `R15_DOUBLE` / `bubble_*` wrappers | Bubble Phase D product timings + track |
| `bubble_runway_dash` | dash ± arm-pump |
| `bubble_spin_glide` | preserve spin approach |
| `bubble_wait_wall_ready` | RIGHT+B+A only until wall-ready |
| `bubble_walljump_once` | track-aware one pulse → shared skill |
| `bubble_consecutive_walljumps` | N≥2 chain + follow spin |
| `bubble_double_walljump_r15` | product double + Phase D push |
| `bubble_period_walljump_climb` | right-structure height farm |
| `bubble_prepare_fire_run` | Y-clear / optional crouch; no bare RIGHT walk-off |
| `bubble_save_runway_fire_recipe` | full product compose |
| `bubble_damage_boost_hold` | **experiment** KB→height |

Probe board: [`BUBBLE_RUNWAY_EXPERIMENTS.md`](BUBBLE_RUNWAY_EXPERIMENTS.md).

## Next library targets (after pure Phase D)

- Velocity-matched pure dump + search (R18 card)
- WJ core already in `controller_common` (Parlor second consumer); promote
  runway/dash skills when a second non-Bubble room needs them
