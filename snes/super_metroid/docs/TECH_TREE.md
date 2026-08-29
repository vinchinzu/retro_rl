# Map Rando tech tree → bot builders

Catalog of Super Metroid **tech** (skills) used by Map Rando logic, wired to
our bot’s room-optimization builders.

| Source | Role |
|--------|------|
| [maprando.com/logic](https://maprando.com/logic) | Difficulty tiers + strat demos |
| [maprando.com/logic/tech/23](https://maprando.com/logic/tech/23) | Example tech page (`canStopOnADime`) |
| `refs/sm-json-data/tech.json` | Vendored tech tree (categories + extension deps) |
| `maps/maprando_tech_catalog.json` | On-disk catalog + bot status |
| `rooms/tech_catalog.py` | Loader / rebuild API |
| `routes/skills/builders.py` | Tech name → callable builder registry |
| `routes/skills/basic_moves.py` | Thin Basic/Medium movement builders |

```bash
# Rebuild catalog from sm-json-data + embedded Map Rando difficulties
uv run python snes/super_metroid/scripts/export/maprando_tech_catalog.py --summary --builders
```

## Counts (full tree)

~**242** techs in sm-json-data (+ Map Rando-only extras such as
`canHyperGateShot`). Difficulty is a **Map Rando UI label**, not a field in
sm-json-data.

| Tier | ~Count | Builder policy |
|------|--------|----------------|
| Implicit | 9 | **core** — always assume in logic |
| Basic | 5 | **core** — bot must execute as reusable skills |
| Medium | 17 | **try** — room-opt builders when useful |
| Hard / Very Hard | ~77 | **later** — only when a route demands it |
| Expert → Beyond | rest | **out_of_scope** for reactive builders first (human tape / TAS) |
| Ignored | 2 | never in logic |

## Builder policy

For **room optimization** (hop hill-climb / skill bank), only spend engineering
on:

1. **core** = Implicit + Basic, plus project-core `canMoonwalk` / `canMoonfall`  
2. **try** = Medium  

Higher tiers enter via **human tape** or **TAS hop import**, not by writing
one-off Expert+ reactive controllers first.

## Core + try matrix (bot status)

Status legend:

- **green** — reusable skill API, used in product/practice  
- **partial** — thin builder or room-specific only; needs generalization  
- **missing** — no bot path yet  

### Implicit (core)

| Tech | Status | Bot surface |
|------|--------|-------------|
| `canDash` | green | `basic_moves.dash`, `runway_dash` |
| `canTrivialMidAirMorph` | green | `basic_moves.mid_air_morph` / `ensure_morph` |
| `canStopOnADime` | partial | `basic_moves.stop_on_a_dime` |
| `canTurnaroundSpinJump` | partial | walljump / runway |
| `canUseEnemies` | partial | enemy snap + damage boost hold |
| `canEscapeEnemyGrab` | partial | `knockback.escape_kb` |
| `canSpecialBeamAttack` | partial | combat charge |
| `canUseGrapple` | partial | item capability only |
| `canTrivialUseFrozenEnemies` | **missing** | freeze platforms |

### Basic (core)

| Tech | Status | Bot surface |
|------|--------|-------------|
| `canWallJump` | green | `skills/walljump` |
| `canShinespark` | green | `skills/shinespark` |
| `canMidAirMorph` | green | `ensure_morph` / morph_bomb |
| `canHeatRun` | partial | K4 heat frame budgets |
| `canUseFrozenEnemies` | **missing** | ice freeze builder |

### Project-core (Map Rando Hard / Very Hard, bot still builds them)

Moonfall is a first-credits skill, not a rando-tier luxury. Map Rando
labels stay Hard / Very Hard; `PROJECT_CORE_TECHS` promotes the builders.

| Tech | Map Rando | Status | Bot surface |
|------|-----------|--------|-------------|
| `canMoonwalk` | Hard | partial | `ram.set_moonwalk` (`$09E4`) + `skills/moonfall.moonwalk_buttons` |
| `canMoonfall` | Very Hard | partial | `skills/moonfall.initiate_moonfall`; Climb warp-pin **503f** vs seed 895f (`kpdr/climb_descent`); Parlor handoff **1067f** vs seed 1095f (`kpdr/parlor_descent`). Both `*_MOONFALL_ON_CLEAN` still False |

Wiki: [Moonwalk / Moonfall](https://wiki.supermetroid.run/Moonwalk). Climb
`0x96BA` and Parlor `0x92FD` first descents on the Morph path (clean poke
on, restore off so later seeds stay valid). Probes:
`scripts/probe/climb_descent.py`, `scripts/probe/parlor_descent.py`.

### Medium (try)

| Tech | Status | Bot surface |
|------|--------|-------------|
| `canPreciseWallJump` | green | `walljump_once` + tight timing |
| `canConsecutiveWallJump` | green | `consecutive_walljumps` |
| `canHorizontalShinespark` | green | `activate_shinespark` |
| `canShinechargeMovement` | green | `charge_until_boost` |
| `canMidairShinespark` | green | `store_then_spin_unspin_activate` |
| `canCrouchJump` | partial | `basic_moves.crouch_jump` |
| `canDownGrab` | partial | `basic_moves.down_grab` |
| `canSpeedyJump` | partial | `basic_moves.speedy_jump` |
| `canIBJ` | partial | `morph_bomb_hole_climb` (not general height IBJ) |
| `canBombHorizontally` | partial | morph roll / bomb window |
| `canCarefulJump` | partial | geometry bands / lip timing |
| `canAwakenZebes` | partial | continuous boot |
| `canSuitlessMaridia` | partial | human tape |
| `canSpaceJumpWaterBounce` | **missing** | |
| `canDisableEquipment` | **missing** | pause equip |
| `canGravityJump` | **missing** | |
| `canSpringBallJumpMidAir` | **missing** | |

## Registered builders

```python
from super_metroid.routes.skills.builders import (
    list_builder_skills,
    builder_skill,
    builder_gap_report,
)
from super_metroid.rooms.tech_catalog import builder_coverage_summary

list_builder_skills()           # executable registry
builder_skill("canShinespark")  # → BuilderSkill
builder_gap_report()            # catalog targets missing from registry
builder_coverage_summary()      # green/partial/missing names
```

Use builders as **mutation operators** for hop optimize (hill-climb), not as
full room AIs: pin → try builder with budget → accept if leave kinematics /
progress improve.

## Categories (sm-json-data)

General · Movement · Jumps · Bomb Jumps · Enemy-Dependent · Shots ·
Speed Booster · XRay · Meta · Miscellaneous

Extension techs nest under parents in `tech.json` (`extensionTechs`). Map Rando
tech pages list strats that **directly** require the tech (not transitive
deps) — see the note at the bottom of [the logic index](https://maprando.com/logic).

## Next builder work (priority)

1. **General IBJ height skill** (`canIBJ` → green) — parameterized bomb period +
   height target; current climb is hole-specific.  
2. **Ice freeze platform** (`canUseFrozenEnemies`) — Basic gap.  
3. **Mockball** (`canMockball`, Hard) — door speed carry; high value for room
   PB but one tier above “try”.  
4. **Short charge / stutter** already green in shinespark (Map Rando places
   some of that under Hard/Expert — we still use it for Moat/WO).  
5. Wire `builder_gap_report()` into hop-optimize so missing tech is explicit
   when a strat requires it.

## Related

- Room names: `docs/RUN_TIMING_AND_SKILL_BANK.md`  
- Skill bank hops: `skill_bank.py`  
- Process: `AGENTS.md` (pure-first)  
- Library → bank → reactive: beads `rr-nzrg` (skill index `rr-nzrg.4`)
