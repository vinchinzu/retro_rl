# Boss pipeline — catalog → strategy → continuous

Authoritative process for every Super Metroid boss fight on Track B
(played continuous spine). Complements
[`ARCHITECTURE.md`](ARCHITECTURE.md) Segment contracts.

## Critical rule

Boss scripts stay **deferred** until **natural entry** to that boss room
exists on the played continuous chain. Continuous acceptance still requires:

- natural boss / event flags (no forged bits)
- zero forbidden progression / capacity / item writes
- zero save-state loads on the continuous path

Never write boss, event, or item RAM to claim a win.

## Current foundation (`combat/`)

| Module | Role |
|--------|------|
| `features.py` | `BossCatalogEntry` registry, `CombatFeatures`, 14-dim vector, AABB |
| `protocol.py` | `BossStrategy` / `BossEvidence` / `BossSegment` contracts |
| `primitives.py` | Shared lane / spray / phase / closeout helpers |
| `actions.py` | Shared discrete combat action table (RL + distillation) |
| `env.py` | Gymnasium env for structured (non-pixel) RL |
| `bomb_torizo.py` | Deterministic Torizo strategy + evidence |
| `kraid.py` | Kraid fight → rear door → Varia (living closeout template) |
| `natural_entry.py` | Natural activation capture harness |
| `docs/ARCHITECTURE.md` | Layer + segment contracts |

**Living templates**

| Boss | Status | Notes |
|------|--------|-------|
| Spore Spawn | Continuous (assisted bounce) + no-assist policy in progress | Survival floor-bounce: `combat.spore_spawn.play_spore_spawn_floor_bounce`. Approach/exit hop: `routes/kpdr/spore_spawn.py`. Clean policy: `play_spore_spawn_fight` (left-ledge ball + 2-missile windows). Residual: `docs/tasks/SPORE_NOASSIST.md` |
| Bomb Torizo | Continuous (hash-pinned replay) | Strategy + natural prove + PPO scaffolding |
| Kraid | Wired continuous tip (`--to varia`); verify STATUS after green report | Full fight + Varia closeout |

## What to build *before* any new boss strategy

### 1. Unified boss catalog & features

Single registry of `BossCatalogEntry` for **every** boss (see
`combat/features.py` → `BOSS_CATALOG` / `get_boss_catalog`):

- room ID(s), max HP (body / phases), contact damage
- hitbox dimensions (sm-json-data + live probes)
- idle vs active spritemaps, invuln notes
- primary / secondary weapons, recommended loadout
- defeat condition (HP zero + boss bit / event flag)
- post-fight closeout (door, item PLM, fanfare, next room)
- multi-phase specs where needed (Phantoon, Mother Brain, Ridley, Draygon)

Keep the 14-dim feature vector stable so any boss plugs into the same Gym
env or strategy controller. Use `validate_live_enemy` (catalog helper) to
confirm room + HP range + spritemap against live RAM.

### 2. Formal `BossStrategy` / `BossSegment` protocol

```text
class BossStrategy(Protocol):
    boss_id: str
    catalog: BossCatalogEntry
    entry: StateRequirement          # natural doorway / activation
    exit: ProgressCondition          # boss bit + optional item
    def play(self, session) -> BossEvidence: ...
```

- `BossEvidence` is compatible with continuous hop / segment reporting.
- Every strategy must run from a pure natural-entry state (no door-warp,
  no manual placement for continuous claims).
- Separate concerns inside the strategy:
  1. Activation / settle
  2. Core fight phases
  3. Exit / door
  4. Optional item collect + fanfare

Bosses are just another kind of Segment (see `routes/segment.py`).

### 3. Generalized natural-entry capture

`combat/natural_entry.py`:

- start from continuous tip, doorway-natural state, or named save
- run until boss is *active* (spritemap / AI / full HP filters)
- save scratch state + provenance JSON
- filter room-load garbage (same idea as Torizo capture)
- CLI pattern: `capture-natural`, `prove-natural`

### 4. Shared combat primitives

`combat/primitives.py` — compose strategies instead of new frame loops:

- lane-hold / position windows
- periodic fire + jump (spray)
- weapon select + morph/unmorph (via `controller_common`)
- wait-for-vulnerable / wait-for-HP-zero / wait-for-boss-bit
- multi-phase state machine helpers
- rear-door / blue-door exit patterns
- Chozo / PLM collect patterns (Varia-style)

### 5. Continuous integration hooks

Before writing fight code for boss N:

- progression milestones for defeat + immediate item closeout
- register a boss Segment the same way as a room hop
- define continuous acceptance: natural entry → fight → exit/item → next tip
  with integrity checks
- only promote STATUS after a green continuous report

### 6. Probe CLI template

Standardize on the Kraid / Torizo probe pattern:

```bash
# strategy --state <entry>
# capture-natural
# prove-natural
# closeout variants (e.g. varia)
# optional eval / short train for structured RL
```

### 7. Finish / attach Kraid first (living template)

Complete continuous attachment:

```text
Warehouse → Hi-Jump → Kraid entry → fight → Varia
```

before starting the next bosses. Every subsequent boss should look like a
cleaned-up version of that pattern.

```bash
uv run python snes/super_metroid/scripts/record/continuous.py --to varia --no-video
uv run python snes/super_metroid/scripts/probe/kraid_combat.py varia --state entry
```

## Implementation order (after foundations)

Priority is **KPDR continuous spine + natural-entry availability**, not
difficulty:

| # | Boss | Why this order | Closeout |
|---|------|----------------|----------|
| 1 | **Kraid** | Finish continuous attach (template) | Rear door → Varia |
| 2 | **Phantoon** | Next major spine boss after Alpha PB / ship | WS power restore |
| 3 | **Botwoon** | Maridia gate for Draygon path | Hall exit |
| 4 | **Draygon** | Major + item | Space Jump |
| 5 | **Crocomire** | Often before Ridley; heat / PB relevant | Side, non-blocking |
| 6 | **Ridley** | Lower Norfair major | Escape path |
| 7 | **Golden Torizo** | Optional / speed side; multi-phase practice | Side |
| 8 | **Mother Brain** | Final multi-phase + Zebetites + escape | Escape init |

Crocomire and Golden Torizo can develop in parallel once catalog + protocol
exist; they are less blocking than Phantoon → Botwoon → Draygon → Ridley → MB.

## Per-boss checklist (once foundations exist)

1. Catalog entry + feature support
2. Natural-entry capture (continuous tip or doorway)
3. Deterministic strategy controller (full knowledge)
4. Optional structured RL on the feature vector — only if it improves
   frames/damage and can distill back to deterministic
5. Closeout (door / item / fanfare) as part of the same Segment
6. `prove-natural` evidence
7. Continuous promotion only after the natural-entry chain holds
8. Registration in progression graph + continuous tip system

## Explicit non-goals (until later)

- Vision-only / pixel policies for bosses (parked until Gold)
- Writing boss bits or items to claim progress
- Implementing a boss before natural entry exists on the played chain
- Mixing boss fight logic into room-hop controllers

## Probe commands (current)

```bash
# Bomb Torizo
uv run python snes/super_metroid/scripts/probe/bomb_torizo_combat.py strategy --state BossTorizo
uv run python snes/super_metroid/scripts/probe/bomb_torizo_combat.py capture-natural
uv run python snes/super_metroid/scripts/probe/bomb_torizo_combat.py prove-natural

# Kraid + Varia
uv run python snes/super_metroid/scripts/probe/kraid_combat.py strategy --state entry
uv run python snes/super_metroid/scripts/probe/kraid_combat.py varia --state entry

# Continuous tips (promote STATUS only when integrity green)
uv run python snes/super_metroid/scripts/record/continuous.py --to kraid --no-video
uv run python snes/super_metroid/scripts/record/continuous.py --to varia --no-video
```

## Summary — build order

**Phase 0 (foundations — do first)**

1. Extend BossCatalog + features for the full list
2. Formal `BossStrategy` / `BossSegment` protocol + evidence types
3. Generalized natural-entry capture harness
4. Shared combat primitives + phase-machine helpers
5. Continuous integration hooks (milestones, tip registration, integrity)
6. Probe CLI template + this doc
7. Finish continuous Kraid attachment as the living template

**Phase 1+ (individual policies)**

Phantoon → Botwoon → Draygon → Crocomire → Ridley → Golden Torizo → Mother Brain  
(each following the same pipeline)

This keeps “poking in the dark” cost low: once the skeleton exists, adding a
new boss is mostly catalog data + strategy composition + natural-entry proof,
not a new ad-hoc combat system.
