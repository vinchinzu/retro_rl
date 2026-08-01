# Trigger / Hitbox Handoff — ALTTP Opening Route

Remaining **trigger** (exact interaction) problems so they are not re-discovered
as “route discovery” failures. Route = area; approach = local pocket; trigger =
hitbox / transition / NPC interaction.

See also: `opening_route/anchors.py`, `docs/STATUS.md`,
`ARCHITECTURE_AND_CLEANUP_PLAN.md` (lessons learned).

## Solved (do not re-probe randomly)

### Secret bush hole (entrance 0x7D) — **trigger solved**

| Field | Value |
|-------|--------|
| Tier | trigger |
| Anchor | `HyruleCastle_SecretPassageExactTile` |
| Map | Yaze entrance `0x7D` @ world `(2432, 1696)` |
| Approach | `HyruleCastle_SecretPassageApproach` ~`(2430, 1704)` tol 48 |
| Script | `SECRET_HOLE_ENTRY_SCRIPT`: face UP, `A`×4, wait 20, `UP`×56 |
| Min measured | UP walk after A/wait ≥ 40 frames |
| Exit RAM | indoors room base `0x55` |
| Provenance | `castle_to_sword` headless 2026-07-29 |

Failure modes already known: position drift on natural chain → use
`BUSH_LIFT_CANDIDATES` fallbacks; do not restart full map search.

### Uncle fighter sword — **interaction solved**

| Field | Value |
|-------|--------|
| Tier | trigger (NPC dialogue) |
| Script | uncle approach + mash until `$F359 >= 1` |
| Post | hold-up-item `$5D==21` → ~95 frames LEFT to dismiss |
| Provenance | `castle_to_sword` |

### Secret-entrance stairs exit — **trigger solved**

| Field | Value |
|-------|--------|
| Tier | trigger |
| Anchor | `HyruleCastle_SecretEntrance_StairsAlign` |
| Align | `(2672, 2916)` tol 6, then DOWN |
| Landing | outdoor pocket ~`(2248, 1755)` screen `0x1B` |
| Soft-lock | off-center deep south `y≥2960` stays indoors |
| Provenance | `sword_to_zelda.exit_secret_entrance_stairs` 2026-07-30 |

### Courtyard hedge pocket → main castle door — **trigger solved**

| Field | Value |
|-------|--------|
| Tier | route + approach + trigger |
| Anchors | `HyruleCastle_Courtyard_OpenGardens`, `HyruleCastle_MainDoorApproach`, `HyruleCastle_MainDoorTrigger`, `HyruleCastle_MainHall` |
| Route | bush-cut S/W out of pocket (walk-only boxed ~48×64); gardens then south corridor y≈2024 |
| Approach | ~(2040, 1790) tol 24 (CastleMain exit landing ~(2040, 1779)) |
| Trigger | align x≈2040, hold UP → room base `0x61` |
| Graph edge | `pocket_to_main_hall` (`continuous`) |
| Script | `alttp.opening_route.pocket_to_main_hall` |
| Provenance | headless 2026-07-30 from stairs-exit / FighterSword predecessor |

Failure modes: UP at pocket re-enters secret stairs; west-only from gardens is
blocked by water until south corridor is reached; soldiers on approach path.

## Open (active blockers)

### Main hall room 0x61 — **west edge measured**

| Field | Value |
|-------|--------|
| Tier | route + approach + trigger (main hall only) |
| Anchors | `HyruleCastle_MainHall`, `…_WestDoorApproach`, `…_WestDoorTrigger` |
| Entry | `CastleMain` ~(760, 3520); 3 hostiles on carpet |
| Map | `maps/room_61.json`; side corridor **y≈3320** |
| West edge | approach ~(520, 3320), hold LEFT → room `0x60` landing ~(511, 3320) |
| East edge | approach ~(960, 3320), hold RIGHT → room `0x62` |
| South edge | approach ~(760, 3496), hold DOWN → outdoors ~(2040, 1740) screen `0x1B` |
| Tools | `room_sense` + `room_engine` |
| Script | `scripts/room_engine.py run room_61 --edge west_to_0x60` |
| Graph | `main_hall_west_to_0x60` verification=`isolated` |
| Provenance | headless 2026-07-31 from `CastleMain` |

### Room 0x60 north → 0x50 — **isolated** (Zelda still open)

| Field | Value |
|-------|--------|
| Tier | route + approach + trigger (0x60 only) |
| Anchors | `HyruleCastle_MainWest_0x60`, `HyruleCastle_NW_0x50` |
| Map | `maps/room_60.json` |
| Path | west landing → (400,3320) → (376,3200) → (376,3130) UP |
| North edge | approach ~(376, 3130), hold UP → room `0x50` landing ~(376, 3088) |
| East edge | approach ~(500, 3320), RIGHT → `0x61` |
| South edge | x≈376 south → outdoors west courtyard ~(1832, 1540) |
| Script | `scripts/room_engine.py run room_60 --edge north_to_0x50 --state CastleRoom60` |
| Graph | `room_60_north_to_0x50` verification=`isolated` |

### After 0x50 → Zelda cell — **planned**

Measured map seeds (not continuous): `room_50` east→`0x01`, plus B1
`room_81`/`room_82`/`room_72`/`room_71`/`room_80`. Graph hop
`room_50_to_zelda_cell` remains `planned`.

Continuous tip remains **main hall** until natural-entry reaches further.
Work queue primary: `CastleMain`, `CastleRoom60`, `CastleRoom50`, Zelda B1 states.
Internal key/shutter path in/near `0x55` is **alternate** only.

Acceptance for full rescue: `$F3CC == 1`. Do not claim from room id alone.

### Escort Lamp + sewers → Sanctuary — **planned**

Segment scaffold: `escort_to_sanctuary`. Mantle checks lamp + follower.
Natural boot already collects house lamp. Sanctuary room base `0x12` / OW
screen `0x13` — confirm on ROM before claims.

## Multi-truth checklist (any new hop)

- [ ] RAM predicate (room/screen + inventory + position window)
- [ ] Map/Yaze association if applicable (entrance id / hole tile)
- [ ] Screenshot artifact path under `recordings/`
- [ ] Named anchor in `opening_route/anchors.py` (semantic id)
- [ ] Graph edge verification: `planned` → `isolated` → `natural_entry` → `continuous`
- [ ] Segment registered only when entry/exit contracts are honest

## State semantics (common confusion)

| Filename | Means | Does **not** mean |
|----------|--------|-------------------|
| `HyruleCastleGrounds` | Controllable on screen `0x1B` spawn | Bridge turn east / hole approach |
| `FighterSword` | Room `0x55` post-uncle (dev load) | Natural-chain continuous proof |
| `Castle_55` | Ambiguous chamber in `0x55` | Specific uncle/south/keyed node |

Prefer semantic anchor ids in docs and benchmarks
(`HyruleCastle_SecretPassageApproach`, etc.). Keep short filenames for retro
integration.

## Anti-patterns

- Treating “reached screen `0x1B`” as the secret-hole approach.
- Re-running global bush searches after the proven trigger exists.
- Mixing gauntlet/romhack experiments into opening-route evidence.
- Publishing continuous claims from state-load runs without `--natural`.
