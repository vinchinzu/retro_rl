# Trigger / Hitbox Handoff: ALTTP Opening Route

Remaining **trigger** (exact interaction) problems so they are not re-discovered
as “route discovery” failures. Route = area; approach = local pocket; trigger =
hitbox / transition / NPC interaction.

See also: `opening_route/anchors.py`, `docs/STATUS.md`.

## Solved (do not re-probe randomly)

### Secret bush hole (entrance 0x7D): **trigger solved**

| Field | Value |
|-------|--------|
| Tier | trigger |
| Anchor | `HyruleCastle_SecretPassageExactTile` |
| Map | `maps/screen_1b_grounds.json` door `secret_hole_to_0x55`; Yaze `0x7D` @ `(2432, 1696)` |
| Approach | `HyruleCastle_SecretPassageApproach` ~`(2430, 1704)` tol 48 (map path, axis-aligned) |
| Script | `SECRET_HOLE_ENTRY_SCRIPT`: face UP, `A`×4, wait 20, `UP`×56 |
| Min measured | UP walk after A/wait ≥ 40 frames |
| Exit RAM | indoors room base `0x55` |
| Provenance | `castle_to_sword` headless 2026-07-29 |

Failure modes already known: position drift on natural chain → use
`BUSH_LIFT_CANDIDATES` fallbacks; do not restart full map search.

### Uncle fighter sword: **interaction solved**

| Field | Value |
|-------|--------|
| Tier | trigger (NPC dialogue) |
| Script | uncle approach + mash until `$F359 >= 1` |
| Post | hold-up-item `$5D==21` → ~95 frames LEFT to dismiss |
| Provenance | `castle_to_sword` |

### Secret-entrance stairs exit: **trigger solved**

| Field | Value |
|-------|--------|
| Tier | trigger |
| Anchor | `HyruleCastle_SecretEntrance_StairsAlign` |
| Align | `(2672, 2916)` tol 6, then DOWN |
| Landing | outdoor pocket ~`(2248, 1755)` screen `0x1B` |
| Soft-lock | off-center deep south `y≥2960` stays indoors |
| Provenance | `secret_entrance_clear.exit_secret_entrance_stairs` 2026-07-30 |

### Courtyard hedge pocket → main castle door: **trigger solved**

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

## On the continuous prefix (not open)

`main_hall_west_to_0x60` and `room_60_north_to_0x50` are continuous inside
`castle_dungeon_prefix`. The 2026-07-31 `CastleMain` / `CastleRoom60` runs
were the first isolated measurements. They are not a second tip.

`room_50_east_to_0x01` is natural_entry (2026-08-02), not continuous. After a
clear, the only forward exit from `0x50` is east to `0x01`. South returns to
`0x60`. No B1 stairs in `0x50`. Map door `east_to_0x01`, approach near
(480, 2680), hold RIGHT.

## Open

The F1 well is not an undiscovered stair. Graph hop `room_01_down_to_0x72`
is natural_entry (hold UP on the north wall). Reverse `room_72_north_to_0x01`
stays isolated. Neither hop is the continuous tip.

`room_01_to_zelda_cell` stays planned. The open red is `0x81` `west_to_0x80`:
a small key did not open the gold jail door on a state load. Big-key bytes
were 0 and were not tried. Detail is `docs/tasks/residual.md`. Do not
STATUS-promote that red. `$F3CC == 1` on `CastleZeldaFollower` is the pin as
loaded, not a rescue.

Escort (`escort_to_sanctuary`) stays planned. Mantle checks lamp plus follower.
Sanctuary room base `0x12` / overworld screen `0x13` is not verified.
The internal key and shutter path in `0x55` is alternate practice only.

## Multi-truth checklist (any new hop)

- [ ] RAM predicate (room/screen + inventory + position window)
- [ ] Map/Yaze association if applicable (entrance id / hole tile)
- [ ] RAM glance leftover (room, module, submodule, xy, sword, `$F3CC`, keys). Not an MP4.
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
