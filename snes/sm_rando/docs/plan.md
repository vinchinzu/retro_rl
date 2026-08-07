# SM Rando — Plan

## North star

Clear **S of T** Super Metroid randomizer seeds within budget with a reactive
solver (skills + item-logic planner). Single-game first — no portals, no dual
world — then transfer patterns to SMZ3.

## Why before SMZ3

| | SM rando | SMZ3 |
|--|----------|------|
| Worlds | 1 | 2 + portals |
| Logic | SM item pool | Combined pool |
| Skills | `super_metroid` only | SM + ALTTP |
| Failure modes | SM softlocks | + Z3 + portal settle |

## Phases

1. **Scaffold (M0)** — package, seed schema, coarse logic graph, play spine. *(done)*
2. **Boot (M1)** — power-on vanilla ROM → FirstPlay (Ceres). *(done; patched seeds next)*
3. **Logic grounding** — expand graph; probe transitions under controlled inventories.
4. **Skill bind** — each edge owns a vanilla (or synthesized) skill policy.
5. **Seed-robust early tip** — e.g. ship → morph → bombs across T seeds.
6. **Extend toward SMZ3** — shared L4 planner + seed-robust harness.

## Play spine (speed + fun)

```bash
# FirstPlay (Ceres) + record by default
./play
# or: uv run python -m sm_rando.scripts.play

# Practice vanilla SuperMetroid-Snes skills under this package spine
uv run python -m sm_rando.scripts.play --vanilla

# Rebuild boot state
SDL_VIDEODRIVER=dummy uv run python -m sm_rando.scripts.make_boot
```

Sessions write `recordings/play_*.mp4` + `play_*.json` (or spine manifests).
Use those for demos, imitation, and multi-seed aggregation — not fixed tapes.

## Reuse policy

- Import from `super_metroid` and `retro_harness.adventure`.
- Do not copy room policies into this tree.
- Item logic format should stay compatible with the shared L4 solver epic
  (`rr-gbd`).

## Out of scope (for now)

- Area/boss rando (start with item rando only)
- Multiworld
- Full VARIA ruleset parity (grow graph from play, not from wiki dump alone)
