# ALTTP Rando — Plan

## North star

Clear **S of T** ALTTP randomizer seeds within budget with a reactive solver.
Single-game first — then transfer item-logic + discovery patterns to SMZ3.

## Why before SMZ3

| | ALTTP rando | SMZ3 |
|--|-------------|------|
| Worlds | ALTTP only | SM + ALTTP + portals |
| Logic | ALTTPR item pool | Combined |
| Skills | `alttp` only | both trees |
| Failure modes | dungeon/OW softlocks | + portal settle |

## Phases

1. **Scaffold (M0)** — package, seed schema, coarse logic graph, play spine. *(done)*
2. **Boot (M1)** — JP 1.0 power-on → `FirstPlay` controllable Link. *(done)*
3. **Logic grounding** — expand graph from play + public ALTTPR logic notes.
4. **Skill bind** — edges → vanilla opening / dungeon skills from FirstPlay.
   (`house_to_uncle` natural_entry done 2026-08-09; next edges planned.)
5. **Seed-robust early tip** — e.g. house → sanctuary or Eastern across T seeds.
6. **Extend toward SMZ3** — shared L4 + seed-robust harness.

## Play spine

```bash
./play
uv run python -m alttp_rando.scripts.play
uv run python -m alttp_rando.scripts.play --vanilla   # USA alttp skills
# After seed ROM: --seed test_seed
```

Sessions write `recordings/play_*.json` / `spine_alttp_rando_*.json` (+ MP4).

## Reuse policy

- Import from `alttp` and `retro_harness.adventure`.
- Do not copy room policies into this tree.
- Align capability tokens with shared L4 solver (`rr-gbd`).

## Out of scope (for now)

- Entrance / door randomizer (start with item rando)
- Multiworld
- Full ALTTPR glitch logic
