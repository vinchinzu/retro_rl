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
5. **Seed-robust early tip** — ship → morph across T fixture seeds via
   `SeedCampaignRunner` dry-run *(done 2026-08-09; live/patched next)*.
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

The first solver composition proof is:

```bash
SDL_VIDEODRIVER=dummy uv run python -m sm_rando.scripts.run_vertical_slice
```

It naturally enters Landing Site, injects a retryable primary-edge failure,
replans, then dispatches the real Landing → Parlor → Climb → Pit vanilla
controllers. The fail-closed contract, audit, solver trace, and failed outcome
are retained in `recordings/vertical_slice.run.json`; exact macro actions and
the recovered failure are also retained as canonical trajectories.

The first end-to-end policy product runs the actual `SMRando-Snes` emulator
integration from power-on through Morph Ball and writes package-owned video and
integrity evidence. The current integration is still the documented vanilla
substrate; this proves execution wiring, not patched-seed robustness.

```bash
SDL_VIDEODRIVER=dummy uv run python -m sm_rando.scripts.run_morph_policy
```

## Natural-entry corpus

`recordings/landing_entry_corpus.json` indexes 64 unique states produced by the
real Ceres→Landing predecessor. Raw state blobs stay in the package integration
under `entry_corpus/landing_v1/`. The deterministic hash split is 58 train / 6
held out; the platformer `neuro` entrypoint accepts `--entry-corpus` and loads
only the train partition. The initial structured-policy measurement is 0/58
train and 0/6 held out (gap 0.000), retained in
`recordings/landing_entry_baseline.json`. This measures the actual unsettled-
entry failure rather than treating the settled ship tape as generalized.

The first behavior-cloning experiment learns the expert wait-to-handoff timing
from the 58 train states, then dispatches the unchanged Landing room skill. It
uses no held-out state for fitting and scores 58/58 train plus 6/6 eval on the
real ROM, versus the fixed-tape baseline's 0/58 and 0/6. The checkpoint,
PolicyArtifact, audited report, and six eval trajectories are retained under
`models/` and `recordings/`. This is a **candidate-only** result: do not deploy
or claim general robustness until it replicates on newly harvested predecessor
trajectories.

```bash
uv run python -m sm_rando.scripts.run_landing_bc_experiment
```

```bash
uv run python -m retro_harness.platformer.cli \
  --level sm_rando_landing_entry neuro \
  --entry-corpus snes/sm_rando/recordings/landing_entry_corpus.json
```

## Reuse policy

- Import from `super_metroid` and `retro_harness.adventure`.
- Do not copy room policies into this tree.
- The first binding, `ship_to_morph`, dispatches to
  `super_metroid.routes.kpdr.early_spine:play_ship_to_morph`; its retained
  vanilla natural-entry evidence is `recordings/ship_to_morph.evidence.json`.
- Item logic format should stay compatible with the shared L4 solver epic
  (`rr-gbd`).

## Out of scope (for now)

- Area/boss rando (start with item rando only)
- Multiworld
- Full VARIA ruleset parity (grow graph from play, not from wiki dump alone)
