# Plan: SM Rando

Verified facts: [STATUS.md](STATUS.md).

## North star

Clear **S of T** Super Metroid randomizer seeds within budget with a reactive
solver (skills + item-logic planner). Single-game first, then transfer
patterns to SMZ3. No portals and no dual world here.

## Why before SMZ3

| | SM rando | SMZ3 |
|--|----------|------|
| Worlds | 1 | 2 + portals |
| Logic | SM item pool | Combined pool |
| Skills | `super_metroid` only | SM + ALTTP |
| Failure modes | SM softlocks | + Z3 + portal settle |

## Closed

Scaffold, vanilla FirstPlay boot, the three-edge vertical slice, the
power-on Morph policy, the Landing corpus and candidate timing BC, and the
fixture dry ship-to-morph S/T are done (2026-08-09). Evidence:
`recordings/vertical_slice.run.json`, `recordings/policy_to_morph.json`,
`recordings/landing_entry_corpus.json`,
`recordings/landing_entry_baseline.json`,
`recordings/landing_wait_bc_experiment.json`,
`recordings/early_tip_seed_campaign.json`,
`recordings/ship_to_morph.evidence.json`. The integration ROM is still
vanilla. The Landing BC result is a candidate only.

## Open

1. Wire a real rando generator or IPS patch into seed packages.
2. Harvest a second independent Ceres-to-Landing predecessor trajectory and
   replicate the held-out Landing timing-BC result before any deploy.
3. Live multi-seed morph tip (`--mode live`) only after patched ROMs exist.
   Do not claim shuffled-seed robustness until then.
4. Expand the logic graph and bind remaining edges to vanilla skills.
5. Shared L4 planner hookup with SMZ3 after the live tip exists.

```bash
SDL_VIDEODRIVER=dummy uv run python -m sm_rando.scripts.run_early_tip_campaign --mode dry
# live, once generator ROMs exist:
# SDL_VIDEODRIVER=dummy uv run python -m sm_rando.scripts.run_early_tip_campaign --mode live
uv run python -m sm_rando.scripts.run_landing_bc_experiment
```

## Reuse

- Import from `super_metroid` and `retro_harness.adventure`.
- Do not copy room policies into this tree.
- `ship_to_morph` dispatches to
  `super_metroid.routes.kpdr.early_spine:play_ship_to_morph`.
- Item logic stays compatible with the shared L4 solver epic (`rr-gbd`).

## Out of scope

- Area or boss rando (item rando first)
- Multiworld
- Full VARIA ruleset parity (grow the graph from play, not from a wiki dump)
- Treating the vanilla substrate or the BC candidate as shuffled-seed proof
