# SM Rando — Status

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M1 |
| Best verified result | Power-on → Morph Ball through the real `SMRando-Snes` emulator policy, on the vanilla substrate |
| Last verification | 2026-08-09 |
| Runtime class | Bronze |
| Intervention class | Clean |

## Role

**Single-game Super Metroid randomizer** — simpler solver ground than SMZ3.
Build item-logic + room skill hooks + seed-robust spine here, then extend to
combined SMZ3. Vanilla skills live in `snes/super_metroid/` (do not fork).

Program stack: `docs/SOLVER_ARCHITECTURE.md`.

## Checklist

| Item | State |
|------|--------|
| Package `sm_rando/` | scaffolded |
| Seed package schema | done (offline fixture + `seeds/demo_seed/`) |
| Early logic graph | done (coarse; `ship_to_morph` is the first bound edge) |
| Integration ROM (vanilla SM) | done (`setup_rom` → SMRando-Snes/rom.sfc) |
| Boot → FirstPlay | done (`make_boot`, Ceres `0xDF45`) |
| Play/record spine | done (`./play`, record default, F5 → package integration) |
| Ship → Morph vanilla skill binding | natural-entry evidence (1/1 clean); patched-rando coverage still open |
| Three-edge SolverSession slice | verified 2026-08-09: production adapter has three real edges; the experiment-only injected failure triggers one replan, then Landing/Parlor/Climb/Pit completes at frame 23,866. Backend-owned Clean audit retained in `recordings/vertical_slice.run.json`. |
| First-item policy product | verified 2026-08-09: `SMRando-Snes` power-on → Morph Ball succeeds at frame 26,824 in room `0x9E9F`, with zero state loads, zero progression/resource writes, and exact 6,003-frame video evidence in `recordings/policy_to_morph.mp4`. Current ROM is the documented vanilla substrate. |
| Landing EntryStateCorpus | 64 unique real predecessor states; 58 train / 6 held out; both parities, 64 Y positions, 2 subpixel phases. Metadata v2 includes pose reconstructed from each retained state; state blobs live under `custom_integrations/SMRando-Snes/entry_corpus/landing_v1/`. |
| Structured Landing baseline | 0/58 train and 0/6 held-out from unsettled natural entry (gap 0.000); failures retained in `recordings/landing_entry_baseline.json` |
| Landing timing BC experiment | 58/58 train and 6/6 held out; zero eval states used for fitting; backend-owned Clean/Bronze audits + six canonical eval trajectories retained. Candidate only—new predecessor trajectories required before deployment. |
| Patched seed ROM / generator | open |
| Multi-seed S/T dry-run | verified 2026-08-09: `SeedCampaignRunner` dry campaign on fixture seeds 1337/1338/1339, goal `ship_to_morph`, S=2 of T=3 (actual 3/3), claimable, no spoiler oracle, vanilla substrate labeled. Reports: `recordings/early_tip_seed_campaign.json` + classic `early_tip_seed_robustness.json`. Fail-closed INFRA_ERROR path covered in tests. Not shuffled-seed robustness until generator ROMs land. |

## Next

1. Wire real rando generator or IPS patch into seed packages.
2. Harvest a second independent Ceres→Landing predecessor trajectory and
   replicate the 6/6 Landing timing-BC result before promoting the candidate.
3. Live multi-seed morph tip (`--mode live`) once generator/patched ROMs exist;
   SMZ3 multi-seed (`rr-gbd.13`) may proceed using this pattern.
