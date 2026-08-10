# Program Status

Last updated: 2026-08-06.

Live facts only. Stable rules live in [BENCHMARK_SPEC.md](BENCHMARK_SPEC.md).
Multi-horizon strategy lives in [ROADMAP.md](ROADMAP.md). Solver layer stack
lives in [SOLVER_ARCHITECTURE.md](SOLVER_ARCHITECTURE.md). The full board is
generated in [GAME_MATRIX.md](GAME_MATRIX.md).

## Goal (one sentence)

Produce verified reset-to-ending clears across a broad **NES + SNES** library
and a **reactive game solver** (skills + planning + discovery) that generalizes
to randomizers — starting with RAM-aware skill policies and hardening toward
cleaner runtime classes.

## Flagship results

| Game | Claim | Labels | Evidence |
|------|-------|--------|----------|
| TMNT IV | Continuous hard-mode power-on → staff/cast credits in **00:57:19.635** | Bronze / Resource-assisted + Protection-assisted · M8 (4,667 damage; 0 life losses; Stage1 heal=none segment clear) | [dry manifest](../snes/tmnt_iv/recordings/tmnt_iv_full_hard_dry_run.json), [prior video](../snes/tmnt_iv/recordings/tmnt_iv_full_hard_credits.mp4), [Stage1 probes](../snes/tmnt_iv/recordings/stage1_clean_track/stage1_probes.json) |
| Great Waldo Search | Continuous power-on → five-scrolls ending | Bronze / Clean · M8 | [video](../snes/great_waldo_search/recordings/great_waldo_search_full_credits.mp4) |
| Super Metroid | Continuous power-on → **Varia Suit** (KPDR K3 tip integrity GREEN 2026-08-01; best published ~101,954f / multi-run 104,382f); post-Varia reverse pure + K4 scaffolds; dual-track room farm Wave-10 closed | Bronze / Resource-assisted · M5 | [manifest](../snes/super_metroid/recordings/start_to_varia.json), [STATUS](../snes/super_metroid/docs/STATUS.md), [plan](../snes/super_metroid/docs/plan.md), [assist contract](../snes/super_metroid/docs/ASSIST_CONTRACT.md) |
| Metroid (NES) | Continuous power-on → Maru Mari (Morph Ball); isolated Level1→morph also clear | Bronze / Clean · M5 | [natural](../nes/metroid/recordings/morph_ball_natural.json), [isolated](../nes/metroid/recordings/morph_ball_isolated.json) |
| SMZ3 | Portal settle → Link's House chest (seed 1337); seed-abstract multi-seed clear **not yet claimed** | Bronze / missile assist on red door · M2→M3 | [STATUS](../snes/smz3/docs/STATUS.md), [plan](../snes/smz3/docs/plan.md) |

TMNT IV is the reference **linear combat** clear. Great Waldo is the **harness
fixture**. Super Metroid + ALTTP + SMZ3 are the **solver flagship triangle**
(skills substrate + combined randomizer proof). Two continuous verified M8
clears exist today; seed-robust S/T claims do not yet.

## Active near-term trunks

1. **Solver stack (flagship)** — L4 logic-graph solver + seed-robustness harness;
   ground on **sm_rando / alttp_rando** then SMZ3
   ([SOLVER_ARCHITECTURE.md](SOLVER_ARCHITECTURE.md))
2. **sm_rando / alttp_rando** — SM-rando M1 now has a real three-edge
   SolverSession slice, EntryStateCorpus, held-out BC candidate, and fixture
   multi-seed early tip S/T dry-run (ship→morph, 3/3 claimable); shuffled-seed
   live S/T remains open. ALTTP-rando remains at its boot/graph rung.
3. **Super Metroid** (M5) — Varia tip green; pure reverse + continuous K4 toward ending
4. **ALTTP / Zelda 3** (M1) — open beyond title→castle; dungeon/item capability edges
5. **SMZ3** (M2→M3) — longer one-bot segments; multi-seed after single-game patterns
6. **Final Fight** (M3) — generalize the TMNT combat stack toward a continuous clear
7. **Magical Quest** (M2) / **Joe & Mac** (M2) — first natural-entry platformer segments
8. **NES parallel** — top-10 boot-verified; **Zelda I M5** and **Metroid M5**;
   TMNT II M3; advance SMB, Mega Man 2, Punch-Out

Also: Super Double Dragon (M3) and Rival Turf (M2) in parallel with Final Fight.

Battle Clash remains `blocked: infrastructure` (no Super Scope injection).

## Immediate next actions

| Priority | Work |
|----------|------|
| Solver | S/T harness + SeedCampaignRunner landed; first consumer sm_rando early tip dry-run; next alttp_rando / SMZ3 multi-seed |
| sm_rando | Fixture multi-seed ship→morph S/T dry-run published; live/patched generator next |
| alttp_rando | M1 seed ROM boot; bind house→uncle; `play --vanilla` for opening practice |
| Super Metroid | Pure reverse post-Varia → Business; continuous K4; dual-track room farm |
| ALTTP | Sword/uncle and early dungeon/overworld skills with capability edges |
| SMZ3 | Longer one-bot SM or Z3 segment + video; multi-seed after single-game rungs |
| Final Fight | Natural-entry hardening + Stage 3 continuity → chain toward continuous dry-run |
| Magical Quest / Joe & Mac | First reliable room/segment clears with natural entry |
| NES | Zelda I Level 2 route; SMB / MM2 / Glass Joe skill work |
| Hygiene | Regenerate matrix + update local `STATUS.md` after every verified advance |
| Assists | Explicit `ASSIST_CONTRACT.md` before any assisted published result |

## NES library foothold

Platform ROM storage: `roms/Nintendo/NES/` (SNES also at
`roms/Nintendo/SNES` → `roms/Super Nintendo`). NES is a first-class parallel
track under the same M0–M8 ladder — see [ROADMAP.md](ROADMAP.md).

### Boot-verified (M1) — top-10 automation targets + prior foothold

| Game | Directory | M1 evidence |
|------|-----------|-------------|
| Super Mario Bros. | `nes/smb/` | **M8** Clean power-on → 8-4 ending 3/3 + video capture |
| Mega Man 2 | `nes/mega_man_2/` | **M3** Air Man screen-4 clear from AirScreen2 (s3/s4 3/3) |
| Mike Tyson's Punch-Out!! | `nes/punch_out/` | Glass Joe KD1 done; bout win next |
| Contra | `nes/contra/` | Stage 1 playable |
| Kirby's Adventure | `nes/kirby_adventure/` | Vegetable Valley hub |
| TMNT | `nes/tmnt_i/` | Area 1 overworld map control |
| TMNT II | `nes/tmnt_ii/` | Stage 1 combat control → **M3** first wave (score≥5) |
| TMNT III | `nes/tmnt_iii/` | Stage 1 playable control |
| DuckTales | `nes/ducktales/` | Land select control |
| Castlevania | `nes/castlevania/` | Stage 1 playable |
| Super Mario Bros. 3 | `nes/smb3/` | World 1-1 clear (natural entry) |
| Zelda I | `nes/zelda_i/` | **M5** Clean power-on → Level 1 Triforce shard 1 (2/2) |
| Zelda II | `nes/zelda_ii/` | North Palace control |

### NES top-10 implementation order (capability pairs)

Not popularity rank — capability diversity for harness transfer with SNES:

1. SMB → platformer base · 2. Mega Man 2 → stage/boss framework ·
3. Punch-Out → opponent FSM · 4. Contra → run-and-gun ·
5. Kirby → forgiving platformer · 6. TMNT II → beat-'em-up (SNES transfer) ·
7. DuckTales → stage-select platformer · 8. Castlevania → deterministic action ·
9. Zelda I → route graph · 10. SMB3 → large platform campaign

## Bottlenecks

| Track | Current gate | Blocker |
|-------|--------------|---------|
| **Solver (L4/L3)** | Bounded planner + bindings + SolverSession have one real SM-rando consumer | Second game consumer; online discovery; seed-robust campaign |
| **Single-game rando** | SM-rando M1 vertical slice + held-out BC candidate; ALTTP-rando M1 | Seed ROM integration and S/T early tips |
| Linear combat | Final Fight M3→M4 | Natural-entry and Stage 3 continuity |
| Platforming | Magical Quest / Joe & Mac M2→M3 | First room/segment clears |
| Graph navigation | Super Metroid M5→M6; ALTTP M1; Zelda I M5 | SM: post-Varia pure/K4; ALTTP: beyond opening; Zelda: Level 2 |
| Randomizer proof | SMZ3 M2→M3 | Prefer single-game S/T first; then multi-seed SMZ3 |
| Continuous control | F-Zero / Pilotwings M2→M3 | First lap / lesson objective |
| NES top-10 | M1→M3+ | TMNT II M3 + Zelda I M5 + SMB M8 done; remaining skill work |
| Planning | Harvest M3 (pioneer trunk) | Crop close-loop (money > $100); skill composition / planning stack |

## Capability phase focus

| Phase | Status |
|-------|--------|
| 0 Harness validation | Great Waldo verified; fighters supply match fixtures; NES harness parity (fceumm) |
| 1 Linear full-game clears | TMNT IV done; Final Fight / SDD / Rival Turf in flight; NES TMNT/Contra/Punch-Out at M1 |
| 2 Continuous control | Boot/instrumentation only; Battle Clash blocked |
| 3 Platforming | Magical Quest / Joe & Mac / SMW / DKC instrumented; NES SMB at M8; SMB3 at M3; MM2 / Kirby / DuckTales / Castlevania at M1 |
| 4 Graph exploration | Super Metroid leading (M5); Zelda I at M5; `snes/alttp/` active (M1); Zelda II at M1 |
| 5–6 Campaigns / planning | Harvest is the Phase 6 pioneer trunk (M3 calendar done; crop income + skill composition next) |
| 7 Adaptive / randomizer | **Elevated:** sm_rando + alttp_rando M0 scaffolds; SMZ3 remains combined proof |

## Success snapshot (see roadmap for full metrics)

| Metric | Current |
|--------|---------|
| Continuous verified clears (M8) | 2 (TMNT IV, Great Waldo Search) |
| Games at M5+ | Super Metroid (M5), Metroid NES (M5), Zelda I (M5), SMB (M8), plus TMNT IV / Waldo M8 |
| Seed-robust S/T claims | **0** (harness not yet built) |
| Shared packages with ≥2 consumers | `retro_harness` core + `platformer` / `fighters` / `adventure` subdomains |
| Preferred publication class | Clean; assists only with contracts |

## Directory name authority

| Use | Do not use |
|-----|------------|
| `snes/super_metroid/` | `super_metroid_rl/` |
| `snes/SMW/` | `super_mario_bros/` for Super Mario World |
| `nes/smb/` / `nes/smb3/` | NES Super Mario Bros. / Super Mario Bros. 3 |
| `snes/harvest/` | — |
| `snes/alttp/` | — |
| `snes/smz3/` | — |
| `snes/sm_rando/` | — |
| `snes/alttp_rando/` | — |
| `nes/tmnt_i/` … `snes/tmnt_iv/`, `nes/zelda_i/`, `nes/zelda_ii/` | — |

## Next documentation / tooling checks

```bash
uv run python docs/generate_game_matrix.py
uv run pytest tests/test_docs.py -q
```
