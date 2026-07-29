# Program Status

Last updated: 2026-07-27.

Live facts only. Stable rules live in [BENCHMARK_SPEC.md](BENCHMARK_SPEC.md).
Multi-horizon strategy lives in [ROADMAP.md](ROADMAP.md). The full board is
generated in [GAME_MATRIX.md](GAME_MATRIX.md).

## Goal (one sentence)

Produce verified reset-to-ending clears across a broad **NES + SNES** canonical
library, starting with RAM-aware scripted policies and hardening toward cleaner
runtime classes.

## Flagship results

| Game | Claim | Labels | Evidence |
|------|-------|--------|----------|
| TMNT IV | Continuous hard-mode power-on → staff/cast credits in **00:57:19.635** | Bronze / Resource-assisted + Protection-assisted · M8 (4,667 damage; 0 life losses; Stage1 heal=none segment clear) | [dry manifest](../tmnt_iv/recordings/tmnt_iv_full_hard_dry_run.json), [prior video](../tmnt_iv/recordings/tmnt_iv_full_hard_credits.mp4), [Stage1 probes](../tmnt_iv/recordings/stage1_clean_track/stage1_probes.json) |
| Great Waldo Search | Continuous power-on → five-scrolls ending | Bronze / Clean · M8 | [video](../great_waldo_search/recordings/great_waldo_search_full_credits.mp4) |
| Super Metroid | Continuous power-on → Super; controller to Big Pink main; PB sill entry + mid-maze collect (approach/maze bridges remain); 79/107 path rooms open | Bronze / Resource-assisted · M5 | [manifest](../super_metroid/recordings/start_to_supers.json), [path board](../super_metroid/docs/research/PATH_ROOM_BOARD.md), [assist contract](../super_metroid/docs/ASSIST_CONTRACT.md) |
| Metroid (NES) | Continuous power-on → Maru Mari (Morph Ball); isolated Level1→morph also clear | Bronze / Clean · M5 | [natural](../metroid/recordings/morph_ball_natural.json), [isolated](../metroid/recordings/morph_ball_isolated.json) |

TMNT IV is the reference **linear combat** clear, not “rank 3.” Two continuous
verified clears exist today; the roadmap target is many more at M8 across both
platforms.

## Active near-term trunks

1. **Final Fight** (M3) — generalize the TMNT combat stack toward a continuous clear
2. **Magical Quest** (M2) / **Joe & Mac** (M2) — first natural-entry platformer segments
3. **Super Metroid** (M5) — extend the verified continuous suffix toward ending
4. **NES parallel** — top-10 boot-verified; **Zelda I M5** (Level 1 room 0x54
   cleared) and **Metroid M5** (morph); TMNT II M3; advance SMB,
   Mega Man 2, Punch-Out

Also: Super Double Dragon (M3) and Rival Turf (M2) in parallel with Final Fight.

Battle Clash remains `blocked: infrastructure` (no Super Scope injection).

## Immediate next actions

| Priority | Work |
|----------|------|
| Final Fight | Natural-entry hardening + Stage 3 continuity → chain toward continuous dry-run |
| Magical Quest / Joe & Mac | First reliable room/segment clears with natural entry |
| Super Metroid | Close remaining critical path rooms (PB sill / maze) and inventory bridges |
| NES | Zelda I Clean power-on→Triforce shard 1 done; next: route to Level 2; SMB 1-1, MM2 Air Man, Glass Joe |
| Hygiene | Regenerate matrix + update local `STATUS.md` after every verified advance |
| Assists | Explicit `ASSIST_CONTRACT.md` before any assisted published result |

## NES library foothold

Platform ROM storage: `roms/Nintendo/NES/` (SNES also at
`roms/Nintendo/SNES` → `roms/Super Nintendo`). NES is a first-class parallel
track under the same M0–M8 ladder — see [ROADMAP.md](ROADMAP.md).

### Boot-verified (M1) — top-10 automation targets + prior foothold

| Game | Directory | M1 evidence |
|------|-----------|-------------|
| Super Mario Bros. | `smb/` | **M8** Clean power-on → 8-4 ending 3/3 + video capture |
| Mega Man 2 | `mega_man_2/` | Air Man stage playable |
| Mike Tyson's Punch-Out!! | `punch_out/` | Glass Joe KD1 done; bout win next |
| Contra | `contra/` | Stage 1 playable |
| Kirby's Adventure | `kirby_adventure/` | Vegetable Valley hub |
| TMNT | `tmnt_i/` | Area 1 overworld map control |
| TMNT II | `tmnt_ii/` | Stage 1 combat control → **M3** first wave (score≥5) |
| TMNT III | `tmnt_iii/` | Stage 1 playable control |
| DuckTales | `ducktales/` | Land select control |
| Castlevania | `castlevania/` | Stage 1 playable |
| Super Mario Bros. 3 | `smb3/` | World 1-1 clear (natural entry) |
| Zelda I | `zelda_i/` | **M5** Clean power-on → Level 1 Triforce shard 1 (2/2) |
| Zelda II | `zelda_ii/` | North Palace control |

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
| Linear combat | Final Fight M3→M4 | Natural-entry and Stage 3 continuity |
| Platforming | Magical Quest / Joe & Mac M2→M3 | First room/segment clears |
| Graph navigation | Super Metroid M5→M6; Zelda I M5 | SM: PB sill/maze; Zelda: Level 1 done, route completion warp → Level 2 (`adventure_common`) |
| Continuous control | F-Zero / Pilotwings M2→M3 | First lap / lesson objective |
| NES top-10 | M1→M3+ | TMNT II M3 + Zelda I M5 + SMB M4 (warp→W4) done; remaining: MM2, Glass Joe, pure continuous SMB 1-2, … |
| Planning | Harvest M3 | Crop close-loop (plant/harvest income); then summer natural-entry |

## Capability phase focus

| Phase | Status |
|-------|--------|
| 0 Harness validation | Great Waldo verified; fighters supply match fixtures; NES harness parity (fceumm) |
| 1 Linear full-game clears | TMNT IV done; Final Fight / SDD / Rival Turf in flight; NES TMNT/Contra/Punch-Out at M1 |
| 2 Continuous control | Boot/instrumentation only; Battle Clash blocked |
| 3 Platforming | Magical Quest / Joe & Mac / SMW / DKC instrumented; NES SMB at M4; SMB3 at M3; MM2 / Kirby / DuckTales / Castlevania at M1 |
| 4 Graph exploration | Super Metroid leading (M5); Zelda I at M5; `alttp/` active; Zelda II at M1 |
| 5–7 Campaigns / planning / procedural | Harvest is the Phase 6 foothold; later research |

## Success snapshot (see roadmap for full metrics)

| Metric | Current |
|--------|---------|
| Continuous verified clears (M8) | 2 (TMNT IV, Great Waldo Search) |
| Games at M5+ | Super Metroid (M5), plus the two M8s |
| Shared packages with ≥2 consumers | `retro_harness`, `snes_oneshot`, `fighters_common`, `platformer_common` (growing) |
| Preferred publication class | Clean; assists only with contracts |

## Directory name authority

| Use | Do not use |
|-----|------------|
| `super_metroid/` | `super_metroid_rl/` |
| `SMW/` | `super_mario_bros/` for Super Mario World |
| `smb/` / `smb3/` | NES Super Mario Bros. / Super Mario Bros. 3 |
| `harvest/` | — |
| `alttp/` | — |
| `tmnt_i/` … `tmnt_iv/`, `zelda_i/`, `zelda_ii/` | — |

## Next documentation / tooling checks

```bash
uv run python docs/generate_game_matrix.py
uv run pytest tests/test_docs.py -q
```
