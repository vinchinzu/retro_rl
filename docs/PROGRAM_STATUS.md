# Program Status

Last updated: 2026-07-25.

Live facts only. Stable rules live in [BENCHMARK_SPEC.md](BENCHMARK_SPEC.md).
The full board is generated in [GAME_MATRIX.md](GAME_MATRIX.md).

## Goal (one sentence)

Produce verified reset-to-ending clears across a broad SNES library, starting
with RAM-aware scripted policies and hardening toward cleaner runtime classes.

## Flagship results

| Game | Claim | Labels | Evidence |
|------|-------|--------|----------|
| TMNT IV | Continuous hard-mode power-on → staff/cast credits in **00:57:19.635** | Bronze / Resource-assisted + Protection-assisted · M8 (4,667 damage; 0 life losses; Stage1 heal=none segment clear) | [dry manifest](../tmnt_iv/recordings/tmnt_iv_full_hard_dry_run.json), [prior video](../tmnt_iv/recordings/tmnt_iv_full_hard_credits.mp4), [Stage1 probes](../tmnt_iv/recordings/stage1_clean_track/stage1_probes.json) |
| Great Waldo Search | Continuous power-on → five-scrolls ending | Bronze / Clean · M8 | [video](../great_waldo_search/recordings/great_waldo_search_full_credits.mp4) |
| Super Metroid | Continuous power-on → Spore Spawn (not ending) | Bronze / Resource-assisted · M5 | [manifest](../super_metroid/recordings/start_to_spore_spawn.json), [assist contract](../super_metroid/docs/ASSIST_CONTRACT.md) |

TMNT IV is the reference **linear combat** clear, not “rank 3.”

## Active near-term trio

1. **Final Fight** (M3) — generalize the TMNT combat stack toward a continuous clear
2. **Magical Quest** (M2) — first segment of the platformer trunk
3. **Super Metroid** (M5) — extend the verified continuous suffix toward ending

Battle Clash remains `blocked: infrastructure` (no Super Scope injection).

## Bottlenecks

| Track | Current gate | Blocker |
|-------|--------------|---------|
| Linear combat | Final Fight M3→M4 | Natural-entry and Stage 3 continuity |
| Platforming | Magical Quest / Joe & Mac M2→M3 | First room/segment clears |
| Graph navigation | Super Metroid M5→M6 | Post–Spore Spawn route ownership |
| Continuous control | F-Zero / Pilotwings M2→M3 | First lap / lesson objective |
| Planning | Harvest M2 | Long-horizon evaluation contract |

## Capability phase focus

| Phase | Status |
|-------|--------|
| 0 Harness validation | Great Waldo verified; fighters supply match fixtures |
| 1 Linear full-game clears | TMNT done; Final Fight / SDD / Rival Turf in flight |
| 2 Continuous control | Boot/instrumentation only; Battle Clash blocked |
| 3 Platforming | Boot checkpoints; SMW/DKC tooling present |
| 4 Graph exploration | Super Metroid leading; `alttp/` active at title→castle grounds |
| 5–7 Campaigns / planning / procedural | Harvest is the Phase 6 foothold; later research |

## Directory name authority

| Use | Do not use |
|-----|------------|
| `super_metroid/` | `super_metroid_rl/` |
| `SMW/` | `super_mario_bros/` |
| `harvest/` | — |
| `alttp/` | — |

## Next documentation / tooling checks

```bash
uv run python docs/generate_game_matrix.py
uv run pytest tests/test_docs.py -q
```
