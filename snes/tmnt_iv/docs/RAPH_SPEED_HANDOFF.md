# Raphael hard speed handoff (observer session)

Date: 2026-08-26. Character: Raphael (char 8). Difficulty: Hard.
Goal: **faster continuous hard run + less damage**, toward Clean (pizza-only).

This is a session map, not STATUS. Do not promote probe wins into
`STATUS.md` / `BASELINE_METRICS.md` without a full dry-run.

## Current verified baseline (do not clobber)

Assisted continuous Raphael hard credits:

| Metric | Value | Source |
|--------|-------|--------|
| Power-on → credits | **00:57:19.635** | `recordings/tmnt_iv_full_hard_dry_run.json` |
| Frames | 206,718 | same |
| Damage taken | 4,667 | same |
| Emergency heals (HP≤16→80) | 65 | same |
| Form-2 iframe frames | 4,635 | same |
| Life losses | 0 | same |

Human Any% 1P1C Hard (Raphael) is ~**19:33** (SRC; darkalexandr, 6th).
The bot is ~3× slower. TAS/human speedruns skip screens and use
Raph's **jump-kick / dash-ram**, not grounded Y poke.

Clean (zero HP writes, zero iframe writes, no A-special) is a parallel
track (`rr-t4cl`). Faster play that also takes less damage is the
shared path: fewer emergency heals fall out of lower damage.

## Damage buckets (ROI order)

| Rank | Stage | Damage | Share | Probe state |
|------|-------|--------|-------|-------------|
| 1 | Technodrome (byte 3) | 1,022 | 21.9% | `RaphFullHardStage4` / `RaphFullHardTank` |
| 2 | Prehistoric / Slash | 861 | 18.4% | `RaphFullHardBoss5` (11,386f / 478 / 6) |
| 3 | Starbase | 749 | 16.0% | `RaphFullHardBoss9` |
| 4 | Wounded Knee | 579 | 12.4% | `RaphDiagBoss7` / `RaphFastStage7` |

Slash probe KEEP `spin_dodge_adx=40` is **6,765f / 226 / 3** on
`RaphFullHardBoss5` but **regressed whole-run damage** (5,474 vs 4,667).
Do not port spin-40 without a full-route re-tune.

## This sitting (parent-verified probes)

| Bead | Pin | Before | After | Ship? |
|------|-----|--------|-------|-------|
| `rr-iprz.1` Slash jump-over | `RaphFullHardBoss5` | 11,386f / 478 / 6 | **9,595f / 435 / 6** | YES (policy). Dry-run before STATUS — Slash RNG. |
| `rr-iprz.2` tank jump-behind | `RaphFullHardTank` `--stop-stage-gt 3` | timeout 20k / 478 / 7 | **12,848f / 236 / 3** CLEAR | YES. Stage4 agent 31,197→29,919 / 1,117→1,087. |
| `rr-iprz.3` form-2 offset | `Boss9_phase2` (Leo) | life_loss 485f | **9,420f / 152 / 3** CLEAR | YES play. No Raph form-2 pin; iframe assist still on. |
| `rr-iprz.4` raph_air | `RaphFastStage7` 8k | poke **214 / 3** | dash 277 / 4 (reached boss); jump-only 456 / 7 | Dash stays off. Starbase period-4 jump **is** hooked. |

`policy.py` is ~1,360 LOC after CombatProfile + Baxter extract.
Spin still 52. No A-special. No STATUS / BASELINE edit.

**Next (2026-08-27): `rr-iprz.5` still Starbase stall, not Slash.**
See `docs/tasks/rr-iprz.5-residual.md`. Shipped right-rail skip
(`starbase_rail_right` at x≥220): Diag **33,825→24,645f** (time KEEP,
dmg 1,004→1,076). Fast **23,272f / 917** vs 23,072 / 863. Boss9
**6,540f / 64** vs 6,300 / 144. Recover Fast+Boss9 without restoring
the Diag rail loop. Do not skip dumpster on x=126 / 207; do not
tighten close_gap below `%4`.

## Continuous dry-run 2026-08-27 (do not STATUS)

`recordings/tmnt_iv_full_hard_dry_run_rr_iprz.json` — did **not** overwrite
the on-disk `tmnt_iv_full_hard_dry_run.json` or STATUS 00:57:19.

| | Published 57:19 | On-disk 08-02 file | This sitting |
|--|-----------------|-------------------|--------------|
| Time | **00:57:19.635** | 00:58:15.610 | **01:03:35.001** |
| Damage | **4,667** | 5,801 | 5,721 |
| E-heals | **65** | 81 | 83 |
| Iframe | 4,635 | 5,040 | 7,494 |
| Lives lost | 0 | 0 | **0** |

Probe KEEPs ran (slash_jump_over 2,197f, blocker_jump_behind 38f,
shredder_offset 596f) but later-stage RNG ate the time. First attempt
soft-locked Alleycat mashing Y on an uncollectable box at (70,118);
`PizzaSeek` now gives up after 48f with no HP restore.

Do not promote. Slash follow-up (four isolated algorithms + KEEP
trace + three parent patches) **REJECT** vs 9,595f / 435 — do not
reopen. Next knob: **Starbase stall** (`rr-iprz.5`) — recover Fast
23,272 / 917 and Boss9 6,540 / 64 without giving Diag 24,645 back
to the rail loop. Do not skip dumpster on x=126 / 207; do not
tighten close_gap below `%4`. Technodrome tank in
continuous context is second (prehistoric entry 23:20 vs 22:23
published).

`policy.py` is **2026 LOC** (soft max ~1000). Extract tactics into
`tmnt_iv/tactics/` rather than growing the god file.

## Hard constraints (every agent)

- Raphael char 8. Prefer `RaphFullHard*` / `RaphFast*` / `RaphDiag*`.
- **Never A-special** (HP drain). Dash+Y (shoulder ram) and B+Y (jump
  kick) are allowed — they are not the Power Attack.
- Do not reopen: Stage 1 hazard jump-dodge, global pizza seek, sewer
  dumpster thrash, Slash spin=40 as production.
- Do not overwrite assisted `tmnt_iv_full_hard_*` stems.
- Checkpoint wins need a second entry (natural / continuous-faithful).
- Report **frames and seconds and damage** for every probe.
- `uv run pytest snes/tmnt_iv/tests -q` after policy edits.
- Live probes: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy`.

## Probe commands

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_boss_metrics RaphFullHardBoss5 \
  --max-frames 40000 --stop-stage-gt 4 --heal emergency

SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_boss_metrics RaphFullHardStage4 \
  --max-frames 40000 --stop-stage-gt 3 --heal emergency

SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_boss_metrics RaphFullHardTank \
  --max-frames 20000 --heal emergency

SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_boss_metrics RaphFullHardBoss9 \
  --max-frames 20000 --heal emergency
```

Wiki notes: `docs/SPEEDRUN_STRATEGIES.md`.
Play anti-patterns: `docs/CLEAN_PLAYBOOK.md`.
Tracker: `bd ready -l tmnt_iv`.
