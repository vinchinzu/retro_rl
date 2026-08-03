# Clean track — zero-assist continuous hard clear

Parallel workstream for **Bronze / Clean** continuous hard clear: **no
emergency HP writes, no form-2 iframe hold**. Orthogonal to the primary
**Bronze / Resource-assisted + Protection-assisted** full hard clear (already
M8 green).

Benchmark labels: [BENCHMARK_SPEC.md](../../docs/BENCHMARK_SPEC.md).  
Assisted contract (primary path): [ASSIST_CONTRACT.md](ASSIST_CONTRACT.md).  
Play lessons (do not relearn): [CLEAN_PLAYBOOK.md](CLEAN_PLAYBOOK.md).  
Tickets: [tasks/QUEUE.md](tasks/QUEUE.md).

## Why this exists now

The assisted continuous hard clear is **done** (power-on → hard credits,
~00:57:19 / 4,667 dmg / 65 e-heals / 4,635 iframe frames / 0 lives lost). Clean
is a **privilege-reduction** ladder on that already-green route — stage-by-stage
pizza-only survival — not a second full-route rewrite.

| Fact | Assisted (primary) | Clean (this track) |
|------|--------------------|--------------------|
| Intervention | Resource + Protection (e-HP + form-2 iframe) | **Clean** (zero resource/protection writes) |
| Program tip | Full hard credits dry-run | ★ Target: same clear, both assists = 0 |
| Maturity gate | M8 achieved | Parallel; does **not** open M9 |
| STATUS primary | Assisted only | Secondary section when green |

## Hard rules (do not destroy the assisted path)

1. **Defaults stay assisted.** `scripts/record_full_hard_run.py` keeps emergency
   HP + form-2 iframe **on** unless flags disable them. Never invert the default.
2. **Separate artifacts.** Clean reports/videos use a `_clean` stem (e.g.
   `tmnt_iv_full_hard_clean.json`). Never overwrite
   `tmnt_iv_full_hard_credits.*` / `tmnt_iv_full_hard_dry_run.json` assisted
   baselines.
3. **STATUS primary tip stays assisted.** Clean greens go under a **Clean
   track** section. Do not re-label the 00:57:19 assisted dry-run as Clean.
4. **Shared policy.** Prefer the same `Stage1Policy` / stage knobs. Assist is
   applied only in the continuous recorder session layer. Fork policy only when
   Clean survival forces a one-knob fix, then re-verify **assisted** dry-run
   after any shared edit (stabilize wave).
5. **Clean failure ≠ assisted demotion.** A RED clean Alleycat suite never
   unmarks the assisted continuous clear or rolls back BASELINE_METRICS.
6. **No progression privilege.** Clean still forbids stage/lives/boss/event
   writes and mid-run state loads — same forbidden list as
   [ASSIST_CONTRACT.md](ASSIST_CONTRACT.md).
7. **Serialize STATUS / continuous defaults.** Clean infra may touch recorder
   path helpers and CLI flags, but must not change default assist, default
   output stems, or assisted baselines without a planner card.

## What “Clean” means here

| Write | Allowed? |
|-------|----------|
| Controller buttons | Yes |
| Read-only RAM (Bronze observation) | Yes |
| Natural pizza pickup (`char 0x30`) | Yes (not an assist) |
| Emergency HP restore | **No** |
| Form-2 iframe hold | **No** |
| Lives grants / stage / boss / event | **No** |
| Mid-run state loads | **No** |
| A-special (HP drain) | **No** (policy ban) |

Integrity extras for a successful clean claim:

- `emergency_hp` interventions == **0**
- form-2 iframe guard frames == **0**
- `life_losses == 0`
- `state_loads == 0`, progression/stage writes == 0
- intervention class in report / STATUS: **Clean**

Deaths / life losses: **0 required** for continuous clean claims (same as
assisted continuous). Continue/retry mid-run is not a Clean continuous claim.

## ★ Tip and ladder

Stage-by-stage pizza-only proofs first; whole-run only after all stage suites
are green (or after form-2 play solution lands).

| Order | Milestone | Tooling | Artifact stem (clean) | Status |
|------:|-----------|---------|------------------------|--------|
| − | Infra (paths / `--clean` / integrity) | `tests/test_clean_track.py` | `tmnt_iv_full_hard_clean*` | **done** |
| 0 | Stage 1 Big Apple Clean suite | `probe_stage1_clean --suite` | `stage1_clean_track/` | **done** |
| 1 | Stage 2 Alleycat **thin rungs** | `probe_stage2_clean`; [CLEAN_LADDER](tasks/CLEAN_LADDER.md) | `stage2_clean_track/` | BOSS+LATE done; suite **2/4** |
| 2 | Stage 3 Sewer **thin rungs** | `probe_stage3_clean` + LiveHard | `stage3_clean_track/` | PROBE ready |
| 3 | Stage 4 Technodrome Clean rungs | INFRA-PROBE then rungs | `stage4_clean_track/` | open |
| 4 | Stage 5 Prehistoric / Slash rungs | stage probe | `stage5_clean_track/` | open |
| 5 | Stage 6 Skull rungs | **no global pizza seek** | `stage6_clean_track/` | open |
| 6 | Stage 7 Wounded Knee rungs | stage probe | `stage7_clean_track/` | open |
| 7 | Stage 8 Neon rungs | stage probe | `stage8_clean_track/` | open |
| 8 | Stage 9 WAVE + form-2 **F2** | form-2 **without** iframe write | `stage9_clean_track/` | open |
| 9 | ★ Power-on → hard credits **Clean** | `FULL-ATTEMPT` then STAB | `tmnt_iv_full_hard_clean` | ★ product tip |

**Executor rule:** pizza-only Clear is much harder than emergency-HP assist.
Assign **one rung card** (PROBE / REACH / EDGE / CKPT / …), never “green whole
stage suite” or the epic shell alone.

Preferred continuous (after infra cards land):

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.record_full_hard_run \
  --clean --dry-run
# → recordings/tmnt_iv_full_hard_clean.json by default
```

## Process (SM-inspired recipe)

1. **Infra first:** artifact isolation + clean integrity + CLI
   (`T4-CLEAN-ARTIFACTS` / `CLI` / `INTEGRITY` / `CONTRACT`).
2. **Stage suite:** multi-entry heal=none (checkpoint + continuous-faithful /
   power-on). See [CLEAN_PLAYBOOK.md](CLEAN_PLAYBOOK.md).
3. **One-knob fix if RED:** stage-local policy only; re-verify assisted dry-run
   after shared knobs (stabilize).
4. **Compose:** when stages 0–9 Clean are suite-green, continuous `--clean`.
5. **STATUS secondary** only — planner apply (`T4-CLEAN-STATUS`).

Pure-first analog for TMNT: **checkpoint-only wins are not Clean evidence**.
Always prove a natural/continuous-faithful or power-on entry (playbook rule 5).

## Parallelism vs assisted spine

| Track | Blocks spine? | Notes |
|-------|---------------|-------|
| **A assisted improve** | N/A (primary polish) | Cut damage/heals/iframe on already-green clear |
| **CLEAN** | No | Parallel; serialize only if editing shared policy knobs |
| **Grind / lab** | No | Dual-track; KEEP does not auto-edit production |

If a Clean economy fix touches shared policy, run assisted dry-run re-verify
before claiming either track green.

## Tickets (living)

| Ticket | Role |
|--------|------|
| [`T4-CLEAN-CONTRACT`](tasks/T4-CLEAN-CONTRACT.md) | Docs contract (this file + ASSIST pointer) |
| [`T4-CLEAN-ARTIFACTS`](tasks/T4-CLEAN-ARTIFACTS.md) | `_clean` paths; never overwrite assisted |
| [`T4-CLEAN-CLI`](tasks/T4-CLEAN-CLI.md) | `--clean` / `--no-emergency-hp` / `--no-iframe` |
| [`T4-CLEAN-INTEGRITY`](tasks/T4-CLEAN-INTEGRITY.md) | Zero e-heal + zero iframe asserts |
| [`T4-CLEAN-S2`](tasks/T4-CLEAN-S2.md) … [`T4-CLEAN-S9`](tasks/T4-CLEAN-S9.md) | Stage Clean suites |
| [`T4-CLEAN-FULL`](tasks/T4-CLEAN-FULL.md) | ★ Clean continuous hard credits |
| `T4-CLEAN-STAB` / `T4-CLEAN-STATUS` | Dual re-verify + STATUS secondary |

Queue: [tasks/QUEUE.md](tasks/QUEUE.md). Triage: [tasks/TRIAGE.md](tasks/TRIAGE.md).

## Non-goals (for now)

- Changing program M8 target label to Clean (Clean is publication class, not
  a new maturity gate)
- Observation migration Bronze → Silver (separate workstream)
- Re-opening Stage 1 hazard jump-dodge / global pizza seek (playbook bans)
- Disabling assists in tests that assert assist telemetry shape
