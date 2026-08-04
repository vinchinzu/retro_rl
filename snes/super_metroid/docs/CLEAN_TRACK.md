# Clean track — non-assist continuous tips

Parallel workstream for **Bronze / Clean** continuous tips: **no energy
refill, no ammo refill**, zero resource writes. Orthogonal to the primary
**Bronze / Resource-assisted** KPDR spine (current tip: Bat Cave).

Benchmark labels: [BENCHMARK_SPEC.md](../../../docs/BENCHMARK_SPEC.md).
Assisted contract (primary path): [ASSIST_CONTRACT.md](ASSIST_CONTRACT.md).
Milestone board: [routes/MILESTONES.md](routes/MILESTONES.md) · Clean section.
Backlog epic: `CLEAN` in [routes/BACKLOG.csv](routes/BACKLOG.csv).

## Why this exists now

The assisted spine is past mid-game (Bat Cave / K4.4 primary; Frog Save side
tip). Early-game controllers (Morph → two-Missile detour → Bombs → **Bomb
Torizo** → Parlor) are mature and hash-pinned. Clean is a **privilege-reduction**
lane on that already-green prefix — not a second full-route rewrite.

| Fact | Assisted (primary) | Clean (this track) |
|------|--------------------|--------------------|
| Intervention | Resource-assisted (energy + ammo) | **Clean** (no resource writes) |
| Program tip | Bat Cave (`--to bat_cave`) | ★ Target: Bomb Torizo exit (`--to bombs`) |
| Maturity gate | M5 → M8 assisted full clear | Parallel; does **not** move M5/M8 |
| STATUS primary | Assisted only | Secondary section when green |

## Hard rules (do not destroy the bronze assist path)

1. **Defaults stay assisted.** `scripts/record/continuous.py` keeps energy +
   ammo assists **on** unless flags disable them. Never invert the default.
2. **Separate artifacts.** Clean reports/videos/checkpoints use a `_clean`
   stem (e.g. `bombs_clean.json`). Never overwrite
   `<tip_id>.json` / `.mp4` that are assisted baselines.
3. **STATUS primary tip stays assisted.** Clean greens go under a **Clean
   track** section. Do not re-label Bat Cave / Frog Save / Varia / Business as Clean.
4. **Shared controllers.** Prefer the same `play_*` / policy segments. Assist
   is applied only in the session layer (`assist.py` / `run_*`). Fork a
   controller only when ammo/health economy forces a one-knob fix, and keep
   the assisted path calling the same code with assist enabled.
5. **Clean failure ≠ assisted demotion.** A RED clean bombs run never unmarks
   assisted continuous greens or rolls back the Bat Cave tip.
6. **No progression privilege.** Clean still forbids item/capacity/boss/door
   writes — same forbidden list as [ASSIST_CONTRACT.md](ASSIST_CONTRACT.md).
7. **Serialize STATUS / continuous defaults.** Clean infra may touch
   `continuous.py` report-path helpers and CLI flags, but must not change
   default tip, default assist, or assisted artifact stems without a planner
   card.

## What “Clean” means here

| Write | Allowed? |
|-------|----------|
| Controller buttons | Yes |
| Read-only RAM (Bronze observation) | Yes |
| Energy / health restore | **No** |
| Ammo restore (missiles / supers / PBs) | **No** |
| Capacity / items / boss / doors / pose | **No** (same as assisted) |
| Mid-run state loads | **No** |

Integrity extras for a successful clean claim:

- `assist.energy.writes == 0` and `assist.energy.restored == 0`
- all ammo counters `writes == 0` / `restored == 0`
- `progression_writes == 0`, `capacity_writes == 0`, `state_loads == 0`
- natural inventory unlocks only
- intervention class in report / STATUS: **Clean**

Deaths: start with **deaths_zero required** for continuous clean claims (same
as assisted tips). If clean death-recovery becomes necessary later, open a
dedicated card — do not silently allow continues.

## Early-game note (Morph / Bombs)

Today:

- Tips `morph` and `bombs` use **ammo-only** assist (`UnlimitedAmmoAssist`).
  Energy refill is wired from **spore+** (`supports_unlimited_energy`).
- Assisted bombs baseline already recorded **Missile refill writes** after
  natural unlock (see `START_TO_BOMBS.md`). Clean bombs is primarily
  **`--no-unlimited-ammo`** (and energy flag when present).

So Clean → Bomb Torizo is mostly: same route, zero ammo refill, survive BT
and early skirmishes on natural packs + collected capacity (10 missiles).

## ★ Tip and ladder

| Order | Milestone | CLI | Artifact stem (clean) | Status |
|------:|-----------|-----|------------------------|--------|
| 1 | Power-on → Morph (clean) | `--to morph --clean` | `morph_clean` | **GREEN** 27,074f |
| 2 | Power-on → **Bomb Torizo exit (clean)** | `--to bombs --clean` | `bombs_clean` | ★ product tip (missiles detour green; BT = existing model) |
| 3 | → Spore exit (clean) | `--to spore --clean` | `spore_clean` | parked until BT green |
| … | Later prefixes only after prior clean green | … | `*_clean` | parked |

**Prefix progress (2026-08-02):** clean morph integrity green. Clean bombs path
collects both early Missile packs at the assisted frames (27,928 / 29,690) with
zero resource writes. Pit settle tolerates beam-selected clean detour exit.
Bomb Torizo is **not** re-solved on this track — use the existing assisted BT
policy / combat model; Clean only requires no resource writes and no death.
Clean bombs residual log removed; track progress lives in this file.

`--clean` is landed (disables energy + ammo, defaults to `*_clean` stems,
requires zero resource writes). Prefer:

```bash
uv run python snes/super_metroid/scripts/record/continuous.py \
  --to bombs --clean --no-video
# → recordings/bombs_clean.json by default
```

Equivalent long form: `--no-unlimited-energy --no-unlimited-ammo` (also
routes to `*_clean` stems so assisted baselines stay safe).

## Process (same recipe, dual-track integrity)

1. **Infra first:** artifact isolation + clean integrity + CLI — **done**
   (`SM-CLEAN-ARTIFACTS` / `CLI` / `INTEGRITY`).
2. **Probe:** clean morph (`SM-CLEAN-MORPH`) — **GREEN**; missiles detour on
   clean bombs path logged; then clean bombs tip compose (existing BT model).
3. **Economy fix (only if RED):** one-knob controller/BT ammo-aware change;
   re-verify **assisted** bombs + prefix continuous after any shared controller
   edit (stabilize wave).
4. **Compose / dual re-verify** clean tip ×2 integrity green.
5. **STATUS secondary** section only — planner apply.

Pure-first still applies for *new* geometry. Clean tip work on already-green
assisted prefixes is primarily **compose + integrity**, not a full hop stack.

## Parallelism vs assisted spine

| Track | Blocks spine? | Notes |
|-------|---------------|-------|
| **B spine** (assisted Bat Cave → Speed Hall…) | N/A (primary) | P0 product |
| **CLEAN** | No | Parallel; serialize only if editing shared `play_*` geometry |
| **C practice / ARCH / BOSS-INFRA** | No | Unchanged |

If a clean economy fix touches shared geometry, run assisted stabilize on the
affected prefix before claiming either track green.

## Tickets (living)

| Ticket | Role |
|--------|------|
| `SM-CLEAN-CONTRACT` | Done — docs contract (this file + ASSIST pointer) |
| `SM-CLEAN-ARTIFACTS` | Done — `_clean` paths; never overwrite assisted |
| `SM-CLEAN-CLI` | Done — `--clean` alias + path default |
| `SM-CLEAN-INTEGRITY` | Done — zero resource-write asserts |
| `SM-CLEAN-MORPH` | Done — clean morph continuous |
| [`SM-CLEAN-BOMBS`](tasks/SM-CLEAN-BOMBS.md) | ★ Clean → Bomb Torizo continuous tip |
| `SM-CLEAN-BT-ECONOMY` | Optional; only if clean BT RED |
| `SM-CLEAN-STAB` / `SM-CLEAN-STATUS` | Stabilize + STATUS secondary (after tip green) |

Queue: [tasks/QUEUE.md](tasks/QUEUE.md).

## Non-goals (for now)

- Clean full-game clear / Clean Frog tip (park until early clean ladder holds)
- Changing program M8 target to Clean
- Observation migration Bronze → Silver (separate workstream)
- Disabling assists in tests that assert assist telemetry shape
