# TMNT IV sub-agent process

Planner–executor loop for continuous integrity and Clean rollout. Cards live
in [`docs/tasks/`](./); queue board is [`QUEUE.md`](QUEUE.md); template is
[`docs/TASK_TEMPLATE.md`](../TASK_TEMPLATE.md).

Learnings imported from Super Metroid (`super_metroid/docs/tasks/PROCESS.md`)
and adapted to linear-combat stage suites.

## Roles

| Role | Who | Owns |
|------|-----|------|
| Planner | Grok / human | Continuous integrity, STATUS, natural-entry design, stage order, promote/revert |
| Executor | Bounded agent | Probes, one-knob policy, tests, docs **proposals** |

Never hand the executor open-ended “next Clean stage for the whole game.”

## Non-negotiable gates

### 1. Multi-entry first + stabilize waves

Any **production policy** change that claims Clean must suite-green from
**checkpoint + continuous-faithful / power-on** before continuous re-record.

Wave types:

| Wave type | Allowed work | Exit gate |
|-----------|--------------|-----------|
| **Implement / stress** | One-knob probes, stage suites, CLI/infra, diagnostics | Suite green (or explicit blocked residual) |
| **Stabilize** | Re-verify affected stage suites + assisted dry-run only | Suites green; no new knobs |

Rules:

1. After an implement wave that lands live policy knobs, run a **stabilize
   wave** before stacking more knobs.
2. Never land **two interacting combat knobs** in the same continuous without
   an intervening suite + dry-run gate.
3. Continuous re-record remains a **planner gate**. Executors may *propose*
   re-record commands; they do not claim STATUS.
4. One-knob discipline: each policy card changes **one** named primitive or
   constant group. Multi-file continuous wiring is a separate card.

### 2. Dual track (assisted polish vs Clean)

| Track | Integrity story |
|-------|-----------------|
| **A assisted** | M8 continuous hard clear; cut damage / heals / iframe while keeping 0 lives lost |
| **CLEAN** | Stage pizza-only suites → whole-run Clean continuous |

Practice / local grind / slash lab are dual-track only — not continuous Clean
evidence. KEEP proposals from grind **never** auto-edit `policy.py`.

### 3. Artifact + assist isolation

- Clean continuous defaults to `*_clean` stems
  ([CLEAN_TRACK.md](../CLEAN_TRACK.md)).
- Defaults remain emergency HP + form-2 iframe on.
- Clean RED never demotes assisted BASELINE_METRICS or STATUS primary gate.

### 4. STATUS / baseline sync

After continuous green:

- Assisted improve → update `BASELINE_METRICS.md` + STATUS primary metrics
  (planner).
- Clean tip → STATUS **secondary** section only (`T4-CLEAN-STATUS`).

### 5. Residual loop

Every residual ends with **one** proposed next card ID + **one** change
(see residual schema in [TASK_TEMPLATE.md](../TASK_TEMPLATE.md)).

Serialization hotspots (never parallel-edit):

- `policy.py` (shared combat knobs)
- `scripts/record_full_hard_run.py` (assist defaults, paths)
- `docs/STATUS.md`, `docs/BASELINE_METRICS.md` (planner only)

### 6. Checkpoint ≠ continuous (TMNT pure-first analog)

Super Metroid pure-first maps here to:

1. Fight-ready checkpoint probe (`heal=none` when Clean)
2. Continuous-faithful or power-on entry
3. Only then continuous dry-run claim

Playbook anti-patterns stay hard bans
([CLEAN_PLAYBOOK.md](../CLEAN_PLAYBOOK.md)).

### 7. Metrics (light bookkeeping in QUEUE)

| Metric | Definition |
|--------|------------|
| Suite-green rate | suite-green stage cards / stage Clean cards in wave |
| Continuous regression rate | assisted dry-run worse after wave / dry-run attempts |
| Top damage stages | from latest assisted dry-run stage table |
| Heal / iframe counts | e-heals + form-2 frames on latest dry-run |

## Near-term sequence (2026-08-01)

Assisted continuous is green. Process:

1. Land Clean **infra** (`T4-CLEAN-ARTIFACTS` / `CLI` / `INTEGRITY` / `CONTRACT`).
2. Finish Alleycat + Sewer Clean suites (`T4-CLEAN-S2`, `T4-CLEAN-S3`).
3. Roll stages 4–9 Clean one suite at a time; form-2 play solution is the hard
   Clean gate (`T4-CLEAN-S9`).
4. ★ `T4-CLEAN-FULL` continuous Clean dry-run → stab → STATUS secondary.
5. Parallel **A assisted** polish on Technodrome / Prehistoric / Starbase /
   Wounded Knee damage buckets without blocking Clean infra.

## Process tooling improvements (do not relax gates)

| Improvement | Intent |
|-------------|--------|
| `--clean` on continuous recorder | Disable both assists; default `*_clean` stems |
| Clean integrity assert helper | Fail run if e-heals or iframe frames > 0 when clean |
| Residual skeleton on abort | Always leave PROCESS residual shape |
| Stage suite generator | Copy `probe_stage1_clean` pattern for stages 4–9 |

## Wave bookkeeping

When opening a wave, label it in QUEUE:

```markdown
## Wave N — implement|stabilize (YYYY-MM-DD)
Intent: …
Serialize: …
Exit gate: suite … / continuous --clean …
```

Close the wave with honest rollup before the next implement wave starts.
