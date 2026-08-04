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

Never hand the executor open-ended “next Clean stage for the whole game” or an
**epic shell** (`T4-CLEAN-S2`, `T4-CLEAN-FULL`, …). Give **one thin child card**
from [CLEAN_LADDER.md](CLEAN_LADDER.md).

**Pizza-only Clean ≫ emergency-HP assist.** Full stage suite green is a late
rung. Partial REACH metrics (farther death, lower max_hit, lower damage) are
valid card GREEN without stage_advance.

## Non-negotiable gates

### 0. Thin Clean rungs (before multi-entry claims)

Stage Clean work uses fixed rungs — see [CLEAN_LADDER.md](CLEAN_LADDER.md):

`PROBE → BOSS → LATE → REACH → CKPT → BRIDGE → SUITE → STAB`

| Rung | GREEN means |
|------|-------------|
| PROBE | Residual table matches suite JSON (no policy) |
| BOSS / LATE | That entry pizza-only stage_advance |
| REACH | Named metric improved on full checkpoint (may still life_loss) |
| CKPT | Full fight-ready checkpoint stage_advance |
| BRIDGE | Continuous-faithful entry stage_advance |
| SUITE | All required entries green (verify only) |
| STAB | Re-verify suite + assisted dry-run; no new knobs |

Executors must not STATUS-promote. Suite numbers only from artifact JSON.

### 1. Multi-entry first + stabilize waves

Any **production policy** change that claims **full stage Clean** must
suite-green from **checkpoint + continuous-faithful / power-on** before
continuous re-record. REACH/BOSS partial greens do not unlock continuous Clean.

Wave types:

| Wave type | Allowed work | Exit gate |
|-----------|--------------|-----------|
| **Implement / stress** | One thin rung: probe, one-knob, single-entry, REACH | Rung acceptance (or residual) |
| **Stabilize** | Re-verify affected stage suites + assisted dry-run only | Metrics pasted; no new knobs |

Rules:

1. After an implement wave that lands live policy knobs, run a **stabilize
   wave** before stacking more knobs.
2. Never land **two interacting combat knobs** in the same continuous without
   an intervening suite + dry-run gate.
3. Continuous re-record remains a **planner gate**. Executors may *propose*
   re-record commands; they do not claim STATUS.
4. One-knob discipline: each policy card changes **one** named primitive or
   constant group. Multi-file continuous wiring is a separate card.
5. Do not re-try residual **Rejected knobs** without planner override on the card.

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

## Near-term sequence (2026-08-02)

Assisted continuous is green. Process:

1. Clean **infra** — **done**.
2. Alleycat **thin rungs** (`T4-CLEAN-S2-*`): BOSS+LATE done → REACH/EDGE →
   CKPT → BRIDGE → SUITE → STAB ([CLEAN_LADDER.md](CLEAN_LADDER.md)).
3. Sewer **thin rungs** (`T4-CLEAN-S3-*`) in parallel only if `policy.py`
   serialize allows; else after S2 STAB.
4. S4–S8: INFRA-PROBE then same rungs; never one-card suite green.
5. S9: WAVE then **F2 without iframe** (hard gate).
6. ★ `T4-CLEAN-FULL-ATTEMPT` (measure) → fixes on failing stage rung →
   STAB → STATUS secondary (planner).
7. Parallel **A assisted** PROBE/KNOB polish (not fat “cut stage damage” alone).

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
