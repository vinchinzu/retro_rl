# Super Metroid sub-agent process

Planner–executor loop for KPDR continuous integrity. Cards live in
[`docs/tasks/`](./); queue board is [`QUEUE.md`](QUEUE.md); template is
[`docs/TASK_TEMPLATE.md`](../TASK_TEMPLATE.md).

This document hardens rules that Waves 3–5 already stress-tested. It does
**not** replace natural-entry judgment or continuous integrity ownership —
those stay with the planner.

## Roles (unchanged)

| Role | Who | Owns |
|------|-----|------|
| Planner | Grok / human | Continuous integrity, STATUS, natural-entry design, tip order, promote/revert |
| Executor Flash | OpenCode + Flash | Tracker/docs/dwell reports, STATUS **proposals**, source-state index edits |
| Executor Luna | OpenCode + Luna | Controllers, tests, geometry pure probes, tip wiring skeletons |

Never hand the executor open-ended “next continuous tip” work.

## Non-negotiable gates

### 1. Pure-first + alternate stabilize waves

Any **spine controller** change must go **pure-green** from a continuous-like
source state **before** continuous re-record.

Wave types:

| Wave type | Allowed work | Exit gate |
|-----------|--------------|-----------|
| **Implement / stress** | One-knob pure probes, scaffolds, diagnostics, unit locks | Pure green (or explicit blocked residual) |
| **Stabilize** | Pure re-verify of affected tips + continuous re-record only | Continuous green on affected tip(s); no new knobs |

Rules:

1. After an implement/stress wave that lands live spine knobs, run a
   **stabilize wave** before stacking more knobs.
2. Never land **two interacting spine knobs** in the same continuous without
   an intervening pure + continuous gate.
3. Continuous re-record remains a **planner gate** (expensive, high stakes).
   Executors may *propose* re-record commands; they do not claim STATUS.
4. One-knob discipline: each geometry card changes **one** named primitive
   (or one named constant group). Multi-file tip wiring (recipe steps 2–5)
   is a separate card after pure green.

### 2. Primitive extraction after residuals

Every successful settle, short-hop, lip-approach, door-shot, or climb sequence
that survives **pure + continuous** should be promoted to
`routes/controller_common.py` (or local `routes/kpdr/primitives.py` once a
second in-package consumer exists) **with unit tests**.

Future cards say “use `settle_platform(12)` + `short_hop_left(24)`” instead of
re-inventing timing. This is the main lever for raising Luna/Flash geometry
success rate.

Promotion criteria:

- Pure green from continuous-like source
- Continuous green on the tip that owns the sequence (or multi-run stable
  pure for reverse hops not yet continuous)
- Named function + reason strings + tests in `tests/test_controller_common.py`

### 3. Source-state catalog

Index of continuous-like pure entry states:
[`docs/SOURCE_STATES.md`](../SOURCE_STATES.md).

Dispatch (or a Flash card) should auto-suggest `--source` from room id +
capability set. “Blocked on source” residuals must propose a **capture** card
(`SM-*-SRC`) rather than a free geometry poke.

### 4. STATUS / tracker / board sync

Recurring Flash card [`SM-ROLLUP-STATUS`](SM-ROLLUP-STATUS.md):

- Reads latest continuous JSON reports + QUEUE residuals
- Proposes diffs for `STATUS.md`, `KPDR_TRACKER.csv`, `PATH_ROOM_BOARD.md`
- Planner **only approves** — executor never STATUS-promotes alone

### 5. Dispatch + residual loop

- File ownership / conflict detection in `scripts/dispatch_opencode.sh`
  (parallel cards must not share hot modules).
- Serialization hotspots (never parallel-edit):
  - `routes/kpdr/business_climb.py`
  - `routes/kpdr/hijump_return.py` (and HJ gray-door knobs)
  - `routes/spore_spawn_controller.py` / spore fight dwell knobs
  - geometry on `routes/kpdr/varia_return.py` (door return)
  - `routes/continuous.py`, `docs/STATUS.md` (planner only)
- Every residual ends with **one** proposed next card ID + **one** change
  (see residual schema below).

### 6. Tip wiring (recipe steps 2–5)

Pure controller (step 1) and continuous record + STATUS (steps 6–7) stay
separated. Steps 2–5 (graph edge, catalog tip, RouteHop rows, `run_to`
registration) are multi-file checklist work — one card after pure green, or a
declarative hop skeleton generator later. Integrity judgment stays planner.

### 7. Metrics + dual-track clarity

Track lightly in QUEUE (update each stabilize wave):

| Metric | Definition |
|--------|------------|
| Pure-green rate | pure-green cards / pure geometry cards in wave |
| Continuous regression rate | continuous RED after wave / continuous attempts |
| Top dwell rooms | from `split_dwell.py` on latest green report |
| Cost per wave | session count + planner re-record count |

**Dual track:** `ROOM_WORK_QUEUE` practice rooms never pollute the KPDR
continuous spine integrity story. Practice greens are not continuous evidence.

## Residual schema (required)

Executor final message **and** optional `docs/tasks/<ID>-residual.md` /
`<ID>-note.md` must include every field:

```markdown
## Residual — <card-id>

### Result
GREEN | RED | BLOCKED | PARTIAL

### Files changed
- path — one-line purpose

### Verify paste
(command + exit code + relevant stdout; repo-relative paths only)

### Acceptance
- [x]/[ ] each card checkbox, pass/fail

### Residual risks
What still blocks pure-green / continuous / STATUS (bullet list).

### Next action (required)
- **Next card ID:** SM-XXXX   # or PLANNER-GATE / none
- **One change:** single knob or single decision (one sentence)
- **Source state:** path or “needs capture: SM-*-SRC”

### Non-claims
- Did not STATUS-promote
- Did not forge progression/capacity/door/event/boss RAM
- Not continuous evidence (unless card was continuous — then planner only)

### Probe pin (if pure/geometry) — **mandatory metrics**
room=0x…. pose=… x=… y=… door_transition=…
frames=…   # pure probe total if available
dwell=…    # optional; room dwell if timed
last_pin=… # room/pose/x/y at fail or success
```

On **RED** pure: attach or path-link last pin + short video clip when tooling
supports it (planned: auto-capture). Do not “debug dark” — residual must name
the one next knob and source state.

**Force-pass ban:** pure geometry and continuous integrity are never green from
scaffolds, diagnostics, or unit tests alone.

## Dual-track room farm (continuous tip relaxed)

When the continuous tip is blocked or parked, run **segment-only** practice
farm waves instead of serial tip work:

1. One agent ↔ one room problem (`SM-ROOM-SEG-NN`); own files = that problem's
   policy + state only.
2. Parallel width 8 (or less); never mix spine knobs into the same batch.
3. Between rounds: wait EXIT → residual rollup → path-guard continuous /
   STATUS / `routes/kpdr/*` / `progression.py` → **fresh sessions** (no
   history carry) → generate next cards.
4. Model: OpenCode Luna (`openrouter/openai/gpt-5.6-luna`) with
   `--variant max` (max thinking).
5. Practice promote is dual-track only — never STATUS or continuous evidence.

```bash
./super_metroid/scripts/farm_room_waves.sh --rounds 10 --parallel 8
./super_metroid/scripts/farm_room_waves.sh --rounds 20 --parallel 8 --deadline-hours 2
```

## Near-term sequence (post-Varia tip, 2026-07-31)

Verified continuous tip is power-on → Varia (`--to varia`, ~101,954f). Process
discipline stays fixed; composition beyond Varia is planner-gated.

1. Finish pure reverse-spine residuals (climb/door geometry, Zeela upper,
   warehouse reverse stack) under one-knob + continuous-like sources.
2. Stabilize continuous re-record `--to kraid` / `--to varia` after each
   implement wave; promote frame totals only when multi-run ≤ baseline or
   deliberately documented (Wave-6 re-verify was slower — do not auto-promote).
3. Continue primitive extraction from green tightens into `controller_common`.
4. Flash: `SM-ROLLUP-STATUS` proposals; keep `SOURCE_STATES.md` current.
5. Only after reverse pure green to Business: open post-Varia continuous tip
   (planner compose + integrity).
6. **Parallel dual-track:** room-segment farm (`farm_room_waves.sh`) and combat
   unit scaffolds on non-interacting files while spine advances.
7. **Structure plan (planner-owned code work):** selective-RAM enforcement,
   tip scaffold script, pure RED diagnostics, graph hop ranking — see
   [`../plan.md`](../plan.md) and [`../ARCHITECTURE.md`](../ARCHITECTURE.md).
8. **Parallel Clean track:** no energy/ammo continuous tips (Bomb Torizo first).
   Infra before tip (`SM-CLEAN-ARTIFACTS` / `CLI` / `INTEGRITY`); artifacts use
   `*_clean` stems only. Contract: [`../CLEAN_TRACK.md`](../CLEAN_TRACK.md).
   Shared geometry changes re-verify **assisted** prefix + clean tip.

## Process tooling improvements (do not relax gates)

Targeted upgrades to reduce dark poking further as scale grows. Pure-first /
one-knob / residual rules remain mandatory.

| Improvement | Owner | Intent |
|-------------|-------|--------|
| Pre-dispatch schema validation | Dispatch / Flash | Reject cards missing recipe step, own-files, source path (pure), acceptance |
| Auto-skeleton residual.md | Dispatch / Luna | Always leave PROCESS residual file shape even on early abort |
| Mandatory residual metrics | Luna / Flash | Frames, dwell if known, exact pose/x/y/`door_transition` pin on pure |
| Auto-suggest source state | Dispatch + SOURCE_STATES | From room id + required capabilities |
| Ownership / file-locking declarations | Dispatch | Parallel waves stay safe (extend conflict check) |
| RED pure richer diagnostics | Luna tooling | Replay clip + PLM/door RAM snapshot + last pin |
| Serialize hot modules | Dispatch (existing) | `business_climb`, `varia_return`, spore, continuous, STATUS |

**Scale pattern:** dual-track room-segment farming
(`farm_room_waves.sh`, `generate_room_segment_cards.py`) while the continuous
spine advances. Luna productively clears non-interacting rooms/combat units;
planner owns continuous composition and integrity.

## Wave bookkeeping

When opening a wave, label it in QUEUE:

```markdown
## Wave N — implement|stabilize (YYYY-MM-DD)
Intent: …
Serialize: …
Exit gate: pure … / continuous --to …
```

Close the wave with honest rollup (GREEN/RED, races, reverts) before the next
implement wave starts.
