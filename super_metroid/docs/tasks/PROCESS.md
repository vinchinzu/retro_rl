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

## Tangible progress, anti-ceremony, and honest credit

The purpose of this project is **working, deployable route software** delivered
accretively in the shortest time compatible with correctness, performance,
reliability, and innovation. Process exists to serve that outcome; it must
**never become the product**.

These rules bind human-directed sessions and multi-agent farm/dispatch swarms
alike. Encode them into card acceptance criteria, not only in this doc.

- **No process porn.** Residuals, queue boards, dashboards, meta-reports, and
  process documents are **not** progress. A process artifact may exist only
  when it is a **hard gate** for a named feature or capability — pure-green
  from continuous-like source, dual continuous integrity, STATUS promote, and
  required residual schema qualify; self-referential paperwork does not.
  Choosing process work because it is easy and low-risk is **reward hacking**.
- **Feature-first ratio.** The overwhelming majority of open cards must deliver
  **runnable behavior** — controller/code, graph edges, pure probes, continuous
  tips — that an end user or consuming agent can actually exercise.
  Process/ops/docs-only items are capped (guideline: **at most ~5% of open
  ready cards**), and each must name the feature work it gates; a process card
  that gates nothing does not get created.
- **Honesty is absolute.** Never fake a test, present a fixture / door-warp /
  mock / retained capture as live pure or continuous proof, weaken an
  assertion to make it pass, hard-code a success path, or close work that is
  not done. A false close is reopened with an incident comment on the residual
  and queue.
- **Refusal is not delivery.** A correctly typed **BLOCKED** / residual refusal
  is far better than a fabricated green — and far less valuable than the real
  capability. Implementing only scaffolding, diagnostics, or “not yet” paths
  earns partial credit at most: it **never** closes a feature card. Full credit
  requires the positive capability implemented for real, pure- and/or
  continuous-verified as the card requires. Mark scaffold-only / refusal-only
  states explicitly (e.g. residual `PARTIAL` + next card ID) so they read as
  unfinished, never as shipped.

### Named reward-hacking patterns (all forbidden)

Beyond refusal-farming and process porn, call these out **by name** when
reviewing residuals or wave rollups — this architecture specifically invites
them:

1. **Gate self-weakening** — editing integrity / pure / continuous / STATUS /
   assist-contract / test gates so a failing check “passes.” Gate code and
   STATUS claims are planner-owned (or Flash **proposal** only); batch verify
   diffs them every stabilize wave.
2. **Proof-class inflation** — presenting fixtures, retained captures, door-warp
   topology, mocked endpoints, unit scaffolds, or hand-edited states as live
   pure/continuous proof. Live proof requires the card’s named source state (or
   power-on continuous), recorded selection/command, integrity/report artifacts
   chained to real route manifests, and fresh-process readback.
3. **Golden regeneration reflex** — regenerating expected baselines / goldens /
   tracker “done” marks to match broken output instead of fixing the controller
   or route. Golden or STATUS baseline changes require an explicit planner note
   (treat as `GOLDEN-CHANGE` / STATUS-promote only) and a semantic diff review.
4. **Commit-stream pumping** — trivial or artificially split commits, or
   placeholder scaffolds that pass unit check alone (`pass`, empty stubs,
   `NotImplemented` that still “compiles”). Placeholder macros and force-pass
   scaffolds are banned in committed green claims; every commit names its card
   ID and touched scope. Force-pass ban above still applies.
5. **Tautological tests** — tests that assert the code does whatever the code
   does, or that omit negative cases. Every feature card pre-specifies its key
   behavioral assertions, including **at least one negative case** a naive wrong
   implementation would fail.
6. **Easy-card cherry-picking** — repeatedly claiming low-risk docs/report/
   room-practice cards while articulation-point spine tips starve. Claim the
   highest-priority ready card on the serial tip; act on stalenness for
   unclaimed P0/P1 spine work before farming comfort work.
7. **Close-pump abuse** — closing cards (yours or a peer’s) to flood the ready
   pool, since closure unblocks dependents. **Only the planner closes** feature
   / continuous / STATUS work; false closes are reopened with an incident
   residual comment.
8. **Scope-splitting** — splitting one unit of work into types/impl/tests
   mini-closures to harvest multiple credits. Code and its tests ship in the
   **same card**; test-only follow-ups exist only for cross-cutting integration
   suites or planner-owned continuous re-record.
9. **Spec-editing as progress** — weakening a plan, card acceptance, pure-first
   gate, or frozen tip order instead of implementing it. Plan/process edits are
   a chore lane, never close feature cards, and frozen decisions (tip order,
   assist contract, dual-track rules) change only through planner decision.
10. **Conformance metastasis** — adding speculative checks, matrices, or
    reports because they are safe and satisfying. New checks must cite an
    **observed defect class** or a named release/tip gate.
11. **Dependency smuggling** — vendoring or shimming around banned assists,
    progression/capacity/door/event writes, or dual-track pollution to “make
    progress.” Batch verify and integrity enforce the deny-list; Clean vs
    assisted artifacts stay separate (`*_clean` only).
12. **Demo-path hardcoding** — special-casing pilot rooms, fixed coordinates, or
    development fixtures so the happy path passes. Conformance subjects are the
    named continuous-like sources and runtime-selected route tips, not scratch
    one-offs that differ from natural entry.

**Churn without progress is a process failure.** If a wave produces many
residuals, card opens/closes, or docs edits but no pure-green hop or continuous
tip advance, stop and re-triage to the highest-priority runnable feature card.

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
Clean-track runs use separate artifacts and must not change default CLI assists
or demote assisted continuous greens. Dual-track violations are **hard fails**.

## Hard-room splits (in-room geometry)

When a pure hop stays PARTIAL across serial one-knob cards with the **same**
acceptance checkbox red (especially place-proven finish vs natural approach
gap), use the hard-room playbook instead of more period/window thrash:

- Process + patterns: [`HARD_ROOM_SPLITS.md`](HARD_ROOM_SPLITS.md)
- Bubble → Bat ladder (first consumer): [`SM-K4.4-PHASE-LADDER.md`](SM-K4.4-PHASE-LADDER.md)

Summary rules (full detail in those docs):

1. **Phase ladder** — one card advances one phase (A mid · B height · C
   usable contact · D top · E door). Do not open D while C is red.
2. **Intermediate pure states** accelerate climb-only work; only full pure
   from continuous-like source claims hop GREEN.
3. **Place + velocity** recon before WJ/HJ knobs; place-at-rest alone is not
   natural proof.
4. **RECON → IMPL** ticket pair; no IMPL without a measured pin recipe.
5. **Stagnation @ 3 PARTIALs** on the same checkbox → mandatory planner
   triage (handoff state, new trajectory, topology rethink, or park) —
   not another constant on the same arc.

Probe hooks for Bubble (dev only, not hop GREEN):

```bash
# Capture first Phase C
uv run python super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source …/post_rising_tide_to_bubble_pure.state \
  --dump-phase-c …/post_bubble_right_contact_pure.state \
  --stop-at-phase-c --no-red-diag

# Climb-only from handoff
uv run python super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source …/post_bubble_right_contact_pure.state \
  --start-phase climb --no-red-diag
```

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
- Did not reward-hack (no process porn credit, proof-class inflation,
  gate self-weakening, false close, or scaffold-as-green)

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

### Residual lifecycle (living cards only)

Residuals and one-shot cards proliferate if left in `docs/tasks/` forever.
Enforce:

1. **Archive after successor.** When a pure hop’s successor card is **GREEN**
   (or the next hop is ready and this residual’s one-change is landed), move
   the predecessor residual / note to `docs/tasks/archive/` in the same
   hygiene pass that opens the successor living card.
2. **One living residual per open tip segment.** Prefer a single
   `SM-*-residual.md` for the active hop; do not stack R1/R2/R3 files without
   archiving the obsolete ones.
3. **CSV / boards stay authoritative.** Closed work is `done` or `parked` in
   `BACKLOG.csv` / MILESTONES; living markdown is only for ready/in-flight.
4. **Prune aggressively after continuous promote.** After a continuous tip
   lands, archive pure residuals for that stack and leave only stabilize /
   status follow-ons (or none).

Schema still requires exact **Next card ID** + **one change** on every residual.
Do not invent mega-residuals that mix pure + continuous + STATUS.

**Ticket size:** one pure hop or one residual knob per card; prefer 30–90 min
agent sessions. STATUS / tracker updates are planner-owned or tiny Flash
follow-ons.

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

## Near-term sequence (post–Frog Save tip, 2026-08-02)

Verified continuous tip is power-on → Frog Save (`--to frog`, 114,923f ×2).
Process discipline stays fixed; composition past Frog is planner-gated.
First Bubble path is **Cathedral climb** (no Speed) — not Speedway→Farm.

1. **Serial pure stack:** `SM-K4-CATH-03` → Bubble (CATH-04) → Speed/Wave/Ice
   pure from continuous-like sources; one-knob + residual next-card ID.
2. After each pure green: graph edge → compose tip (planner) → dual continuous
   re-record → STATUS promote only after integrity green.
3. **Stabilize wave** after every continuous tip land before stacking knobs.
4. Primitive extraction from green tightens into `controller_common` when a
   second consumer exists.
5. Flash: board/STATUS proposals; keep `SOURCE_STATES.md` current; archive
   residuals after successors (lifecycle above).
6. **Parallel dual-track:** room farm (`farm_room_waves.sh`), Clean bombs tip
   (`*_clean` only), 1–2 ARCH items, boss **primitives** only — width ≤ 8,
   own-files, never mutate default assists or claim continuous greens.
7. **Structure plan (planner-owned):** hop-table extract, selective-RAM gate,
   pure RED diagnostics — [`../plan.md`](../plan.md) ·
   [`../ARCHITECTURE.md`](../ARCHITECTURE.md).
8. After Speed/Wave/Ice continuous tips: Alpha PB → Moat → natural Phantoon
   entry (fights gated on natural entry).

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
