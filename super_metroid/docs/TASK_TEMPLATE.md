# Super Metroid atomic task card

Cacheable format for **OpenCode + cheap executor** sessions (role nicknames
**Flash** / **Luna**). Planner (Grok / human) owns design and integrity;
executor owns mechanical implementation.

| Doc | Role |
|-----|------|
| [`docs/tasks/QUEUE.md`](tasks/QUEUE.md) | Live wave board + residuals |
| [`docs/tasks/PROCESS.md`](tasks/PROCESS.md) | Pure-first, stabilize waves, residual schema |
| [`docs/SOURCE_STATES.md`](SOURCE_STATES.md) | Pure entry state index |
| Dispatch | `./super_metroid/scripts/dispatch_opencode.sh SM-K4-NN` |

**Public-commit hygiene:** never commit session logs, API keys, or absolute
home paths. Provider model IDs live only in `scripts/dispatch_opencode.sh`
(and a local `opencode.json` copied from `opencode.example.json`). Cards use
Flash/Luna nicknames only.

## Role split

| Role | Who | Owns |
|------|-----|------|
| Planner / reviewer | Grok or strong model + human | Tip design, natural-entry judgment, STATUS promotion, zero-write integrity, continuous re-record |
| Executor | OpenCode + Flash / Luna | Controllers from a clear recipe, policy scaffolds, tests, tracker/docs **proposals**, graph edges marked `controller_dev` |

**Never** give the executor: “figure out the next continuous tip” or open
exploration. **Always** give exact files, recipe step, and acceptance checks.

## Model pick

| Work | Prefer | Why |
|------|--------|-----|
| Tracker / CSV / offline dwell report / docs-only / STATUS **proposal** | Flash | Fast, low risk |
| Unit contracts, registration, controller scaffold, primitive extract | Luna | Strong scope discipline |
| Geometry pure green with **explicit source state path** | Luna | Bounded; still planner reviews residual |
| Natural entry design, continuous compose, STATUS **apply** | Planner | Integrity gate |

## Non-negotiable card rules (from PROCESS)

1. **Pure-first:** spine controller cards that claim green must pure-green from
   a continuous-like source listed in `SOURCE_STATES.md` (or capture first).
2. **One knob:** geometry / efficiency cards change one named constant or
   primitive. Interacting knobs serialize across cards + stabilize wave.
3. **No dual spine knobs → same continuous** without intervening pure +
   continuous gate (planner stabilize wave).
4. **Residual → next card:** every residual ends with `Next card ID` +
   **one** change (see schema below).
5. **Own files only:** list paths; dispatch rejects parallel cards that share
   hot modules.

## Card template

```markdown
# TASK <id>: <one-line goal>

## Recipe step
1 pure controller | 2 graph edge | 3 catalog tip | 4 continuous hops |
5 run_to wire | 6 record baseline | 7 STATUS promote | docs | efficiency |
primitive promote | stabilize | source capture

## Model
Flash | Luna   # optional hint for dispatch script

## Wave type
implement | stabilize   # stabilize = pure re-verify + continuous only

## Own files only
- path/a.py
- path/b.py  # create | optional residual note

## Context (minimal)
- Continuous tip verified: power-on → Varia (`--to varia`, 101,954f)
- Next play: K4 return → Business → Bubble → Speed (see ROUTE_KPDR K4)
- Architecture: docs/ARCHITECTURE.md continuous tip-extension recipe
- Process: docs/tasks/PROCESS.md
- (if pure probe) Source state path + expected room id hex
  (prefer docs/SOURCE_STATES.md row)

## Read first (only these)
# Paths: OpenCode --dir is super_metroid/, so prefer package-relative
# (tests/..., routes/..., docs/...). Shell verify may still use
# super_metroid/... when uv is invoked from monorepo root.
- tests/test_foo.py  # why
- routes/kpdr/bar.py  # style reference

## Do
1. …
2. …

## Do not
- Touch continuous.py / STATUS.md unless the card says so
- Claim continuous / integrity green without a green report
- Add start_to_*.py scripts
- Progression/capacity RAM writes
- Full-bank WRAM copies in hot loops
- Second interacting spine knob in the same card
- Force-pass pure / continuous from scaffolds or units alone

## Acceptance
- [ ] `uv run pytest <narrow tests> -q` green
- [ ] (if controller pure-green card) pure probe green from listed source state
- [ ] No unrelated file churn
- [ ] Residual uses PROCESS schema (next card ID + one change)

## Verify commands
```bash
uv run pytest super_metroid/tests/test_foo.py -q
# pure probe only when card requires green (list exact --source path):
# uv run python super_metroid/scripts/probe/kpdr.py pure <seg> --source <state>
```

## Done when
Executor returns residual (schema below). Integrity / STATUS stays with
planner/reviewer.
```

## Residual schema (required)

Paste into the final message; optional file `docs/tasks/<ID>-residual.md`.

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
- …

### Next action (required)
- **Next card ID:** SM-XXXX | PLANNER-GATE | none
- **One change:** …
- **Source state:** path | needs capture: SM-*-SRC

### Non-claims
- Did not STATUS-promote
- Did not forge progression/capacity/door/event/boss RAM
- Not continuous evidence (unless planner continuous card)

### Probe pin (if pure/geometry) — mandatory metrics
room=0x…. pose=… x=… y=… door_transition=…
frames=… dwell=… last_pin=…
```

On RED pure: prefer last pin + clip path (when available) over prose-only
failure notes. Next action still requires **one** next card ID + **one** change.

## Good vs bad scopes

**Good**

- Implement pure `play_X_to_Y` in `routes/kpdr/` following `warehouse.py` style.
  Entry: room A ordinary. Exit: room B + item mask. No continuous.py.
- Scaffold PolicySegment + JSON from natural-entry fixture.
- From a dwell report, tighten one high-dwell slice; report frame delta.
- Add DoorEdge + milestone; mark `verification=controller_dev`.
- Update KPDR_TRACKER.csv + regenerate md (**or** SM-ROLLUP-STATUS proposal).
- Unit tests for StateRequirement / tip registration.
- Geometry hop with **named source state** and expected room id (e.g.
  `scratch/post_varia_to_kraid_pure.state` → 0xA59F).
- Primitive extract after pure+continuous green (`controller_common` + tests).
- Source-state capture card that only dumps + catalogs a state.

**Bad**

- “Figure out the next continuous tip after Varia and implement it.”
- Multi-room continuous compose + STATUS promotion in one card.
- Geometry debug without a source state and room bounds.
- Optional pure probe using the **wrong room** fixture (e.g. post-Varia collect
  in 0xA6E2 when the hop needs 0xA59F) — card must name the correct state.
- Two interacting knobs (settle **and** setup jumps) in one continuous without
  pure + continuous gate between them.
- Practice-room greens claimed as continuous spine evidence.

## Executor session hygiene

```bash
# From repo root — dispatch owns model pick, logging, ownership conflicts
./super_metroid/scripts/dispatch_opencode.sh SM-K4-03
./super_metroid/scripts/dispatch_opencode.sh --flash SM-K4-05
./super_metroid/scripts/dispatch_opencode.sh SM-K4-03 SM-K4-04 SM-K4-05  # parallel if disjoint
```

- One session per card. Context target: 10–40k, not 240k.
- Prefer file references over pasting large blobs.
- Stable recipe + AGENTS.md + ARCHITECTURE.md + PROCESS.md stay in project
  OpenCode instructions for cache hits.
- Scaffold cards may allow pure-probe **bonus**; pure-green cards require green
  or an explicit residual note (never force-pass).
- When pasting probe stdout into reports, use **repo-relative** paths only
  (no `/home/...` prefixes).

## Integrity gate (planner only)

Before promoting STATUS / continuous:

1. Natural entry from real predecessor (not door-warp alone).
2. Pure-green on the new hop(s) from continuous-like source.
3. `scripts/record/continuous.py --to <tip> --no-video` integrity green
   (0 state loads, 0 progression writes).
4. Tracker + graph verification promoted only after that report.
5. Cheap models may first-pass lint/tests; they do **not** own the claim.
6. After pure green on one reverse hop: promote that edge to
   `controller_dev` only — never jump to continuous.
7. After implement/stress knobs land: run a **stabilize** continuous re-record
   before the next implement wave.
