# Super Metroid atomic task card

Cacheable format for **OpenCode + cheap executor** sessions (role nicknames
**Flash** / **Luna**). Planner (Grok / human) owns design and integrity;
executor owns mechanical implementation.

| Doc | Role |
|-----|------|
| [`docs/tasks/QUEUE.md`](tasks/QUEUE.md) | Live ready / in-flight only |
| [`docs/tasks/PROCESS.md`](tasks/PROCESS.md) | Pure-first, stabilize, residual schema |
| [`docs/SOURCE_STATES.md`](SOURCE_STATES.md) | Pure entry state index |
| Dispatch | `./snes/super_metroid/scripts/dispatch_opencode.sh <CARD-ID>` |

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
6. **Delete on close:** residuals and closed cards are ephemeral — delete after
   successor ready or continuous promote. Prefer a QUEUE/BACKLOG line over an
   archive card.

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
- Continuous tip verified: power-on → Bat Cave (`--to bat_cave`, 122,304f)
- Next play: Bat → Speed Hall → Speed / Wave / Ice (see ROUTE_KPDR K4)
- Process: docs/tasks/PROCESS.md
- (if pure probe) Source state path + expected room id hex
  (prefer docs/SOURCE_STATES.md row)

## Read first (only these)
# Paths: OpenCode --dir may be super_metroid/; prefer package-relative
# (tests/..., routes/..., docs/...). Shell verify from monorepo root uses
# snes/super_metroid/... under uv.
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
uv run pytest snes/super_metroid/tests/test_foo.py -q
# pure probe only when card requires green (list exact --source path):
# uv run python snes/super_metroid/scripts/probe/kpdr.py pure <seg> --source <state>
```

## Done when
Executor returns residual (schema below). Integrity / STATUS stays with
planner/reviewer. On close: delete residual file + card (or leave only the
next living card); history lives in BACKLOG/MILESTONES/STATUS.
```

## Residual schema (required)

Paste into the final message; optional file `docs/tasks/<ID>-residual.md`
(**delete after successor / close** — do not archive).

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
- Update KPDR_TRACKER.csv + regenerate md (**or** STATUS proposal).
- Unit tests for StateRequirement / tip registration.
- Geometry hop with **named source state** and expected room id.
- Primitive extract after pure+continuous green (`controller_common` + tests).
- Source-state capture card that only dumps + catalogs a state.

**Bad**

- “Figure out the next continuous tip after Bat and implement it.”
- Multi-room continuous compose + STATUS promotion in one card.
- Geometry debug without a source state and room bounds.
- Optional pure probe using the **wrong room** fixture.
- Two interacting knobs in one continuous without pure + continuous gate.
- Practice-room greens claimed as continuous spine evidence.
- Keeping closed cards / residual trees as archive.

## Executor session hygiene

```bash
# From monorepo root — dispatch owns model pick, logging, ownership conflicts
./snes/super_metroid/scripts/dispatch_opencode.sh SM-BAT-SPEED-PURE
./snes/super_metroid/scripts/dispatch_opencode.sh --flash SM-ROLLUP-STATUS
./snes/super_metroid/scripts/dispatch_opencode.sh SM-A SM-B  # parallel if disjoint
```

- One session per card. Context target: 10–40k, not 240k.
- Prefer file references over pasting large blobs.
- Scaffold cards may allow pure-probe **bonus**; pure-green cards require green
  or an explicit residual note (never force-pass).
- When pasting probe stdout into reports, use **repo-relative** paths only
  (no `/home/...` prefixes).

## Integrity gate (planner only)

Before promoting STATUS / continuous:

1. Natural entry from real predecessor (not door-warp alone).
2. Pure-green on the new hop(s) from continuous-like source.
3. `snes/super_metroid/scripts/record/continuous.py --to <tip> --no-video`
   integrity green (0 state loads, 0 progression writes).
4. Tracker + graph verification promoted only after that report.
5. Cheap models may first-pass lint/tests; they do **not** own the claim.
6. After pure green on one reverse hop: promote that edge to
   `controller_dev` only — never jump to continuous.
7. After implement/stress knobs land: run a **stabilize** continuous re-record
   before the next implement wave.
