# Super Metroid sub-agent process

Planner–executor loop for KPDR continuous integrity. Cards:
[`docs/tasks/`](./). Live queue: **`bd ready -l super_metroid`** (beads).
Markdown snapshot: [`QUEUE.md`](QUEUE.md). Template:
[`docs/TASK_TEMPLATE.md`](../TASK_TEMPLATE.md).

## Beads (primary work tracker)

Monorepo **bd** prefix `rr-`. Product claims still live in STATUS /
MILESTONES / pure residuals — beads track ready / in-flight / blocked only.

```bash
bd ready -l super_metroid          # unblocked next work
bd update <id> --status in_progress
# … pure-first implement …
bd close <id> --reason "…"
bd create "Found X" -p 1 -l super_metroid,pure --deps discovered-from:<id>
bd sync && git add .beads/issues.jsonl   # with code commit
```

Labels: `super_metroid` + kind (`pure` | `graph` | `compose` | `stabilize` |
`status` | `clean` | `meta`) + optional epic (`k4`, `spazer`, …).
External refs keep BACKLOG ids (`SM-K4.5-PURE`).

**Do not** open a second tracker. QUEUE.md is a human snapshot of `bd ready`,
not the source of truth.

## Roles

| Role | Who | Owns |
|------|-----|------|
| Planner | Grok / human | Continuous integrity, STATUS, natural-entry design, tip order, promote/revert, beads triage |
| Executor Flash | OpenCode + Flash | Tracker/docs, STATUS **proposals**, source-state index |
| Executor Luna | OpenCode + Luna | Controllers, tests, pure probes, tip wiring skeletons |

Never hand the executor open-ended “next continuous tip” work. Point at one
bead id + residual/card acceptance.

## Pure-first + stabilize waves

Spine controller changes must **pure-green** from a continuous-like source
before continuous re-record.

| Wave type | Allowed work | Exit gate |
|-----------|--------------|-----------|
| **Implement / stress** | One-knob pure probes, scaffolds, units | Pure green (or explicit BLOCKED residual) |
| **Stabilize** | Pure re-verify + continuous re-record only | Continuous green; no new knobs |

1. After an implement wave lands live spine knobs, run a **stabilize wave**
   before stacking more knobs.
2. Never land **two interacting spine knobs** in the same continuous without
   intervening pure + continuous gates.
3. Continuous re-record is **planner-only** for STATUS claims.
4. One-knob: each geometry card changes **one** named primitive (or constant
   group). Tip wiring (graph/catalog/`run_to`) is a separate card after pure.

## Residual schema (ephemeral)

Optional `docs/tasks/<ID>-residual.md` + final message must include:

```markdown
## Residual — <card-id>
### Result
GREEN | RED | BLOCKED | PARTIAL
### Files changed
- path — one-line purpose
### Verify paste
(command + exit code + stdout; repo-relative paths only)
### Acceptance
- [x]/[ ] each card checkbox
### Residual risks
- …
### Next action (required)
- **Next card ID:** SM-XXXX | PLANNER-GATE | none
- **One change:** single knob or decision (one sentence)
- **Source state:** path | needs capture: SM-*-SRC
### Non-claims
- Did not STATUS-promote / forge progression RAM / claim continuous evidence
### Probe pin (if pure/geometry)
room=0x…. pose=… x=… y=… door_transition=… frames=… last_pin=…
```

**Lifecycle:** residuals are ephemeral. Delete residual + closed card when the
successor is ready or the hop is continuous-promoted. **One living residual per
open tip segment** — delete on promote/close; do not archive trees. CSV /
MILESTONES are authoritative for done work.

## Room policy layout (prevention)

**Gold standard:** `routes/kpdr/spazer/` (package from day 1: `geometry`,
`scripts`/`data`, hop modules, shared helpers). **Cautionary tale:** Wave
megafile (`k4_wave.py` ~1.3k lines) + bare/`_DOOR_X` shadow across rooms.
Do not land the next multi-hop tip (e.g. Ice pure) as another megafile.

### Layout checklist

- [ ] **Package early** — multi-hop tip (≥2 pure hops or human RLE) → package
      under `routes/kpdr/<tip>/` from day 1 (mirror `spazer/`). Single-hop
      one-knobs may stay a module; split before the second hop lands.
- [ ] **Room-prefixed geometry** — constants live in `geometry.py` with
      **room-prefixed** names (`BSC_DOOR_X`, `DC_DOOR_X_MIN`, …). Never bare
      `_DOOR_X` / `DOOR_X` in multi-hop modules (shadow risk across rooms).
- [ ] **Human tape as data** — RLE → `routes/kpdr/data/*.json` via the parse
      tool; product loads JSON. No inlined RLE tuples in controller modules.
- [ ] **Shared helpers only** — `play_script`, `escape_knockback`,
      `wait_ordinary_room`, `require_room` (and package `helpers` that wrap
      them). No private reimplementation of settle / knockback / script play.
- [ ] **Size cap** — file under ~500 lines or split before the next knob;
      never grow past **1k** without a package split first.

### Tip extension recipe (after pure green)

Same pure-first gates as above. Do **not** add a new `start_to_*.py` / per-tip
runner pair.

1. Pure controller (+ `KPDR_SEGMENTS`)
2. Graph edges in `progression/stages/` (re-exported via `progression/data.py`)
3. `SpineHop` + `TipSegment` CLI fields (hop groups / `TipSpec` via spine + hops)
4. `run_to` automatic via `TipSpec` — no per-tip runners

Full continuous promote/record steps: [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md)
§ Continuous tip extension recipe. Catalog mirror: `routes/catalog.py` module
doc.

### Residual lifecycle (tip segments)

- One living residual per **open tip segment**
- Delete residual (+ closed card) on promote or close
- Next action always names the successor bead/card or `PLANNER-GATE`

## Hot modules (never parallel-edit)

- `routes/kpdr/business_climb.py`
- `routes/kpdr/hijump_return.py` (and HJ gray-door knobs)
- `routes/spore_spawn_controller.py` / spore fight dwell knobs
- geometry on `routes/kpdr/varia_return.py`
- `routes/continuous.py`, `docs/STATUS.md` (planner only)
- also serialize: `progression.py`, `catalog.py` when tip-wiring

Dispatch rejects parallel cards that share these paths.

## Dual-track rule

`ROOM_WORK_QUEUE` / room-practice farm greens are **not** continuous evidence
and are **not** the product work assignment board. Product next-work is only
`docs/STATUS.md` + `docs/tasks/QUEUE.md`. Practice metrics stay dual-track.

Clean-track runs use `*_clean` artifacts only and must not change default CLI
assists or demote assisted continuous greens. Dual-track pollution is a hard
fail.

Room farm is **planner opt-in** research (not the default P0 path):

```bash
./snes/super_metroid/scripts/farm_room_waves.sh --rounds 10 --parallel 8
```

## Force-pass ban + honesty

Pure geometry and continuous integrity are never green from scaffolds,
diagnostics, or unit tests alone. Honesty is absolute.

Forbidden patterns (short list):

1. **Gate self-weakening** — editing integrity/pure/STATUS gates to “pass.”
2. **Proof-class inflation** — fixtures, door-warps, mocks as live pure/continuous.
3. **Golden regen reflex** — rewriting baselines instead of fixing the controller.
4. **Scaffold-as-green** — placeholders / force-pass paths claiming pure or tip green.
5. **Easy-card cherry-pick** — farming practice/docs while the serial spine tip starves.

Refusal / honest BLOCKED beats a fabricated green. Scaffold-only work is
PARTIAL at most — never closes a feature card.

## Tangible progress (no process porn)

Process exists to ship runnable route software. Residuals, boards, and meta
docs are not progress.

- Open cards deliver controllers, graph edges, pure probes, or continuous tips.
- **Delete cards on close.** When in doubt, list work as a QUEUE/BACKLOG line
  instead of keeping archive cards.
- Churn without pure-green hop or continuous tip advance is a process failure —
  stop and re-triage to the highest-priority runnable feature.

## Hard-room splits

When a pure hop stays PARTIAL across ≥3 serial one-knob cards on the same
checkbox, use [`HARD_ROOM_SPLITS.md`](HARD_ROOM_SPLITS.md) (phase ladder,
stagnation triage). Do not thrash the same constant class.

## Pointers

| Doc | Role |
|-----|------|
| [`TASK_TEMPLATE.md`](../TASK_TEMPLATE.md) | Card format + residual paste |
| [`SOURCE_STATES.md`](../SOURCE_STATES.md) | Pure entry states |
| [`STATUS.md`](../STATUS.md) | Verified continuous tip (planner) |
| [`QUEUE.md`](QUEUE.md) | Live ready / in-flight only |
| [`HARD_ROOM_SPLITS.md`](HARD_ROOM_SPLITS.md) | In-room geometry playbook |
| [`ARCHITECTURE.md`](../ARCHITECTURE.md) | Tip extension recipe + hop graph |
| `routes/kpdr/spazer/` | **Gold-standard** multi-hop package layout |
| [`routes/BACKLOG.csv`](../routes/BACKLOG.csv) | Full ticket buffer |
| [`routes/MILESTONES.md`](../routes/MILESTONES.md) | Product tip board |
