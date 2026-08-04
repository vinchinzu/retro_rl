# Super Metroid sub-agent process

Planner–executor loop for KPDR continuous integrity. Cards:
[`docs/tasks/`](./). Queue: [`QUEUE.md`](QUEUE.md). Template:
[`docs/TASK_TEMPLATE.md`](../TASK_TEMPLATE.md).

## Roles

| Role | Who | Owns |
|------|-----|------|
| Planner | Grok / human | Continuous integrity, STATUS, natural-entry design, tip order, promote/revert |
| Executor Flash | OpenCode + Flash | Tracker/docs, STATUS **proposals**, source-state index |
| Executor Luna | OpenCode + Luna | Controllers, tests, pure probes, tip wiring skeletons |

Never hand the executor open-ended “next continuous tip” work.

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
successor is ready or the hop is continuous-promoted. One living residual per
open tip segment. CSV / MILESTONES are authoritative for done work.

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
| [`routes/BACKLOG.csv`](../routes/BACKLOG.csv) | Full ticket buffer |
| [`routes/MILESTONES.md`](../routes/MILESTONES.md) | Product tip board |
