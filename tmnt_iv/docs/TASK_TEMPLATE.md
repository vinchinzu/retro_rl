# TMNT IV atomic task card

Cacheable format for **executor** sessions. Planner (Grok / human) owns design,
continuous integrity, and STATUS; executor owns mechanical implementation.

| Doc | Role |
|-----|------|
| [`docs/tasks/QUEUE.md`](tasks/QUEUE.md) | Live wave board + residuals |
| [`docs/tasks/PROCESS.md`](tasks/PROCESS.md) | Multi-entry, one-knob, residual schema |
| [`docs/CLEAN_TRACK.md`](CLEAN_TRACK.md) | Clean privilege-reduction track |
| [`docs/CLEAN_PLAYBOOK.md`](CLEAN_PLAYBOOK.md) | Play lessons (do not relearn) |

**Public-commit hygiene:** never commit session logs, API keys, or absolute
home paths. Cards use recipe IDs only.

## Role split

| Role | Who | Owns |
|------|-----|------|
| Planner / reviewer | Grok or strong model + human | Stage order, natural-entry judgment, STATUS promotion, zero-write integrity, continuous re-record |
| Executor | Flash / Luna / any bounded agent | Policy one-knobs, probes, tests, docs **proposals** |

**Never** give the executor: “figure out Clean for the whole game” or open
exploration. **Always** give exact files, recipe step, and acceptance checks.

## Model pick

| Work | Prefer | Why |
|------|--------|-----|
| Docs / tracker / STATUS **proposal** | Flash | Fast, low risk |
| Probe scripts, unit tests, CLI wiring | Luna | Scope discipline |
| Policy one-knob from listed residual | Luna | Bounded; planner reviews residual |
| Continuous compose, STATUS **apply** | Planner | Integrity gate |

## Non-negotiable card rules (from PROCESS)

1. **Multi-entry first:** Clean stage cards must green a suite (checkpoint +
   continuous-faithful / power-on), not checkpoint alone.
2. **One knob:** policy cards change one named constant group or one tactic.
   Interacting knobs serialize across cards + stabilize wave.
3. **No dual spine knobs → same continuous** without intervening suite +
   dry-run gate (planner stabilize wave).
4. **Residual → next card:** every residual ends with `Next card ID` +
   **one** change.
5. **Own files only:** list paths; parallel cards must not share hot modules
   (`policy.py` sections, `record_full_hard_run.py`, `STATUS.md`).

## Card template

```markdown
# TASK <id>: <one-line goal>

## Recipe step
docs | infra | probe suite | policy knob | continuous | stabilize | status

## Model
Flash | Luna   # optional

## Wave type
implement | stabilize   # stabilize = re-verify only, no new knobs

## Own files only
- path/a.py
- path/b.py  # create | optional residual note

## Context (minimal)
- Assisted continuous: 00:57:19 / 4,667 dmg / 65 e-heals (BASELINE_METRICS)
- Clean track: docs/CLEAN_TRACK.md
- Playbook: docs/CLEAN_PLAYBOOK.md
- Process: docs/tasks/PROCESS.md
- (if probe) Source state + stage byte

## Read first (only these)
- docs/CLEAN_PLAYBOOK.md  # anti-patterns
- scripts/probe_stageN_clean.py  # pattern

## Do
1. …
2. …

## Do not
- Touch record_full_hard_run defaults / STATUS.md unless the card says so
- Claim continuous / Clean green without suite evidence
- Re-open playbook-banned traps (global pizza seek, Stage1 jump-through, …)
- Progression / stage / lives RAM writes
- Second interacting spine knob in the same card
- Force-pass suite / continuous from unit tests alone
- Overwrite assisted `tmnt_iv_full_hard_*` baselines with clean runs

## Acceptance
- [ ] `uv run pytest <narrow tests> -q` green (if code)
- [ ] (if Clean suite) multi-entry 0 e-heals, 0 lives lost
- [ ] No unrelated file churn
- [ ] Residual uses PROCESS schema (next card ID + one change)

## Verify commands
```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_stageN_clean --suite
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
What still blocks suite green / continuous / STATUS (bullet list).

### Next action (required)
- **Next card ID:** T4-XXXX   # or PLANNER-GATE / none
- **One change:** single knob or single decision (one sentence)
- **Source state:** path or “needs capture: T4-*-SRC”

### Non-claims
- Did not STATUS-promote
- Did not forge stage/lives/boss/event RAM
- Not continuous Clean evidence (unless card was continuous — then planner only)

### Probe pin (if suite/geometry) — **mandatory metrics**
stage=… event=… hp=… lives=… frames=… damage=… e_heals=… pizza=… outcome=…
```

On **RED** suite: residual must name the failure mode (life_loss mid-wave,
boss finish, spike, soft-lock) and **one** next knob. Do not “debug dark.”

**Force-pass ban:** suite green and continuous Clean are never claimed from
scaffolds, diagnostics, or unit tests alone.
