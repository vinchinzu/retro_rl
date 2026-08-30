# Hard-room splits — process ideas for future geometry

Session halt lives in `.grok/skills/sm-pure-hop/`. This page is the phase
ladder only — do not farm window/period knobs past three serial PARTIALs.

Reusable playbook when a pure hop stays **PARTIAL** across serial one-knob
cards (same acceptance checkbox red). First consumer: Bubble → Bat (continuous tip **done** 122304f; techniques in [`BUBBLE_TECHNIQUES.md`](BUBBLE_TECHNIQUES.md)).

This document captures **working patterns**, not ceremony. Use it when full
pure thrash is expensive and place-isolation has outrun natural approach.

## When to open this playbook

Open a hard-room triage (planner) when **any** of:

1. Same pure acceptance checkbox red after **≥3** serial PARTIAL residuals.
2. Place/isolation proves a finish, but full pure never enters that band.
3. Metric noise (max_x thrash, frame count) moves while the real pin
   (height class, contact band, top, door) does not.
4. Executors keep retuning the same constant class (period, y-window, charge
   frames) without a new trajectory hypothesis.

Do **not** farm more window/period knobs past the stagnation budget.

## Phase ladder (in-room acceptance)

Door-aligned hops (CATH-01…04) already split at room boundaries. Hard
**in-room** climbs need invented phase edges:

| Phase | Meaning | Exit gate (example) |
|-------|---------|---------------------|
| A | Entry → mid pin / first stable seat | standing pin band |
| B | Height class / launch peak | `min_y ≤ H` |
| C | **Usable** structure contact at height | band `(x,y)` + pose/vel class — **not** thrash `max_x` |
| D | Top / goal platform | `top_reached` |
| E | Door / ordinary next room | room id + settle |

Rules:

- One living card advances **one** red phase checkbox.
- Do not open phase-D work while phase C is still red.
- Freeze green phases (do not re-touch R5/R6-class lower/lip once green
  unless that phase regresses).
- Full pure from continuous-like source remains the only **GREEN claim** for
  the hop. Intermediate states accelerate development only.

## Intermediate pure states (dev accelerators)

Capture scratch states at phase exits from the **natural** controller path:

```text
full pure (CATH-like source)
  → dump at first Phase-C hit   → climb-only iteration
  → full pure re-compose          → only path that can claim GREEN
```

- Intermediate dump ≠ continuous evidence, ≠ STATUS, ≠ forged progress.
- Prefer save-state capture over place-at-rest: freezes pose **and** velocity.
- Climb-only / phase probes may use `--start-phase climb` (Bubble) and
  never replace full pure acceptance for hop closeout.

Harness (Bubble):

```bash
# Capture first usable right contact (Phase C) — diagnostic success if hit
uv run python snes/super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state \
  --dump-phase-c super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bubble_right_contact_pure.state \
  --stop-at-phase-c --no-red-diag

# Climb-only iteration from that handoff (or place air band)
uv run python snes/super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bubble_right_contact_pure.state \
  --start-phase climb --no-red-diag
```

## Place recon must match velocity

Place-at-rest `(x,y)` proves **geometry existence**. Natural contact often
arrives with fall velocity. Before shipping a WJ/HJ knob:

1. Trace best pure around the gap: `(frame,x,y,pose,vx,vy,reason)`.
2. Diff vs place-success envelope that reaches the next phase.
3. Recon seeds **place + velocity** or uses a captured save-state.

If velocity-matched place at natural first contact **cannot** complete the
next phase, the card class is wrong (need new trajectory / mid seat), not
“WJ period again.”

## Ticket pair: RECON → IMPL

| Ticket | Owns | Exit |
|--------|------|------|
| **RECON** | pin dump, place+vel grid, one numbered hypothesis | residual with exact pin + button recipe (no controller claim) |
| **IMPL** | one named controller change | phase checkbox green **or** honest BLOCKED |

No IMPL card until RECON residual names the measured approach. Exploratory
IMPL that rediscovers the same gap is thrash.

## Stagnation rule (attempt budget)

After **3** serial PARTIALs with the **same** phase acceptance still red:

**Mandatory planner triage.** Allowed next moves only:

1. Intermediate state + phase-only pure loop, or
2. New movement primitive / trajectory (not a window on the same arc), or
3. Route topology rethink (different door/order if product allows), or
4. Temporary park + parallel work if the spine tip can wait.

Forbidden without triage: another period / y-band / charge-frame tweak on
the same open-loop path.

## Controller structure

Prefer phase-local helpers once a phase greens:

```text
bubble_lower() → bubble_run_mid(start=launch) → bubble_top_super_door()
```

- Unit or pure phase probes lock green phases.
- Living residual knobs edit only the red phase function.
- Promote sequences that survive pure (+ continuous later) into
  `controller_common` / local primitives per PROCESS.

## Patterns table (program-wide)

| Pattern | Use when | Anti-pattern |
|---------|----------|--------------|
| Door-aligned hops | room transitions | one mega-controller for 4 doors |
| In-room phase ladder | multi-story climb; place ≠ natural | R1…Rn constant thrash on full pure |
| Mid pure handoff states | hard middle; lower green | replaying full entry every probe |
| RECON → IMPL pair | physics unknown | IMPL that “explores” |
| Place + velocity grid | WJ/HJ air games | place-at-rest only |
| Primitive extract after phase green | sequence repeats | mega open-loop in `play_*` |
| Stagnation triage @ 3 | same checkbox red | infinite PARTIAL series |

## Non-claims

- Hard-room tooling never weakens pure-first or continuous integrity.
- Phase-C capture / climb-only green is **not** hop GREEN to the next room.
- No STATUS promote from intermediate states.
- Dual-track room practice remains separate from KPDR continuous.

## Bubble → Bat (first consumer) — closed

Phases A–E green; pure R19 + continuous tip **122,304f** → Bat Cave. Technique
ref (maintenance only): [`BUBBLE_TECHNIQUES.md`](BUBBLE_TECHNIQUES.md). Card
stack deleted after promote. Next serial: `bd ready -l super_metroid`.

## Main Shaft → Attic (second consumer) — closed

Phases A–E green; natural Basement-leave → Attic is dual-exact **2,745f**
×2. Seats live in `routes/kpdr/wrecked_ship/ws_main_geometry.py`. The
next open boundary on `rr-kw8t` is Attic → West Ocean; evidence and probe
CLI: [`rr-kw8t-residual.md`](rr-kw8t-residual.md).

## See also

- Pure-first: `AGENTS.md`
- Live work: [`rr-kw8t-residual.md`](rr-kw8t-residual.md) while `rr-kw8t` is in_progress
- Source catalog: [`../SOURCE_STATES.md`](../SOURCE_STATES.md)
