# TASK SM-K4.4-PURE-R18: Phase D — pure velocity dump + wall-jump contact

## Recipe step

1. Pure controller (Phase D only). Geometry green before graph / continuous.

## Model

Luna

## Wave type

implement

## Own files only

- `routes/kpdr/bubble_mountain_primitives.py` — closed-loop WJ and/or search
  recipe; entry guards only on red phase
- `routes/kpdr/bubble_mountain_mid.py` — **one** named launch/climb change that
  uses the dump-proven recipe (do not re-touch R5/R6 lower/lip)
- `routes/kpdr/bubble_mountain_params.py` — only if a new named constant is
  proven on pure dump
- `scripts/probe/bubble_r18_*.py` — optional recon / short-horizon search CLI
- `tests/test_k4_norfair_scaffold.py` — unit for new predicate / params
- `docs/tasks/SM-K4.4-PURE-R18-residual.md` — required residual
- Tip boards only as needed (`AGENTS.md`, `QUEUE.md`, `BUBBLE_*`, phase ladder)

Do **not** edit `continuous.py`, `STATUS.md`, CATH controllers, or progression
ranks. Do **not** open Phase E / Super door until pure `top_reached`.

## Source and contract

- Full pure GREEN claim only from:
  `custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state`
- Expected entry: `0xACB3` (CATH-04 successor)
- Phase D gate: `y≤200` and `x≥300` (`bubble_phase_d_top_band` / `top_reached`)
- Caps: Morph, Bombs, Missiles, Supers (≥1), Hi-Jump, Varia — **no Speed**
- Human isolation (dev only, already GREEN):
  `scratch/bubble_human_runway.state` + `bubble_double_walljump_r15` family

## Context (R17 RECON — read first)

Authoritative residual:
[`SM-K4.4-PURE-R17-residual.md`](SM-K4.4-PURE-R17-residual.md).

| Proven | Still red |
|--------|-----------|
| Phases A/B/C pure green | Phase D pure `top_reached` |
| Human pin Phase D GREEN (p132 @ ~264,297) | Pure never earns p132 on fire arc |
| R16 pure fire seat min_y=228, phase_c | Integer (27,395)p2 ≠ human velocity |
| Named double-WJ primitive | Open-loop R15 on pure dump → mx200=0 |

**Forbidden (no new pin):** another period / y-band / charge-frame / run-frame
tweak on the same open-loop arc. Rejected list is in R17 residual.

**Spine stays on Bubble→Bat.** Do not divert to park / CATH / farm for this card.

## Do (ordered)

### 1) Capture pure velocity dumps (RECON → state files)

From full pure (or lower+fire only) on CATH-04 source, dump **save-states**
(not place-at-rest) at:

| Dump ID | When | Path suggestion |
|---------|------|-----------------|
| fire seat | first grounded fire-window seat after lower | `scratch/post_bubble_fire_seat_pure.state` |
| post_run | end of RIGHT+B run windup | `scratch/post_bubble_fire_postrun_pure.state` |
| post_spin / wall approach | first frame with x≥240, y∈[200,340], falling or spin | `scratch/post_bubble_wall_approach_pure.state` |

Trace fields every frame in the critical window: `f,x,y,pose,vx,vy,buttons`.
Prefer `scripts/probe/bubble_trace_and_seat.py` patterns or a new
`bubble_r18_velocity_dump.py`. Diff vs human pin critical window (R17 table).

Success of this step: JSON + states on disk with documented pins — **not**
hop GREEN.

### 2) Short-horizon search or closed-loop on the dump

Treat the **30–60 frame** window from wall-approach dump as the optimization
problem. Allowed tools:

- Grid / hillclimb over WJ hold/flip timings starting from R15 params
- Closed-loop: hold LEFT+A only when `pose==132` or measured wall contact;
  flip on release; then right-spin for Phase D
- Dense reward: progress toward `x≥300 ∧ y≤200` + survival in Bubble

**Exit for search:** at least one recipe with `top_reached` from the **pure
dump** (velocity-matched). Log recipe + pin in residual.

### 3) One IMPL on full pure

Wire the dump-proven recipe into mid launch **one named change** only.
Acceptance pure probe:

```bash
uv run python super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state \
  --pin-json super_metroid/debug/bubble_to_bat_pure_pin_r18.json --no-red-diag
```

Must hold R16 envelope: `launched=True` `phase_c_hit=True` min_y≤260 (prefer
≤228). Goal: `top_reached=True`.

### 4) If Phase D greens

Freeze lower work. Residual with Phase E as next card (Super door → ordinary
`0xB07A` + successor state). No continuous/STATUS in this card.

### 5) Library update

If Phase D greens: document entry conditions + failure modes in
[`BUBBLE_TECHNIQUES.md`](BUBBLE_TECHNIQUES.md) before Phase E.

## Acceptance

- [ ] Pure velocity dump(s) at fire / post_run / wall-approach with traces
- [ ] Dump-isolated recipe earns Phase D **or** honest BLOCKED with new pin
      (e.g. “pure fire arc never reaches wall-contact band even with search”)
- [ ] If dump Phase D green: full pure `top_reached=True` from CATH-04 source
- [ ] Unit green; no Phase A–C regress
- [ ] Residual with probe numbers + next card id

## Read first

- `docs/tasks/SM-K4.4-PURE-R17-residual.md`
- `docs/tasks/SM-K4.4-PHASE-LADDER.md`
- `docs/tasks/HARD_ROOM_SPLITS.md`
- `docs/tasks/BUBBLE_TECHNIQUES.md`
- `routes/kpdr/bubble_mountain_primitives.py`
- `routes/kpdr/bubble_mountain_mid.py` (launch fire branch only)

## Non-claims

- Climb-only / dump Phase D ≠ hop GREEN until full pure composes
- Human pin success is isolation only
- No continuous tip / STATUS from this card
