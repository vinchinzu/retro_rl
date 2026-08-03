# TASK SM-SPAZER-FOLD: Fold Spazer detour into default continuous spine

## Recipe step
compose + stabilize (planner-serial)

## Model
Planner

## Wave type
stabilize

## Own files only
- `routes/continuous.py` — default hops after Below Spazer include detour
- `routes/catalog.py` — warehouse / later tips include Spazer if chain requires
- `docs/routes/ROUTE_KPDR.md` — Spazer from optional → on-spine
- `docs/STATUS.md` — planner promote after dual re-record of affected tips
- residual required

Depends on: `SM-SPAZER-STAB` green + policy branch exists or accepted defer.

## Context
- Epic: [`SPAZER_EARLY.md`](SPAZER_EARLY.md)
- Until this card, Spazer is a **secondary tip** only. Fold means every
  continuous run that passes Below Spazer collects Spazer by default
  (100%-aligned KPDR variant).
- Re-record warehouse / kraid / later tips that share the prefix — planner
  owns integrity wave (no dual spine knobs without stabilize).

## Do
1. Insert detour hops into default post-`below_spazer` chain.
2. Re-verify pure detour still green from continuous-like source.
3. Dual continuous on first affected tip (at least warehouse or kraid prefix).
4. Update ROUTE_KPDR Spazer row: on-spine for 100% / default assisted.
5. Residual: any tip frame deltas + next policy work.

## Do not
- Skip pure re-verify
- Silent STATUS primary swap without evidence
- Mix unrelated K4 Bubble knobs in same stabilize wave

## Acceptance
- [ ] Default continuous path collects Spazer
- [ ] Dual integrity on re-recorded tip(s)
- [ ] ROUTE_KPDR + STATUS honest
