# TASK SM-100-TRACK: 100% run board scaffold (items / maps / bosses)

## Recipe step
docs

## Model
Flash

## Wave type
implement

## Own files only
- `docs/routes/TRACK_100.md` (**create**) — item + map + boss checklist
- `docs/routes/TRACK_100.csv` (**create**, optional machine source)
- `docs/README.md` — index row
- `docs/tasks/QUEUE.md` — pointer under parallel tracks (if not already)
- residual optional

Gates feature work: early Spazer epic is first concrete 100% item; board
lists remaining major collectibles so fold/planning stays honest.

## Context
- Continuous product today: assisted **any% KPDR** full clear (M8 credits).
- User goal: eventually **100%** (all items / maps as project defines), with
  early Spazer as an inserted collect that changes later policies.
- Do **not** invent a second continuous integrity contract yet — board only.
- Early Spazer epic: [`SPAZER_EARLY.md`](SPAZER_EARLY.md)

## Read first
- `docs/routes/MILESTONES.md`
- `docs/routes/ROUTE_KPDR.md` optional table
- `docs/routes/BACKLOG.csv` OPTIONAL epic rows
- `docs/BENCHMARK_SPEC.md` if completion classes mentioned

## Do
1. Create `TRACK_100.md` with sections:
   - Definition (project 100%: major items + maps + bosses + escape; note
     any% vs 100% divergence)
   - Item checklist table (Morph, Bombs, Charge, Spazer, Varia, HJ, Speed,
     Wave, Ice, Grapple?, Plasma, Gravity, SJ, Spring?, PB packs, E-Tanks,
     missiles/supers packs — mark continuous status: done / open / optional)
   - Map stations
   - Boss order (KPDR + any 100% extras)
   - Insertion notes: Spazer early (this epic); Charge return; Pink PB parked
   - Continuous fold policy: optional tips → on-spine when pure+stab green
2. Mark Spazer as **in progress** via SPAZER_EARLY ladder.
3. Link from docs README + QUEUE parallel tracks.
4. No STATUS primary tip changes.

## Do not
- Claim 100% continuous exists
- Expand BACKLOG to hundreds of pack rooms in this card (list majors only;
  residual may say “pack depth later”)
- Block K4 spine language

## Acceptance
- [ ] `TRACK_100.md` exists with Spazer row → SPAZER_EARLY
- [ ] Indexed from docs README
- [ ] Explicit: any% KPDR remains primary until planner fold decisions

## Verify commands
```bash
test -f super_metroid/docs/routes/TRACK_100.md
rg -n "Spazer|SPAZER_EARLY|100%" super_metroid/docs/routes/TRACK_100.md
```
