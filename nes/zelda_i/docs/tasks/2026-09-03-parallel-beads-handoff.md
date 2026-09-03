# 2026-09-03 parallel sub-agent bead session — handoff

Orchestrator (Sonnet) dispatched Sonnet sub-agents in parallel on independent
`bd` beads. Session ended early (credit budget). This is the handoff for
whoever resumes.

## Landed and committed this session (code + tests verified, safe to build on)

- **rr-kdul** — folded `level6/stairs3a.py` into `stairs3a_warp.py`, deleted
  dead through-names. All 3 done-when checks green (file gone, `rg` clean,
  `test_level6_stairs3a_warp.py` 6/6). **Recommend: close.**
- **rr-0240** — retargeted stale Gohma/`rr-tne2` doc pointers to the live
  `rr-8t4.2` / `l7-handoff.md` residual across `plan.md`,
  `ASSIST_CONTRACT.md`, `overnight-spine.md`, `l7-handoff.md`,
  `rr-tne2-residual.md`. Done-when checks green. **Recommend: close.**
  **Known follow-up (not done):** `nes/zelda_i/AGENTS.md` "Immediate goal"
  section (~lines 15-27) still names closed `rr-tne2` / links
  `rr-tne2-residual.md` — the bead assumed this was already correct; it
  wasn't. Small fix, one-line-scale, left for next session.
- **rr-ps7.1** — generic `RupeeFarmController` extracted to
  `nes/zelda_i/overworld/rupee_farm.py` (no `$066D` writes). `arrow_shop.py`
  refactored onto it. 17 new tests + 6 existing pass; live-ROM validated
  (no rupee poke, correct glance shape). 80R farm throughput itself stays
  slow (pre-existing, matches the bead's own text — not a regression).
  **Recommend: close.**
- **rr-ps7.2** — generic `CaveShopBuyController` in
  `nes/zelda_i/overworld/cave_shop.py` (screen, cave xy, buy xy, price,
  dialog idle, `success_getter`, optional farm). `arrow_shop.py`'s
  `OverworldToArrowShopController` is now a thin subclass. Added `candle`/
  `food` fields to `ZeldaSnapshot` in `ram.py` for `success_getter` use.
  Live-validated a real candle buy (0→1, rupee HUD drop, no controller
  poke) against the `CandleShop5E` fixture. `level8/overworld.py` got a
  `make_candle_shop_buy_controller()` sketch (buy-step only, not spliced
  into the existing hop/maze navigation — flagged as future work).
  **Recommend: close** (acceptance criteria = "one live buy", met).

## In flight when the session ended — status unknown, check before reusing

- **rr-8t4.5** (BAIT) — agent `aeac3d19a6bbe0f07`. Was told to live-recon the
  never-before-observed 0x34 Bait shop geometry (cave xy, buy xy) and, only
  if genuinely verified live, wire `NaturalBaitPurchaseController` onto the
  new `CaveShopBuyController`/`RupeeFarmController`. Explicitly told NOT to
  remove `SurvivalBaitPurchaseController` from the spine (rr-8t4.4, the
  natural L6→0x34 walk, is still open) and not to touch
  `ASSIST_CONTRACT.md`. **Check `bd show rr-8t4.5` history/notes and `git
  status` / `git diff nes/zelda_i/level7/entry.py` before trusting any
  claimed geometry — this was real, uncertain recon, not a lookup.**
- **rr-6o7.1** (L8 bush burn) — agent `a308ea821c6e25734`. **SUCCEEDED** —
  confirmed by reading `Level8EntranceReconFixture.provenance.json` directly
  (final agent report never arrived before session end, but the artifact is
  real): a 5856-trial live sweep (`nes/zelda_i/logs/level8_bush_burn_sweep.json`)
  found the burn recipe — stand `(136,93)`, face RIGHT, fire, then **continue
  RIGHT** (not UP) through the mouth transition — and it converges on live L8
  entry room **0x7E** from several stand positions. This is the first-ever
  live observation of an L8 interior room. Two things flagged by the agent,
  not yet fixed:
  1. `BurnLevel8BushController`'s `ENTER` phase in `level8/entry.py` always
     sends `UP` after the mode-16 transition — per this sweep that's wrong;
     it should continue whatever direction was actually pushed. Left
     unchanged (flagged only) since the natural-track controller shouldn't
     be edited on unreviewed recon alone.
  2. `Level8EntranceReconFixture` is a **teleport-based** fixture (poked
     straight to the burn stand from a Stage-1 fixture, not walked) — still
     correctly marked `fixture_only`/`route_eligible: false`/
     `natural_entry: false`, but confirm that before treating room 0x7E as
     more than dev-only evidence. rr-6o7.2/rr-6o7.3 can now build against a
     real interior room instead of pure hypothesis. `docs/tasks/l8-handoff.md`
     was added/updated by this agent — read it for the full writeup.
- **rr-sz8.5** (L9 overworld approach) — agent `a87e5fc212dc3439e`. Task:
  build a fixture-only OW start (TF=0xFF, Magical Sword, bombs) and run the
  Spectacle-Rock-approach → bomb → entry → Old-Man-TF-gate chain live.
  Provenance files `Level9OverworldReconFixture.provenance.json` and
  `Level9RockHopsPartialReconFixture.provenance.json` landed (the "Partial"
  name suggests it did **not** fully reach the interior — treat as
  incomplete until confirmed). `nes/zelda_i/level9/overworld.py` and
  `nes/zelda_i/level9/dungeon.py` were modified — diff them before building
  on top.

None of the three in-flight agents were told to touch `bd`, `git commit`, or
`STATUS.md` — if their work is still sitting uncommitted in the working tree
when you resume, that's expected; review and commit it deliberately rather
than assuming it's final.

## What every agent was told (ground rules, still apply)

- Never poke rupees/inventory/progression in a natural-track controller —
  only the poke-free farm/buy controllers may move state, via real contact.
- Fixture/recon work (`*ReconFixture`) must stay `route_eligible: false`,
  `fixture_only: true`, `natural_entry: false` in provenance — never
  conflated with the real natural-entry gates (`PostLevel7Handoff`,
  `PostLevel8Handoff`, `UNMEASURED_POST_L8_HANDOFF`, etc.), which must not be
  weakened. This is the "poke a save now, natural-entry-patch it later"
  pattern already established for Level 9 (`route/natural_entry.py`,
  `route/eligible.py`'s `*ReconFixture` exclusion).
- Do not touch `STATUS.md` — planner owns it.
- No `bd` status/export writes, no git commits — left for the orchestrator.

## Bd state at handoff

`rr-kdul`, `rr-0240`, `rr-ps7.1`, `rr-ps7.2` closed by the orchestrator this
session. `rr-6o7.1`, `rr-sz8.5`, `rr-8t4.5` left `in_progress` (real status
unverified — confirm before trusting). `.beads/issues.jsonl` exported and
committed alongside this file.
