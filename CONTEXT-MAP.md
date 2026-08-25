# Context Map

This monorepo has one context per game. Program-wide vocabulary stays in
[`docs/GLOSSARY.md`](docs/GLOSSARY.md). Game glossaries are created when a
grill or design actually resolves terms.

## Contexts

- [Harvest Moon](snes/harvest/CONTEXT.md): Clean skill API that plays Harvest
  Moon (SNES) from power-on to a 10–20 hour YouTube through credits, with a
  score; first path is basic, then rewrite
- [Super Metroid](snes/super_metroid/CONTEXT.md): Survival skill API that plays
  any% KPDR (noob loadout) from power-on through credits; living tip is
  Phantoon; first credits may be a two-hour class, then rewrite; 100% is later

## Relationships

- **Harvest → program glossary**: Harvest is **Clean** (no RAM writes). M0–M8
  is a matrix label, not Harvest’s working board. Harvest **rungs** are.
- **Super Metroid → program glossary**: Super Metroid first pass is **Survival**
  (energy + unlocked ammo). M0–M8 is a matrix label, not the working board.
  The living **tip** is. Clean is a parallel track.
- **Harvest ↔ Super Metroid**: Equal token weight. Same shape (skill API,
  power-on credits, then rewrite). Harvest Finish is Clean; Super Metroid first
  pass keeps Survival because the game is harder. Solver/SMZ3 is downstream of
  vanilla credits.
- **Scratch → Tip**: Scratch duals may lead the living tip. Phantoon was
  scratch and is now the tip. Gravity is get-ahead until it is power-on on
  that tip. Rung green is power-on.
- **Scratch ending → Natural campaign**: Year 3 probes do not satisfy Harvest
  rungs. Rungs must arrive from power-on.
