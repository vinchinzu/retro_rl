# Composer required

Every game has one production tick (**Composer**) that dispatches **Skills**
from a table of rows. New behavior is a row or a **Tactic**. Soft max ~1000
LOC: merge into the owner or delete; a sibling extract to beat the bar is a
violation. This is how TMNT IV shipped M8 at 12k LOC with zero files over 1k,
and how Super Metroid retired `start_to_*.py` via `TipSpec`.

**Considered:** nest-vs-flat as the standard (Harvest packages vs Super
Metroid folders) — rejected; folders are cheap, the tick is the interface.
Split-before-500 — rejected; that produced Harvest’s mixin clusters.
Per-game line bars — rejected; one review file (`CODING_STANDARDS.md`).
