# Plan — Harvest Moon (SNES)

Future work only. Proven facts live in [STATUS.md](STATUS.md).

## Bottleneck

Continuous **morning → town/day work → find house → sleep → next morning** is not
yet ROM-verified end-to-end. Planner pieces exist; natural-entry chaining and
sleep reliability on remodeled houses are the gate to M3–M5.

## Next acceptance test

1. From pinned `Y1_Inside_House` (or `Y1_Front_House`), run
   `--day-plan boot_to_day2` or multi-day `--days 1`.
2. Assert calendar day advances and scene is stable morning house/farm.
3. No mid-run state load; timeout budget ~30–45 minutes wall / ~50k frames.

## Next three milestones

1. **M3 isolated day segment** — pin start/end states; one overnight succeeds
   repeatedly with hard timeout.
2. **M4 natural-entry** — same overnight from the real predecessor (post-sleep
   morning of the previous day, or power-on if boot script exists).
3. **M5 multi-day suffix** — two overnights without state load (`--days 2`).

## Active workstreams

| Stream | Focus |
|--------|--------|
| Sleep / house | `GoToSleepTask` house recovery; L2 bed routes; door hand-clear |
| Town / go-home | `town_explore` route; `READY_TO_GO_HOME` flag; menu dismiss in nav |
| Macro chain | Reuse `leave_house_to_farm`, `get_*` tools, `buy_potato_seeds`, `go_to_sleep` recordings as optional bridges |
| Domain | Coop 2-adult eggs, cow brush/milk, rainy day ordering, festivals |
| Specs | Full-run process alignment; eventual ASSIST_CONTRACT if RAM assists used |

## Deferred

- Power-on title → new game → Spring 1 (intro length / menus)
- Full multi-year campaign objective contract
- Ending credits path (see `ending_probe.py` presets)
- Local LLM plan advisor executable rewrites

## Infrastructure blockers

- ROM not in git; local path via `harvest.runtime.retro_setup`
- Long ROM-backed soaks are manual (`logs/long_runs/`), not CI
