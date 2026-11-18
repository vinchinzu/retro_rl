# Chrono Trigger autoplay

SNES US v1.0, package `chrono_trigger` under `snes/`. Controller-only
automation with read-only RAM and ROM geometry (Bronze / Clean). The route
covers the opening, the 600 AD queen rescue and Yakra, the trial/prison and
Dragon Tank, and 2300 AD through Arris Dome, Lab 32, Robo's recruitment,
the Factory and R-Series, followed by the scripted trip to the End of Time.
[Verified status and evidence](docs/STATUS.md).

From the repository root:

```bash
# Uses the existing local archive; also accepts a .sfc/.smc path.
QT_QPA_PLATFORM=offscreen uv run python -m snes.chrono_trigger setup
QT_QPA_PLATFORM=offscreen uv run python -m snes.chrono_trigger maps
QT_QPA_PLATFORM=offscreen uv run python -m snes.chrono_trigger run --output snes/chrono_trigger/recordings/route-factory
QT_QPA_PLATFORM=offscreen uv run pytest snes/chrono_trigger/tests -m rom -q
```

`run` starts from power-on with [route.py](route.py)'s `ROUTE` table;
`--route opening` selects the canyon opening. Each row has a bounded action
and a position, scene, inventory, story, or event predicate. Failure stops
with diagnostics and a `Frontier` pin. `--headed` opens the emulator window.
Pins live in `custom_integrations/ChronoTrigger-Snes/`. ROMs and generated
evidence remain game-local and ignored.

Practice uses the same Composer, [bot.py](bot.py):

```bash
# Loads ArrisOut, then plays Lab 32, Proto Dome, Factory and Robo's recovery.
QT_QPA_PLATFORM=offscreen uv run python -m snes.chrono_trigger run --from Lab32West --until RoboRecovery --output snes/chrono_trigger/recordings/factory-practice

# Override the entry pin, or use --route path/to/rows.json for a bounded experiment.
QT_QPA_PLATFORM=offscreen uv run python -m snes.chrono_trigger run --state Gate2300 --from BangorOut --output snes/chrono_trigger/recordings/future-practice
```

Every run writes `report.json` (rows, snapshots, frame counts, intervention
audit), `inputs.json` (exact controller tape), `battles.json` (live HP/MP,
commands, targets, KOs, battle phases), `atlas.html` / `atlas.json`
(observed geometry and transitions), and `final.png`.

Replay a controller tape from reset:

```bash
QT_QPA_PLATFORM=offscreen uv run python -m snes.chrono_trigger run --route snes/chrono_trigger/recordings/route-factory/inputs.json --output snes/chrono_trigger/recordings/replay-factory
```

Routed rows use `exit` (destination location; optional exit tile), `go`
(pixel target or nearest reachable position), `talk` (object or fixed
point), `shop` (buy to a target stock), and `outfit` (best offered weapon,
helmet and armor). Travel handles dialogue and fights along the way, with
MP-aware tech selection, Tonic/Revive use, focus fire and escape after enemy
HP stalls. Boss rows supply battle orders; R-Series also uses Ethers and
prioritizes middle enemy slots for area attacks. `talk` can chase a moving NPC;
`hold` keeps modifiers pressed for password switches. Explicit goals are
required when reaching a nearby point alone does not establish progress.

Lab 32 completes a race attempt, then takes the foot crossing; the route
does not claim a race win. Robo joins with Lucca while Marle stays behind.
The Factory route uses the left-side consoles and ladder, entering X-A-B-Y
at Zabie without visiting the crane rooms. Each password button is held
until its script variable acknowledges it. After R-Series, confirmation
alone carries Robo's return, repair and gate trip to the End of Time;
overworld coordinates during that sequence are not a walking entry.

The opening also uses `walk`, `touch`, `interact`, `confirm`, `battle`,
`press`, and `wait`. Its explicit `battle` rows require witnessed enemy
KOs and settled field control; a zero result byte alone cannot prove a win.
Ordinary routed travel can escape an encounter, so a boss needs its own
leave predicate and battle evidence.

Authoring tools use the same emulator and row player:

```bash
QT_QPA_PLATFORM=offscreen uv run python -m snes.chrono_trigger.dev scout ArrisOut --map
QT_QPA_PLATFORM=offscreen uv run python -m snes.chrono_trigger.dev explore PIN 0xDA
QT_QPA_PLATFORM=offscreen uv run python -m snes.chrono_trigger.dev script 0xDB
QT_QPA_PLATFORM=offscreen uv run python -m snes.chrono_trigger.dev keys PIN A:2 -:30
```

`scout` accepts optional JSON rows, `--save NAME`, and `--picks`;
`explore` searches exit hops from practice states. The optional event
decoder uses the ignored Temporal Redux reference clone. Map extraction
decodes 512 location records and 814 static exits. The collision planner
uses live field shapes, levels, slopes, NPCs, exits and overworld entrances;
refused steps are remembered and routed around. Static exit connectivity
does not encode story prerequisites. [Map details](maps/README.md).

Remaining work and the controller-only completion contract:
[docs/plan.md](docs/plan.md). RAM fields and primary source references:
[docs/ram_map.md](docs/ram_map.md).
