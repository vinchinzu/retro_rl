# Chrono Trigger

One Composer: `route.ROUTE` / `Runner.play_row`; CLI:
`QT_QPA_PLATFORM=offscreen uv run python -m snes.chrono_trigger --help`.
Route work starts with [docs/plan.md](docs/plan.md); verified progress is
[docs/STATUS.md](docs/STATUS.md). ROM gate:
`QT_QPA_PLATFORM=offscreen uv run pytest snes/chrono_trigger/tests -m rom -q`.

- `run` without `--state` is the continuous power-on route. `--state` is
  practice only. `--from ROW` loads its predecessor's pin; `--until ROW`
  includes that row. Replay the full controller tape from power-on after edits.
- Authoring: `uv run python -m snes.chrono_trigger.dev --help` lists
  scout/explore/script/keys. Use `scout PIN '[rows]' --save NAME`
  for bounded experiments; `explore` loads search states and is practice only.
- Routed rows belong in `route.py`. `go` without a goal can stop at the nearest
  reachable point; progression needs a location, story, inventory or flag goal.
  A moving blocker may need a reachable `via` point before an exit opens.
- Field DP is `$0100`: leader object index is `$0197`, not `$0097`.
  Positions are words at `$1800+slot` / `$1880+slot`, divided by 16.
  Overworld coordinates use `$0283/$0285` and an 8-pixel movement lattice.
- Read `env.data.memory.blocks[0x7E0000]`, not high `get_ram()` aliases.
- A naming screen needs START. A mashing changes the name; duplicate names
  are refused. Menu NMI recognition must settle before reading its fields.
- Control-off with no message can be a long scripted animation. Release
  the pad and wait; movement stall accounting only belongs to live control.
- Static exit graphs are geometry only. Exported tile attributes include
  half-tile heights/shapes; they are not a binary passability map.
- Guardian's story and event bit are set before combat. Require entry to the
  food locker, and verify the Guardian's zero HP in the battle trace.
- Mandatory bosses set `battle.escape: false`; healing/menu recovery can
  exceed the enemy-HP stall timer and trigger futile L+R inputs otherwise.
- Arris rat: the reverse chase uses `talk` with `chase: true`. Password
  switches use `hold: ["L", "R"]` with normal A confirmation. After the Bike
  Key, a basement round trip disperses Doan's crowd before the Enertron/exit.
- ROM/pin/media/atlas/reference clones stay in this game directory and
  ignored. Keep new route behavior in the existing row table.
