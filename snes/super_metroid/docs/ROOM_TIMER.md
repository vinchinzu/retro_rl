# Room timer (stock ROM / stable-retro)

Per-room and door-transition timing for Super Metroid **vanilla** sessions
driven by stable-retro. Inspired by the usefulness of external room timers,
but implemented from this project's own RAM map and emulator frame counter.

## What it measures

| Field | Meaning |
|-------|---------|
| `entry_frame` | First **settled ordinary** frame in the room |
| `leave_frame` | First frame that left settled ordinary (door/load starts) |
| `exit_frame` | First settled ordinary frame in the **destination** room |
| `room_frames` | `exit_frame - entry_frame` (dwell + door load) |
| `dwell_frames` | `leave_frame - entry_frame` (controllable room time) |
| `transition_frames` | `exit_frame - leave_frame` (door/load) |

**Frame basis:** one `env.step` = one emulator frame (nominal 60 Hz NTSC).
This is **not** wall-clock time and **not** the practice-hack IGT / room-lag /
door-lag counters (those live in practice-ROM WRAM this project does not use).

**Settle rule:** `phase == ordinary_gameplay`, `game_state == 8`,
`door_transition == 0`, and `room_id != 0` (see `docs/ram_map.md` and
`super_metroid.ram.phase_for_game_state`).

A hop is recorded only when ordinary gameplay settles in a *new* room after a
transition phase. Source/destination room IDs, area indices, inventory masks,
and transition direction are stored for route analysis.

## How to run

```bash
# Import / logic smoke (no ROM)
uv run python snes/super_metroid/scripts/probe/room_timer.py self-check

# Offline fixture → durable JSON under super_metroid/
uv run python snes/super_metroid/scripts/probe/room_timer.py offline \
  -i path/to/samples.json \
  -o super_metroid/recordings/room_timings/offline.json

# Live idle session from a save state (needs ROM + integration)
uv run python snes/super_metroid/scripts/probe/room_timer.py session \
  --state super_metroid/custom_integrations/SuperMetroid-Snes/dev_red_tower_stable.state \
  --frames 600 \
  -o super_metroid/recordings/room_timings/session.json

# Opt-in continuous start-to-Supers baseline timing (needs ROM; no integrity change)
uv run python snes/super_metroid/scripts/record/continuous.py --to supers --no-video --room-timing
# → recordings/room_timings/supers_room_timing.json

# Rank high-dwell splits / action_reasons from an existing continuous report
uv run python snes/super_metroid/scripts/export/split_dwell.py \
  super_metroid/recordings/varia.json --top 15
uv run python snes/super_metroid/scripts/export/split_dwell.py \
  super_metroid/recordings/varia.json --reasons --top 20
```

`room_timer.split_dwells_from_report` / `rank_split_dwells` /
`action_reason_hotspots` rank tightening targets offline after a tip is
integrity-green (no emulator re-run).

Offline JSON shape:

```json
{
  "samples": [
    {"frame": 0, "room_id": 39641, "game_state": 8, "door_transition": 0},
    {"frame": 10, "room_id": 39641, "game_state": 9, "door_transition": 1},
    {"frame": 40, "room_id": 39771, "game_state": 8, "door_transition": 0}
  ]
}
```

(`room_id` may be decimal or you can write hex as integers in Python fixtures.)

## Library use

```python
from super_metroid.room_timer import RoomTimer, TimingSnapshot
from super_metroid.ram import parse_state

timer = RoomTimer()
# each emulator step:
state = parse_state(env.get_ram(), frame=frame)
visit = timer.observe(state)  # RoomVisit or None
report = timer.report(source="my_run")
```

Core module: `super_metroid/room_timer.py`.
Tests: `super_metroid/tests/test_room_timer.py` (no ROM).

## What is ignored / abandoned

| Event | Behavior |
|-------|----------|
| Boot / title / menu | No timing until first settle |
| Soft reset → menu | Open visit abandoned (`boot_or_menu`) |
| Frame counter goes backward | Treated as load/rewind; open visit abandoned |
| Room ID changes while still ordinary (no door phase) | Save-state / door-warp jump; not a timed hop |
| Death / game over | Open visit abandoned |
| Ending / credits | Open visit abandoned |
| Pause / inventory | Does not complete a hop; return to same room continues the visit |
| Session end with open room | Abandoned (`session_end`); no synthetic exit |

## Limitations

- Stock ROM only: no practice-hack gametime/realtime/lag breakdown.
- Timing is per confirmed room hop, not LiveSplit-style segment UI.
- Door identity is limited to `transition_direction` and
  `door_transition` flags already on `SuperMetroidState` (not full door BTS
  tables unless you join against `maps/` separately). Leave/entry **speed and
  position** for TAS door tech live on continuous
  `ObservedTransition.leave_kinematics` / `entry_kinematics` (see
  `door_kinematics.py`), not on the timer records.
- Multi-screen loads are only complete once settle rules pass (same discipline
  as `dev.common.door_warp`: do not treat mid-load ordinary-looking frames as
  done).
- Idle `session` mode only steps with no input; it will not produce hops unless
  the loaded state is mid-transition or you integrate the timer into a real
  controller/run loop.
- Continuous routes keep the timer **opt-in**. `RouteSession` accepts an
  optional shared `RoomTimer`; `continuous.py --to supers --room-timing`
  attaches it and writes a separate JSON under `recordings/room_timings/`.
  Timing never feeds assists, integrity evaluation, or route decisions.

## Output location

Default artifacts: `super_metroid/recordings/room_timings/`
(`ROOM_TIMINGS_DIR` in `paths.py`). Keep generated JSON under `super_metroid/`,
not the repo root.
