# Screen / room timer (stock ROM / stable-retro)

Per-screen (overworld) and per-room (dungeon) transition timing for Zelda I
sessions driven by stable-retro. Implemented from this project's own RAM map
and emulator frame counter — not copied from external practice tools.

## What it measures

| Field | Meaning |
|-------|---------|
| `entry_frame` | First **settled play** frame at `(level, screen)` |
| `leave_frame` | First non-settled frame after dwelling (scroll starts, etc.) |
| `exit_frame` | First settled play frame at the **destination** |
| `location_frames` | `exit_frame - entry_frame` (dwell + scroll/load) |
| `dwell_frames` | `leave_frame - entry_frame` (settled time) |
| `transition_frames` | `exit_frame - leave_frame` (scroll/load) |

**Frame basis:** one `env.step` = one emulator frame (nominal 60 Hz NTSC).
This is **not** wall-clock time and **not** official IGT / lag metrics (the
stock ROM does not expose practice-hack counters this project can read).

**Settle rule:** `mode == PLAY_MODE` (5). Location identity is
`(level, screen)` from `ADDR_LEVEL` / `ADDR_SCREEN` so overworld screens and
dungeon rooms never collide. Context is `overworld` when `level == 0`, else
`dungeon`.

A hop is recorded only when settled play appears in a *new* location after a
non-settled phase (modes 6/7 scroll, 16 cave enter, 18 triforce fanfare, etc.).

## How to run

```bash
# Library: zelda_i.room_timer (no probe CLI). Opt-in hop timing on the
# Composer / Clean M5 (needs ROM + state):
uv run python nes/zelda_i/scripts/run_survival_spine.py \
  --through level2 --no-video --trials 1
uv run python zelda_i/scripts/run_level1_complete.py --room-timing --trials 1
uv run python zelda_i/scripts/run_level1_complete.py \
  --natural-entry --room-timing --trials 1
```

Artifacts: `zelda_i/recordings/room_timings/level2_prefix_*_timing.json` and
`level1_complete_*_timing.json`.

Offline JSON shape:

```json
{
  "samples": [
    {"frame": 0, "mode": 5, "level": 0, "screen": 119},
    {"frame": 10, "mode": 6, "level": 0, "screen": 119, "next_screen": 120},
    {"frame": 40, "mode": 5, "level": 0, "screen": 120}
  ]
}
```

(`screen` may be decimal; hex integers are fine in Python fixtures.)

## Library use

```python
from zelda_i.room_timer import RoomTimer, TimingSnapshot
from zelda_i.ram import read_snapshot

timer = RoomTimer()
# each emulator step:
snap = read_snapshot(env.get_ram())
visit = timer.observe(snap, frame=frame)  # LocationVisit or None
report = timer.report(source="my_run")
```

Core module: `zelda_i/room_timer.py`.

## What is ignored / abandoned

| Event | Behavior |
|-------|----------|
| Boot / title / file select (mode 0/1) | No timing until first settle |
| Soft reset → menu | Open visit abandoned (`boot_or_menu`) |
| Frame counter goes backward | Treated as load/rewind; open visit abandoned |
| Location changes while still settled (no scroll phase) | Save-state / warp jump; not a timed hop |
| Death (mode 17) | Open visit abandoned |
| Hit freeze (mode 8) | Continues dwell; does **not** start a transition |
| Cave play (mode 11) / cave enter (16) | Not timed destinations; same-screen return cancels leave |
| Session end with open location | Abandoned (`session_end`); no synthetic exit |

## Limitations

- Stock ROM only: no official IGT, lag, or lagless-frame breakdown.
- Timing is per confirmed location hop, not LiveSplit-style segment UI.
- Door identity is limited to `mode_at_leave` and `next_screen_at_leave`
  already on the snapshot (not full door-tile tables).
- Scroll/load is only complete once settle rules pass — mid-scroll frames are
  not destinations.
- Idle `session` mode only steps with no input; it will not produce hops unless
  the loaded state is mid-transition or you integrate the timer into a real
  controller/run loop.
- Controllers are unchanged by default. Opt-in live capture is available on the
  shared stage loop (`zelda_i.route.chain.run_controller_stage(..., room_timer=...)`) and the
  Level 1 complete / Level 2 prefix runners via `--room-timing`.

## Output location

Default artifacts: `zelda_i/recordings/room_timings/`
(`ROOM_TIMINGS_DIR` in `paths.py`). Keep generated JSON under `zelda_i/`,
not the repo root.
