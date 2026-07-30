# Screen timer (stock NES ROM / stable-retro)

Per-screen map-cell timing for Metroid **vanilla** sessions driven by
stable-retro. Same *role* as Super Metroid’s room timer (confirmed hops +
emulator-frame durations), but implemented from **this** package’s RAM map and
map-cell coordinates — not a port of SM room IDs or practice-hack counters.

## What it measures

| Field | Meaning |
|-------|---------|
| `entry_frame` | First **settled** frame on the source map cell |
| `leave_frame` | First non-settled frame after dwelling (door / mode leave) |
| `exit_frame` | First settled frame on the **destination** map cell |
| `screen_frames` | `exit_frame - entry_frame` (dwell + door load) |
| `dwell_frames` | `leave_frame - entry_frame` |
| `transition_frames` | `exit_frame - leave_frame` (door/load) |

**Frame basis:** one `env.step` = one emulator frame (nominal 60 Hz NTSC).
This is **not** wall-clock time and **not** IGT/lag (NES Metroid has no
practice-hack timer fields in this stack).

**Settle rule:** `engine_mode == game`, `game_mode == playing` (3),
`paused == 0`, `in_door == 0`, `map_x` / `map_y` `< 0xF0`, and health bytes
not both zero (see `docs/ram_map.md` and `metroid.ram.MetroidSnapshot`).

**Map identity:** system RAM `map_x` (`$50`) / `map_y` (`$4F`) — the same
cells used by `brinstar.py` (`(3,14)` start, `(1,14)` morph, east probe, …).

A hop is recorded when settled play lands on a *new* map cell after:

1. a leave phase (normally `in_door != 0` door load), or
2. a **seamless** adjacent cell change while still settled (Manhattan
   distance 1) — multi-screen corridors often keep `in_door == 0`.

Non-adjacent settled jumps are treated as load/warp discontinuities, not hops.
Inventory context at the open visit (equipment, missiles, capacity, energy
tanks) is stored when available.

## How to run

```bash
# Import / logic smoke (no ROM)
uv run python metroid/scripts/probe_screen_timer.py self-check

# Offline fixture → durable JSON under metroid/
uv run python metroid/scripts/probe_screen_timer.py offline \
  -i metroid/tests/fixtures/screen_timer_sample.json \
  -o metroid/recordings/screen_timings/offline_sample.json

# Opt-in live hop timing on the first-missiles runner (does not change policy)
uv run python metroid/scripts/run_first_missiles.py --natural-entry --screen-timing
uv run python metroid/scripts/run_first_missiles.py --screen-timing  # AfterMorph diagnostic
```

Live runner artifacts (when `--screen-timing`):

- `metroid/recordings/screen_timings/first_missiles_natural_timing.json`
- `metroid/recordings/screen_timings/first_missiles_after_morph_timing.json`

Natural-entry is labeled `clean_natural_entry`. AfterMorph / Level1 are labeled
`diagnostic_state_load` and are **not** Clean natural-entry evidence.

Integration helper: `metroid/screen_timing_session.py` (passive
`observe_env` after each step; reuses `ScreenTimer`, no controller rewrite).

Offline JSON shape:

```json
{
  "samples": [
    {"frame": 0, "map_x": 3, "map_y": 14, "game_mode": 3, "in_door": 0, "health_hi": 3},
    {"frame": 10, "map_x": 3, "map_y": 14, "game_mode": 3, "in_door": 1, "health_hi": 3},
    {"frame": 40, "map_x": 2, "map_y": 14, "game_mode": 3, "in_door": 0, "health_hi": 3}
  ]
}
```

(`map_cell: [x, y]` is also accepted.)

## Library use

```python
from metroid.screen_timer import ScreenTimer, TimingSnapshot
from metroid.ram import read_snapshot

timer = ScreenTimer()
# each emulator step:
snap = read_snapshot(env.get_ram(), env=env)
visit = timer.observe(snap, frame=frame)  # ScreenVisit or None
report = timer.report(source="my_run")
```

Core module: `metroid/screen_timer.py`.
Tests: `metroid/tests/test_screen_timer.py` (no ROM).

Does **not** alter morph/first-missiles controllers or route graphs. Wire
`observe()` from your own probe/run loop when you want live hop timing.

## What is ignored / abandoned

| Event | Behavior |
|-------|----------|
| Boot / title / password | No timing until first settle |
| Soft reset → title | Open visit abandoned (`boot_or_menu`) |
| Frame counter goes backward | Treated as load/rewind; open visit abandoned |
| Adjacent map cell change while settled | Timed seamless hop (`transition_frames == 0`) |
| Non-adjacent map cell change while settled | Save-state / warp jump; not a timed hop |
| Zero energy under game engine | Open visit abandoned (`death_or_reset`) |
| Item fanfare (`game_mode == 9`) / pause | Does not complete a hop; return to same cell continues the visit |
| Session end with open screen | Abandoned (`session_end`); no synthetic exit |

## Limitations

- Emulator frames only — no IGT, lag frames, or wall-clock.
- Screen identity is map cell only; multi-screen “rooms” in lore terms may
  span several cells (each cell is timed separately).
- Door identity is limited to `in_door` at leave (no door BTS table).
- `area` (`$74`) may be 0 early before Brinstar latches `$10`.
- Inventory fields need WRAM via `read_snapshot(..., env=env)`; system-RAM-only
  samples leave equipment/missiles at 0.
- Segment runners leave timing **off by default**; opt in with
  `--screen-timing` on `run_first_missiles.py` (passive observer only).
- Offline probe does not boot the ROM; live timing uses the segment runner.

## Output location

Default artifacts: `metroid/recordings/screen_timings/`
(`SCREEN_TIMINGS_DIR` in `paths.py`). Keep generated JSON under `metroid/`,
not the repo root.
