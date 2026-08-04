# Hal's Hole in One Golf — Agent Notes

Primary repo-wide instructions live in [../AGENTS.md](../../AGENTS.md).

## Goal

Autonomous Bronze-tier bot that:

1. Gets through title / mode / name / club menus into stroke play **or VS HAL**
2. Plays all 18 holes (Amateur stroke play first; VS HAL match vs computer Hal)
3. Manages save states (`Title`, `Clubs`, `Hole1_Command`, `VsHal_Hole1_Command`, `latest`, F5 QuickSave)
4. Survives human ↔ autoplay switching (`~` or L+R+SELECT) with menu recovery

## Commands

```bash
# Cold-boot probe (refresh debug frames + candidate states)
HEADLESS=1 ./run_bot.sh probe --frames 2000

# Human play
./run_bot.sh play --state Title
./run_bot.sh play --state Hole1_Command

# Autoplay from title (menu script + shots)
./run_bot.sh play --autoplay --state Title

# Already on the tee / command menu
./run_bot.sh play --autoplay --state Hole1_Command --skip-bootstrap

# Full-course verification; refresh Hole1_Command and latest checkpoints
HEADLESS=1 ./run_bot.sh clear --state Title

# VS HAL match (Amateur); writes VsHalWin.state on match_won
HEADLESS=1 ./run_bot.sh clear --mode vs-hal --state Title

# Same clear, but write a watchable 60fps MP4 under recordings/
HEADLESS=1 PYTHONUNBUFFERED=1 ./run_bot.sh clear --state Title --video

# Reproducible VLC-compatible VS HAL video with sound + post-win scorecard
./record_vs_hal_win.sh

# Full metal stroke-play clear video + scorecard/codec validation
./record_metal_clear.sh

# Windowed autoplay + MP4 (looks like a normal play session)
./run_bot.sh play --autoplay --state Title --video
./run_bot.sh play --autoplay --mode vs-hal --state Title

# Pro difficulty bootstrap (route overlays still Amateur until calibrated)
./run_bot.sh play --autoplay --difficulty pro --state Title

# Score hole-in-one tee neighborhood from a fixed command-menu state
HEADLESS=1 ./run_bot.sh search-hio --state Hole1_Command --max-candidates 25

# List states / tests
./run_bot.sh list
cd .. && UV_PROJECT_ENVIRONMENT=.venv PYTHONPATH=hals_golf:. \
  uv run --frozen pytest hals_golf/tests -v
```

## Menu Flow (verified)

```text
Title mode box (two columns):
  Stroke Play | VS HAL
  Match Play  | Practice
  Tournament  | Memory Shot

Stroke Play:
  Title --B--> Players --B--> Difficulty (Amateur) --START--> Name
    --START--> Clubs --DOWN/RIGHT + START--> Hole intro --B--> Command

VS HAL:
  Title --RIGHT--> VS HAL --B--> Difficulty (Amateur) --START--> Name
    --START--> Clubs --DOWN/RIGHT + START--> Hole intro --B--> Command
  (Players select is skipped)
```

In-round: select **SHOT**, aim, then 3× confirm for the swing meter
(start → power → impact). Club select legend: `A,B-Enter`, `X,Y-Cancel`.
VS HAL reuses Amateur hole plans; idle through Hal’s turns when the command
panel is absent (`WAIT_OPPONENT`).

## Human ↔ Autoplay

Shared `retro_harness.PlaySession`:

- `~` or L+R+SELECT toggles bot
- On resume: `StrokePlayMission.on_autopilot_resume` runs cancel/confirm warmup
  then restarts the current shot from the command menu
- F5 disk QuickSave; F7/F8 load

## RAM

- Stroke count `7E10A1` → `0x10A1` in `data.json` / `get_ram()`
- Hole index `0x10F5` (zero-based; becomes 18 after the final hole)
- REST distance `0x11B3` (little-endian yards)
- Lie/surface `0x11CD` (`1` tee, `2` fairway, `3` bunker, `6` green)
- Aim offset `0x10B1` (the red aiming byte; **not** the round total)
- Round total is the sum of the peak per-hole stroke counts
- Do **not** use `0x7E0000 + offset` in `data.json` for this core

## Acceptance

```bash
HEADLESS=1 PYTHONUNBUFFERED=1 ./run_bot.sh clear --state Title
# Expect: course_complete, total=63, to_par=-9, over_par=[], writes CourseComplete.state

HEADLESS=1 PYTHONUNBUFFERED=1 ./run_bot.sh clear --mode vs-hal --state Title
# Expect: match_won, writes VsHalWin.state
```

Verified clear (2026-07-18): `course_complete` in 61,485 frames, **total=63**
(-9), scorecard `[4, 3, 5, 2, 3, 3, 5, 3, 3, 4, 5, 4, 3, 3, 3, 4, 2, 4]`.
Every hole is at par or better; H3=5, H7=5, and H18=4.

Verified VS HAL win (2026-07-19): `match_won` in 70,663 frames, record
**3-2-7**, scorecard `[3, 3, 6, 3, 4, 4, 3, 2, 2, 2, 5, 3]`. Hole 1 is a
birdie (3 vs Hal's 4) for an immediate 1-up lead. `record_vs_hal_win.sh`
appends 1,800 post-win frames and verifies Theora/Vorbis plus non-silent audio.

Bronze = Amateur stroke-play clear. Silver path: beat **VS HAL** for the
metal-club password, then Pro / tournament if the harvest track asks.

Metal stroke-play overlays live in ``tasks/routes/metal.py``; calibration
memory and worst-hole priorities are in ``docs/metal_stroke.md``.

## Layout

```text
hals_golf/
├── run_bot.sh
├── custom_integrations/HalsHoleInOne-Snes/
├── hals_golf/
│   ├── core/          # RAM, scenes, actions, recovery
│   ├── tasks/
│   │   ├── mission.py       # phase machine (orchestration only)
│   │   ├── shot_policy.py   # RoutePolicy + HIO candidate expansion
│   │   ├── profile.py       # MissionProfile / difficulty helpers
│   │   ├── menus.py         # PlayMode, ClubSet, Difficulty, bootstrap
│   │   ├── shot.py          # ShotTask / PuttTask
│   │   └── routes/          # Amateur / VS HAL / Pro table data
│   └── runtime/       # CLI, video, hio_search, bootstrap
├── tests/
└── debug_frames/
```

Verified clears keep ``DeterministicRoutePolicy``. HIO exploration uses
``search-hio`` / ``HoleInOneSearchPolicy.candidates`` without changing the
mission clear path. Pro overlays in ``routes/pro.py`` stay empty until
calibrated.
