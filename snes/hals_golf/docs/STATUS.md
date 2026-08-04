# Status — Hal's Hole in One Golf (SNES)

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | **M2** instrumented; Bronze stroke-play clear verified |
| Best verified result | Amateur stroke-play `course_complete` total **63** (−9); VS HAL `match_won` 3-2-7 |
| Last verification | 2026-07-19 (VS HAL); 2026-07-18 (stroke clear) |
| Runtime class | Bronze |
| Intervention class | Clean |
| Integration | `HalsHoleInOne-Snes` |

## Acceptance

```bash
HEADLESS=1 PYTHONUNBUFFERED=1 ./run_bot.sh clear --state Title
# Expect: course_complete, total=63, to_par=-9, over_par=[], CourseComplete.state

HEADLESS=1 PYTHONUNBUFFERED=1 ./run_bot.sh clear --mode vs-hal --state Title
# Expect: match_won, VsHalWin.state
```

Verified clear (2026-07-18): `course_complete` in 61,485 frames, **total=63**
(−9), scorecard `[4, 3, 5, 2, 3, 3, 5, 3, 3, 4, 5, 4, 3, 3, 3, 4, 2, 4]`.

Verified VS HAL win (2026-07-19): `match_won` in 70,663 frames, record
**3-2-7**, scorecard `[3, 3, 6, 3, 4, 4, 3, 2, 2, 2, 5, 3]`.
`record_vs_hal_win.sh` appends 1,800 post-win frames and checks audio.

Metal stroke-play overlays: `hals_golf/tasks/routes/metal.py` and
[metal_stroke.md](metal_stroke.md).

## Menu flow (verified)

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
```

In-round: **SHOT**, aim, then 3× confirm on the swing meter. Club select:
`A,B-Enter`, `X,Y-Cancel`. VS HAL reuses Amateur hole plans; idle through Hal's
turns when the command panel is absent (`WAIT_OPPONENT`).

## RAM (get_ram offsets)

| Field | Offset | Notes |
|-------|--------|-------|
| Stroke count | `0x10A1` | |
| Hole index | `0x10F5` | zero-based; 18 after final hole |
| REST distance | `0x11B3` | LE yards |
| Lie/surface | `0x11CD` | 1 tee, 2 fairway, 3 bunker, 6 green |
| Aim offset | `0x10B1` | red aiming byte; **not** round total |

Round total = sum of peak per-hole stroke counts. Do **not** use
`0x7E0000 + offset` in `data.json` for this core.

## States

`Title`, `Clubs`, `Hole1_Command`, `VsHal_Hole1_Command`, `latest`, F5 QuickSave.
