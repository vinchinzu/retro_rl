# Status — Super Double Dragon

| Field | Value |
|-------|-------|
| Ladder rank | 4 |
| Tier | 1 |
| Status | **Missions 1–2 complete; M3/M5 transition work** |
| Integration | `SuperDoubleDragon-Snes` |
| ROM | USA ROM installed and boot-verified |

## Proven

- Boot from `NONE` reaches fight-ready Mission 1 at frame **1570** and saves
  `Stage1`.
- The first Mission 1 combat lock clears without B/block. Latest canonical
  run: **4953 frames**, HP **87 -> 38**, enemies **1 -> 0**; report at
  `recordings/stage1_segment/stage1_segment.json`.
- The parser follows the engine's P1 actor-page pointer (`0x1CF9`). This fixes
  the elevator, where Billy moves from page `0x0C00` to `0x1100` and changes
  actor kind.
- Mission 1 areas `0x10`–`0x13` clear into a natural `Stage2` checkpoint.
  Area `0x13` scrolls left after the elevator.
- Mission 2 areas `0x14`–`0x16` clear into a natural `Stage3` checkpoint.
  `0x15` needs long spiral-walkway holds; `0x16` needs `DOWN+RIGHT` runway
  progression and a wider attack band.
- Mission 3 areas `0x17` and `0x18` advance naturally. Area `0x19` combat is
  cleared with a low-HP block-counter finisher.
- A development transition clone (`Area1A`) loads the next gym fight. The
  natural `0x19 -> 0x1A` stair handoff remains open.
- `Stage4` is a documented transition clone made from the fight-ready Stage 3
  transition with area byte `0x1C`. It loads correctly, and Mission 4 then
  clears naturally into `Stage5` in **21273 frames**.
- Mission 5's first `0x1D` combat group clears (`Stage5_FirstClear`). Further
  wave activation is still open.

Development heals are explicit: area-completion probes restore the current
actor page to HP 88 below HP 32 and set lives to 2. The reusable runner records
the heal count in its JSON report.

## Preferred resumes

| State | Notes |
|-------|-------|
| `Stage1` | fresh fight-ready boot checkpoint |
| `Stage1_FirstSegment_Clear` | canonical first no-block lock clear |
| `Stage1_Elevator` | Mission 1 elevator transition (`0x12`) |
| `Stage1_Area3` | final Mission 1 area transition (`0x13`) |
| `Stage2` | natural Mission 2 start (`0x14`) |
| `Area15_Clear` | airport spiral passage after its first locks |
| `Area16_Clear` | runway between wave groups |
| `Stage3` | natural Mission 3 start (`0x17`) |
| `Area18`, `Area19` | natural Chinatown area transitions |
| `Area19_Clear` | gym fight clear; natural staircase handoff open |
| `Area1A` | development transition clone for the next gym fight |
| `Stage4` | documented transition clone; correct Mission 4 scene |
| `Stage5` | natural Mission 5 start from the Mission 4 clear |
| `Stage5_FirstClear` | first Mission 5 combat group clear |

## Run

```bash
uv run --frozen python super_double_dragon/scripts/boot_probe.py
uv run --frozen python super_double_dragon/scripts/run_stage1_segment.py
uv run --frozen python super_double_dragon/scripts/run_area.py \
  --state Stage1_Elevator --target-area 0x13 --save-state Stage1_Area3
```

Use `--no-dev-heal` for a natural-health evaluation.

## Open

- Natural Mission 3 stair handoff (`0x19 -> 0x1A`), later gym floor, and Chin
  brothers (`0x1B`)
- Mission 5 remaining `0x1D` waves and `0x1E`
- Missions 6–7 and ending
- One continuous title-to-ending evaluation
