# Status

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M3 |
| Best verified result | Air camera >= 4 from AirScreen2 (documented 3/3). Heat camera >= 9 from HeatScreen8 (3/3 in the cam9 report). |
| Last verification | 2026-08-10 |
| Runtime class | Bronze |
| Intervention class | Clean |

Maturity stays M3. Camera 9 is an isolated Heat segment from `HeatScreen8`,
not a stage clear and not natural entry. The cam9 JSON has no date. A path
note dated that run 2026-08-29. That date is not the gate date.

## Air Man

Documented in the previous status. Not re-run here.

- Camera >= 2 from `Level1`: about 522 frames, HP 22, 3/3.
- Camera >= 2 from `AirLanded`: about 226 frames, 3/3.
- Camera >= 3 from `AirScreen2`: 241 frames, 3/3.
- Camera >= 4 from `AirScreen2`: 502 frames, HP 16, progress 1024, 3/3.

Post-screen-4 cloud stand is not cleared. No camera >= 5. A pulsed-B rider
kill is documented. The open gap after progress 984 is about 296 pixels.
Item-1 is the remaining Clean way past it. Evidence directory:
`recordings/air_segment/`.

## Heat Man

File-backed:

- First Yoku stand from `HeatScreen8`:
  `recordings/heat_s8_yoku_land/heat_s8_yoku_land.json`, 3/3, 44 frames,
  sx 168, sy 100, HP 18.
- Camera >= 9 from `HeatScreen8`:
  `recordings/heat_s8_cam9/heat_segment.json`, 3/3, 680 frames, HP 18,
  progress 2304, sx 40, sy 7, weapons 0, items 0.

Documented earlier segments, not re-opened here: Heat1 through camera 5;
camera >= 7 from `HeatScreen5Ground` (about 293 frames, 3/3); camera >= 8
from `HeatScreen7Mid` (about 587 frames, HP 18, 3/3).

## Not done

- Heat sections E, F, and G, the boss door, the boss, and Item-1.
- Air with Item-1 past camera 5.
- Natural-entry M4 from power-on.
- A full robot-master clear.

## Next

`docs/plan.md`.
