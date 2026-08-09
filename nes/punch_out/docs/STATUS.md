# Status — Mike Tyson's Punch-Out!! (NES)

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M3 |
| Best verified result | Glass Joe bout win from Match1 (KO after 2 opp KDs; 3/3) |
| Last verification | 2026-08-07 |
| Runtime class | Bronze |
| Intervention class | Clean |

| Field | Value |
|-------|-------|
| Status | **isolated bout win** |
| Integration | `PunchOut-Nes` |
| ROM zip | `roms/Nintendo/NES/Mike Tyson's Punch-Out!!.zip` |
| Checkpoints | `Level1.state` (ring entry), `Match1.state` (clock live), `GlassJoe_Clear.state` |
| Evidence | [glass_joe/](../recordings/glass_joe/) (`summary.json`, `trial_01/bout.mp4`) |

## Done

- Directory layout and NES integration stubs
- Deterministic reset → first controllable play (`scripts/boot_probe.py`)
- M2 RAM: health, hearts, stars, clock, round, fight flag, opp pattern/timer/action, taunt window
- `Match1` save at bout clock start
- Glass Joe policy: taunt counter → KD; timed post-attack dodge (wait ~32f + 5f pulse); get-up mash; between-round advance
- Segment runner `scripts/run_glass_joe.py --goal win --trials 3`
- **M3 verified**: 3/3 `ko_win` from Match1, hard timeout 30k frames, Clean Bronze (no mid-run RAM writes / state loads). Frames 6762, mac HP 96, mac_kd 0, opp_kd 2 (count-out on second knockdown).

## Not done

- Natural-entry chain from power-on / Level1 through bout clear (M4)
- Continuous multi-opponent circuit (Von Kaiser+)

## Next

M4 natural-entry: same Glass Joe win from power-on or Level1 predecessor (not only Match1 load).
