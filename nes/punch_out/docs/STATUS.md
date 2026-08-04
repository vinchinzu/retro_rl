# Status — Mike Tyson's Punch-Out!! (NES)

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M2 |
| Best verified result | Glass Joe first knockdown from Match1 (taunt counter ~0:42) |
| Last verification | 2026-07-28 |
| Runtime class | Bronze |
| Intervention class | Clean |

| Field | Value |
|-------|-------|
| Status | **instrumented + first KD** |
| Integration | `PunchOut-Nes` |
| ROM zip | `roms/Nintendo/NES/Mike Tyson's Punch-Out!!.zip` |
| Checkpoints | `Level1.state` (ring entry), `Match1.state` (clock live) |
| Evidence | [glass_joe/](../recordings/glass_joe/), [match1.png](../recordings/match1.png) |

## Done

- Directory layout and NES integration stubs
- Deterministic reset → first controllable play (`scripts/boot_probe.py`)
- M2 RAM: health, hearts, stars, clock, round, fight flag, opp pattern/timer/action, taunt window
- `Match1` save at bout clock start
- Glass Joe policy: taunt counter → KD1; get-up mash; dodge survival; between-round advance
- Segment runner `scripts/run_glass_joe.py --goal knockdown`

## Not done

- Full bout win (3 opp KDs / KO / decision) — often reach KD2 then lose on Mac TKO in R2
- Natural-entry chain from power-on through bout clear
- Continuous multi-opponent circuit

## Next

Finish Glass Joe bout (third knockdown or decision) with post-KD2 offense that lands damage while `opp_health ≤ 48`.
