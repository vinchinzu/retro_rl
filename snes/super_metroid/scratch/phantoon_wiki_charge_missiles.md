# Phantoon wiki Charge Plus Missiles (MassHesteria) — rr-7lc5

Public policy:
https://wiki.supermetroid.run/Phantoon#Any.25_KPDR_.28Charge_Plus_Missiles.29

**Not the product body.** Product is left-corner Ice/Wave/Spazer charge-only,
assist dual **20537f** ×2, 9×300 chips. Do not wire this into the spine,
STATUS, `WS_ONLY_HOPS`, or `recordings/phantoon.json`. Super default off.

## Wiki mapping

| Wiki | This module |
|------|-------------|
| Each round: 2 missiles, wait, 2 missiles, Charge Shot | `ROUND_RECIPE` + `play_round` |
| 10f missile doppler tempo | `MISSILE_SPACING = 10` |
| 300+ one barrage **or** 400+ after third → he vanishes | Gap `BARRAGE_GAP_FRAMES=24` between the two 2-missile barrages |
| 4 rounds to kill (~700/round) | Attempted; **did not land** (see below) |
| Optional 3-round Super if remaining HP ≤ 600 | `should_fire_super`; `--allow-super` default **off** |
| Super on a living Phantoon enrages (8 waves, ~18s) | Never fired Super (inventory still 5) |
| Left corner; rain: crouch + aim up | Product `_go_to_seat` / `_rain_corner_wait` |
| Skip right / bad rain | Product `charge_window_ok` (right x≥155; rain only (48,96) y 88–104) |

Pin: `scratch/post_ws_basement_to_phantoon.state` room `0xCD13` ~(39,124)
p81 gs=8, HP 2500, health 299, missiles 20, supers 5, beams `0x1007`.

Hits counted by ammo/charge actually decreasing **and** `enemy0_hp` delta,
not by pressing X. Boss bit is `$7E:D82B` bit 0 via `read_bank7e_wram`.

## Window proof (one-window first)

Seat, wait first `charge_window_ok` open (beam hold-X wait so W1 matches
the product left fig-8 at ~(120,108)), jump-fire missiles in the product
release band.

| | |
|--|--|
| opened | true (`func 0xD4A8`, eye IL `0xCC57`) |
| missiles | 20 → 19 (delta **1**) |
| HP | **2500 → 2400** (drop **100**) |
| hits counted | 1 |
| time | **1736f** / 28.886s / `00:28.93` |
| success | **true** — HP chipped before any 20k fight |

Report: `scratch/phantoon_wiki_charge_missiles_window.json`.

## Full fight vs product 20537f

Assist energy ON, ammo natural. Dual-green **27645f** ×2. Super unused.

| run | frames | seconds | clock | body 0 | boss bit | HP | gs | health | rounds | missiles | charges | supers |
|-----|-------:|--------:|------:|-------:|---------:|---:|---:|-------:|-------:|---------:|--------:|-------:|
| 1 | **27645** | 459.993 | 07:40.75 | 26617 | 27645 | 0 | 8 | 299 | 12 | 22 | 2 | 0 |
| 2 | **27645** | 459.993 | 07:40.75 | 26617 | 27645 | 0 | 8 | 299 | 12 | 22 | 2 | 0 |
| product charge-only | **20537** | 341.721 | 05:42.28 | 19507 | 20537 | 0 | 8 | 299 | 9×300 | 0 | 9 | 0 |

Delta vs 20537f: **+7108f** (~+118s). Dual identical. Deaths 0. Supers still 5.
Assist telemetry ×2: energy_restored 1480 / 73 writes, max hit 40, missile_writes 0.

## Did 4-round 2+2+charge happen?

**No.** 12 windows, not 4. Every round `missiles_b = 0` — after the 2-missile
opener Phantoon left the hittable window before barrage B or the charge.
Charges landed twice (rounds 4 and 8). Typical chip was 200 (two missiles)
then a long rain/snipe wait (`phan_farm_snipe` 13089f). Last hit was a
single missile to 0 at frame 26617; boss bit 1028f later.

Wiki 4-round would be ~700×4. This run was a 12-window missile grind plus
two charge chips, slower than sitting in the corner and dumping nine 300s.

## Recommendation

**Do not replace product 20537f. Do not wire into spine.**

Keep the module as a measured wiki experiment: the 2-missile opener from
the pin is proven (2500→2400), the full fight is dual-green HP 0 + `$D82B`,
but it is **7108f slower**, does not execute 2+2+charge, and burns more
energy assist. Product left-corner charge-only stays the body.

Super kill-gate (`HP ≤ 600`) is implemented and tested; leave it off until
a window actually needs a 600 finish.

## Files

- `snes/super_metroid/combat/phantoon_charge_missiles.py` (500 lines)
- `snes/super_metroid/scripts/probe/phantoon_charge_missiles.py` (`window` / `strategy` / `bench`)
- `snes/super_metroid/tests/test_phantoon_charge_missiles.py` (4 passed, no emulator)
- `snes/super_metroid/scratch/phantoon_wiki_charge_missiles.json`
- `snes/super_metroid/scratch/phantoon_wiki_charge_missiles_window.json`
- `snes/super_metroid/scratch/phantoon_wiki_charge_missiles_dual.json`

## Honest reds / non-claims

- Did not land wiki 4-round (12 windows, B barrage never hit)
- Did not beat 20537f (slower)
- Did not STATUS-promote, did not write `recordings/phantoon.json`
- Did not edit `combat/phantoon.py`, `routes/kpdr/k6/phantoon_fight.py`,
  STATUS, AGENTS.md, `DEFAULT_CONTINUOUS_TIP`, `WS_ONLY_HOPS`
- Did not run power-on `--to phantoon`
- Did not Super-spray / enrage
- Did not clobber `post_phantoon_poweron.state`
- Did not invent X-factor
