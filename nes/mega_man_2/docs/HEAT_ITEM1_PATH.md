# Heat → Air Item-1 path (rr-f3nr scaffold)

Alternate past Air screen-4 without Thunder Chariot stand. TAS / RTA use
**Heat-before-Air** so Item-1 platforms bridge the LL sky gap.

## Why this path

| Fact | Detail |
|------|--------|
| Air-first cloud stand | RED under fceumm after Clean LL kill (`CLOUD_LAND_RED_PIN.md`) |
| Jump skip | Gap ~296px after prog 984; max ~1086 — impossible |
| Item-1 | Unlocked on **Heat Man clear** (`$009B` bit `$01`) |
| AirFanPlatform | `weapons=$00`, `items=$00` — no Item-1 on Air-first isolate |
| Atomic Fire | `$009A` bit `$01` also from Heat clear |

## Stage-select map (`$002A`)

```
1 Bubble   2 Air     3 Quick
8 Heat     0 Wily    4 Wood
7 Metal    6 Flash   5 Crash
```

Password → robot select lands on **Wily (0)**. `LEFT` → Heat (8). `UP` → Air (2).

## Dual-green (2026-08-10)

| Milestone | State / script | Result |
|-----------|----------------|--------|
| Heat entry | `boot_to_heat_man_script` → `Heat1` | GREEN (~controllable ~f926+) |
| Heat screen ≥1 | `HeatManPolicy` early from `Heat1` | GREEN (~244f, HP24, prog256) |
| Heat screen ≥2 | early from `HeatScreen1` | GREEN 3/3 (~194f, HP24, prog512) |
| Heat screen ≥3 | `start=screen2` from `HeatScreen2` | GREEN 3/3 (~302f cam, ~351f gnd) |
| Heat screen ≥4 | `start=screen3` pillars from `HeatScreen3` | GREEN 3/3 (~161f cam, ~181f gnd) |
| Heat screen ≥5 | `start=screen4` from `HeatScreen4` | GREEN 3/3 (~131f cam, ~320f gnd) |
| Heat screen ≥7 pre-boss | `start=screen5` from `HeatScreen5Ground` | GREEN 3/3 (~293f, HP22, prog1792) |

Recipes (`HeatManPolicy`):

| Start | Checkpoint | Jump recipe |
|-------|------------|-------------|
| `early` | Heat1 / HeatScreen1 | period 50 / hold 12 |
| `screen2` | HeatScreen2 | mid 60/14 until i=260, then 25/12 |
| `screen3` | HeatScreen3 | period 25 / hold 10 / phase 10 |
| `screen4` | HeatScreen4 | period 20 / hold 12 / phase 4 |
| `screen5` | HeatScreen5Ground | idle2 → j1/20 → LEFT4 → j2/24 → hop9/gw3 (A-edge) |

**Screen5 notes:** mid-air `HeatScreen5` (cam-hit pin) freefalls into prog~1500 pit.
Use grounded `HeatScreen5Ground` (LEFT-land ~prog 1462). Long jumps overshoot the
sy=132 ledge; short hop hold 9 lands. Cam locks at prog 1792 (screen 7 wall /
pre-boss alcove under low ceiling).

Commands:

```bash
uv run python nes/mega_man_2/scripts/boot_heat_probe.py
uv run python nes/mega_man_2/scripts/run_heat_segment.py --trials 3
uv run python nes/mega_man_2/scripts/run_heat_segment.py --state HeatScreen1 --target-screen 2 --trials 3
uv run python nes/mega_man_2/scripts/run_heat_segment.py --state HeatScreen2 --target-screen 3 --trials 3
uv run python nes/mega_man_2/scripts/run_heat_segment.py --state HeatScreen3 --target-screen 4 --trials 3
uv run python nes/mega_man_2/scripts/run_heat_segment.py --state HeatScreen4 --target-screen 5 --trials 3
uv run python nes/mega_man_2/scripts/run_heat_segment.py --state HeatScreen5Ground --target-screen 7 --trials 3
# s7 climb residual (wall lock) — no target-screen past 7 yet
```

Evidence: `recordings/heat_boot/`, `recordings/heat_segment/`, `heat_s7_seg/`,
`heat_preboss/`, `heat_s7_midpin/`, `heat_s7_climb_residual/`.
Pins (gitignored): `Heat1`, `HeatScreen1`–`HeatScreen7`, `HeatScreen5Ground`,
`HeatScreen7Mid`.

## Residual chain (not done)

1. ~~**Heat late / pre-boss**~~ — dual-green cam ≥7 from `HeatScreen5Ground` (rr-809 PARTIAL)
2. **Heat boss door climb** — s7 alcove wall-lock (see below); no boss_hp / Item-1 yet
3. **Heat boss clear + Item-1 unlock pin** — post-Heat `items\|$01` + Atomic Fire
   `weapons\|$01`
4. **Stage select → Air with Item-1** — weapons/items persist after Heat clear
5. **Air Fan → Item-1 platforms past s4** — deploy Item-1 (weapon menu + B) at
   prog ~984; clear camera ≥5 Clean
6. Optional: FCEUX human cloud-stand RAM pin (parallel residual; see below)

### HeatScreen7 climb residual (2026-08-10)

Verified dual-green s5→s7 3/3. Pins: `HeatScreen7`, `HeatScreen7Mid`
(sx152 sy124 under Telly). Evidence: `recordings/heat_s7_seg/`,
`heat_s7_midpin/`, `heat_s7_climb_residual/residual.json`.

| Fact | Detail |
|------|--------|
| Cam lock | prog **1792** (cam7 cam_x=0) |
| Playable floor | sy148 sx **130–152** (white platform) |
| Micro-ledge | sy**124** sx **140–152** under Telly (type36) |
| Hard wall | sx **152** full jump-arc height — no damage-boost past |
| Mapset7 ladder | TSA type2 tiles `$30`/`$31` at **x192–255 y192** (scroll_down exit) — unreachable past wall |
| Scroll table | `scrolling_heatman_wily_00`: 7 right rooms then **scroll_down** → mapset8+ Sniper Armor shaft |
| Stage map | Upper corridor ends at Telly ledge; multi-level + boss are **below/right** via vertical transition |
| Telly | Chips HP while camping mid (~−4/60f); kills ~650f if only hopping |

**Swept:** hop→mid grids; LEFT high (scrolls cam6); UP/DOWN never `tile_feet==2`;
screen5 policy 700f on s7; random s5/s6/s7 2–3k frames 0 ladders; RIGHT wall
probe + damage boost.

**Next tip:** reach ladder past sx152 (clip/route?) or re-enter vertical shaft from
an earlier drop; then Sniper Armor room → boss door → clear → Item-1 pin.
Do not tip rr-810 until Item-1.

## FCEUX / human cloud-stand pin protocol

External (needs FCEUX + human or Lua). Use when cloud path is revisited.

**Goal:** On one frame where feet **stick** on empty Thunder Chariot (`0x3E`
after `0x3D` dead), dump vs freefall:

| Field | Addr / note |
|-------|-------------|
| player sy / sx | `$04A0` / `$0460` |
| body by / bx / scr | slot of type `$3E` |
| status | `$002C` |
| body flag | `$0420+i` (appear `$10`?) |
| tsa | `$04E0+i` |
| xs, ys, xsf, ysf | `$0600/$0640` + frac |
| cam screen / x | `$0020` / `$001F` |
| tile_feet | `$0032` |

**Steps:**

1. Load Air Man to fan platform (or FCEUX savestate at LL section).
2. Kill rider with pulsed B; land / ride empty cloud if human can.
3. Freeze frame on first Y-lock ≥4f (or first `tile_feet`/status change).
4. Hex dump listed fields; compare to freefall dumps in
   `recordings/air_post4_cloud_solid/` and `air_post4_altpath/`.
5. If a missing arm bit/type is found, implement Clean under fceumm.

**Do not** re-grid goblin solid, feet_dy alone, screen-align alone, or
zero-mask global solid (already proven).

## Beads

Parent residual: **rr-f3nr** (scaffold PARTIAL).

| Bead | Role | Status |
|------|------|--------|
| **rr-808** | Heat mid/late toward boss | PARTIAL — dual-green through screen 5 |
| **rr-809** | Heat boss + Item-1 / Atomic Fire pin | PARTIAL — dual-green cam ≥7 pre-boss; boss/Item-1 open |
| **rr-810** | Air + Item-1 cam ≥5 | open (blocked until Item-1) |
