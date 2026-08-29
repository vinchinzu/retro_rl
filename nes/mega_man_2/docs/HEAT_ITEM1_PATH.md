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
| Heat cam ≥8 Sniper shaft | `start=screen7` from `HeatScreen7Mid` | GREEN 3/3 (~587f, HP18, prog2048) |
| HeatScreen8 first Yoku land | `start=screen8` from `HeatScreen8` | GREEN 3/3 (~44f, sx168 sy100 HP18) |
| Heat Yoku room → cam ≥ 9 | `start=screen8` from `HeatScreen8` | GREEN 3/3 (~680f, HP18, prog2304) |

Recipes (`HeatManPolicy`):

| Start | Checkpoint | Jump recipe |
|-------|------------|-------------|
| `early` | Heat1 / HeatScreen1 | period 50 / hold 12 |
| `screen2` | HeatScreen2 | mid 60/14 until i=260, then 25/12 |
| `screen3` | HeatScreen3 | period 25 / hold 10 / phase 10 |
| `screen4` | HeatScreen4 | period 20 / hold 12 / phase 4 |
| `screen5` | HeatScreen5Ground | idle2 → j1/20 → LEFT4 → j2/24 → hop9/gw3 (A-edge) |
| `screen7` | HeatScreen7Mid | high-path: LEFT off alcove → cam6 climb → high cross past sx152 → ladder DOWN → cam8 |
| `screen8` | HeatScreen8 | wait187 → first Yoku → catch B → D → left ladder → cam≥9 |

**Screen5 notes:** mid-air `HeatScreen5` (cam-hit pin) freefalls into prog~1500 pit.
Use grounded `HeatScreen5Ground` (LEFT-land ~prog 1462). Long jumps overshoot the
sy=132 ledge; short hop hold 9 lands. Cam locks at prog 1792 (screen 7 **low
alcove** under Telly — dead-end if you only RIGHT-spam).

**Screen7 high-path (2026-08-10 breakthrough):** Low alcove wall at sx152 is solid
only for **sy≥~96** (mapset7 collision column x160–191). Human/TAS path is not a
clip: climb **left/high on cam6** (sy~68 platforms), cross **above** the column
into cam7 (land ~sx168 sy84), then drop to mapset7 ladder TSA `$30`/`$31` at
**x208–255 y192**, hold DOWN → `scroll_down` (`cam_st=$80`) → mapset8 Sniper
shaft. Cites: StrategyWiki section C (platforms → Springers → ladder down);
TMMN Heat stage (down ladder into Sniper Armor);
`/tmp/mm2-disasm/stages/heatman_wily1/` scrolling + mapset7 TSA.

Commands:

```bash
uv run python nes/mega_man_2/scripts/boot_heat_probe.py
uv run python nes/mega_man_2/scripts/run_heat_segment.py --trials 3
uv run python nes/mega_man_2/scripts/run_heat_segment.py --state HeatScreen1 --target-screen 2 --trials 3
uv run python nes/mega_man_2/scripts/run_heat_segment.py --state HeatScreen2 --target-screen 3 --trials 3
uv run python nes/mega_man_2/scripts/run_heat_segment.py --state HeatScreen3 --target-screen 4 --trials 3
uv run python nes/mega_man_2/scripts/run_heat_segment.py --state HeatScreen4 --target-screen 5 --trials 3
uv run python nes/mega_man_2/scripts/run_heat_segment.py --state HeatScreen5Ground --target-screen 7 --trials 3
uv run python nes/mega_man_2/scripts/run_heat_segment.py --state HeatScreen7Mid --target-screen 8 --trials 3
uv run python nes/mega_man_2/scripts/run_heat_segment.py --state HeatScreen8 --yoku-land --trials 3
uv run python nes/mega_man_2/scripts/run_heat_segment.py --state HeatScreen8 --target-screen 9 --trials 3
```

Evidence: `recordings/heat_segment/`, `heat_s7_seg/`, `heat_preboss/`,
`heat_s7_midpin/`, `heat_s7_route/`, `heat_s7_dual/`,
`heat_s7_climb_residual/`, `heat_s8_yoku_land/`, `heat_s8_cam9/`.
Pins (gitignored): `Heat1`, `HeatScreen1`–`HeatScreen8`, `HeatScreen5Ground`,
`HeatScreen7Mid`, `HeatScreen7HighPast`, `HeatLadder`, `HeatScrollDown`,
`HeatScreen8Yoku`.

## Residual chain (not done)

1. ~~**Heat late / pre-boss**~~ — dual-green cam ≥7 from `HeatScreen5Ground`
2. ~~**Heat s7 wall / ladder / scroll_down**~~ — dual-green cam ≥8 from `HeatScreen7Mid` (high-path)
3. ~~**HeatScreen8 first Yoku land**~~ — dual-green sy~100 from `HeatScreen8` (~44f)
4. ~~**Heat Yoku room → cam ≥ 9**~~ — wait no-ceiling, catch appearing upper B, D, left ladder scroll
5. **Heat E columns / F lava Yoku / G Sniper → boss door** — cam≥9 in; no boss_hp yet
6. **Heat boss clear + Item-1 unlock pin** — post-Heat `items\|$01` + Atomic Fire
   `weapons\|$01`
7. **Stage select → Air with Item-1** — weapons/items persist after Heat clear
8. **Air Fan → Item-1 platforms past s4** — deploy Item-1 (weapon menu + B) at
   prog ~984; clear camera ≥5 Clean
9. Optional: FCEUX human cloud-stand RAM pin (parallel residual; see below)

### HeatScreen7 high-path (2026-08-10) — dual-green cam≥8

| Fact | Detail |
|------|--------|
| Dual-green | cam ≥8 from `HeatScreen7Mid` **3/3 ~587f** HP18 prog2048 |
| Low alcove trap | sx152 wall at **sy≥96** only; `HeatScreen7`/`Mid` pins are dead-end if RIGHT-only |
| High cross | cam6 climb sy~68 → cam7 land ~sx168 sy84 (`HeatScreen7HighPast`) |
| Ladder | first grab ~sx209 sy180 ft2 (`HeatLadder`); TSA `$30`/`$31` x208–255 y192 |
| Scroll_down | `cam_st=$80`, `cam_y` rises; lands mapset8 (`HeatScreen8` sx216 sy148) |
| Scroll table | `scrolling_heatman_wily_00`: `7\|scroll_down` then `0\|scroll_down` |
| Human/TAS | StrategyWiki C + TMMN: down ladder into Sniper Armor (not wall-clip) |

**Swept (still dead — do not re-spam):** low alcove hop/UP-DOWN/damage-boost/RIGHT-only;
screen5 policy 700f on s7 floor.

### HeatScreen8 first Yoku (2026-08-10) — dual-green land

| Fact | Detail |
|------|--------|
| Dual-green | first Yoku stand **3/3 ~44f** sx168 sy100 HP18 from `HeatScreen8` |
| Recipe | (historical) LEFT10 idle1 A+LEFT14; current `screen8` waits 187 then same jump (`--yoku-land` ~f231) |
| Room | cam8 after scroll_down; RIGHT wall; ladder UP = reverse to cam7 |
| Yoku objs | types 83/84/85 @ (168,119), (168,71), (120,87), (104,55) |
| Solid window | fl `$90` (~70f) then `$A0` (~120f); stand lasts **~20f** then drop |
| Upper block | (168,71) solid while lower vanishes; jump-from-below **bonks underside** |
| Also present | Springer type70 bottom sy200; Tellies; mid platform ~sx124 sy148 |
| No Sniper yet | type78 never spawned in room traverse |
| Pin | `HeatScreen8Yoku` (unstable after ~20f — live chain, don't idle) |

**Swept (do not re-spam):** pure RIGHT from s8; climb ladder to top (scrolls back cam7);
jump straight up from first Yoku into (168,71); bottom floor full-width walk (no down exit);
left-wall climb from bottom (springer damage only).

### HeatScreen8 Yoku room → cam ≥ 9 (2026-08-29) — dual-green

| Fact | Detail |
|------|--------|
| Dual-green | cam ≥ 9 from `HeatScreen8` **3/3 ~680f** HP18 prog2304 sx40 sy7 |
| Phase | wait 187f: first+D on, upper B **off** (cycle 62f windows) |
| Catch | land first ~f231, jump up; appearing B catches sy~52 (~f258) |
| D | jump LEFT to (104,55) stand sy36 ~f305 (D vanishes ~f312) |
| Exit | hop to left-wall ledge sx~55 sy52, walk onto ladder sx40, DOWN |
| Scroll | `cam_st=$80` → mapset9 section E (`--target-screen 9`) |
| Evidence | `recordings/heat_s8_cam9/` |

**Swept (do not re-spam):** jump-from-below while B is already solid (ceiling
caps hop ~8px). Residual: E columns + F lava Yoku + G Sniper → boss door.
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
| **rr-808** | Heat mid/late toward boss | closed — dual-green through screen 5 |
| **rr-k1ea** | HeatScreen8 Sniper/Yoku → boss door | PARTIAL — Yoku room cam≥9; E/F/G + boss door residual |
| **rr-809** | Heat boss + Item-1 / Atomic Fire pin | PARTIAL — cam≥8 + first Yoku; boss/Item-1 open |
| **rr-810** | Air + Item-1 cam ≥5 | open (blocked until Item-1) |
