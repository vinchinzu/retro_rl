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

Recipes (`HeatManPolicy`):

| Start | Checkpoint | Jump recipe |
|-------|------------|-------------|
| `early` | Heat1 / HeatScreen1 | period 50 / hold 12 |
| `screen2` | HeatScreen2 | mid 60/14 until i=260, then 25/12 |
| `screen3` | HeatScreen3 | period 25 / hold 10 / phase 10 |
| `screen4` | HeatScreen4 | period 20 / hold 12 / phase 4 |

Commands:

```bash
uv run python nes/mega_man_2/scripts/boot_heat_probe.py
uv run python nes/mega_man_2/scripts/run_heat_segment.py --trials 3
uv run python nes/mega_man_2/scripts/run_heat_segment.py --state HeatScreen1 --target-screen 2 --trials 3
uv run python nes/mega_man_2/scripts/run_heat_segment.py --state HeatScreen2 --target-screen 3 --trials 3
uv run python nes/mega_man_2/scripts/run_heat_segment.py --state HeatScreen3 --target-screen 4 --trials 3
uv run python nes/mega_man_2/scripts/run_heat_segment.py --state HeatScreen4 --target-screen 5 --trials 3
```

Evidence: `recordings/heat_boot/`, `recordings/heat_segment/`, `heat_mid_*`, `heat_pillars/`, `heat_s5/`.
Pins (gitignored): `Heat1`, `HeatScreen1`–`HeatScreen5`.

## Residual chain (not done)

1. **Heat late / pre-boss** — past screen 5 (prog~1500 death band), ladders / vertical
2. **Heat boss** — isolated door entry; Bubble Lead preferred (not required Clean)
3. **Item-1 unlock pin** — post-Heat `items\|$01` and Atomic Fire `weapons\|$01`
4. **Stage select → Air with Item-1** — weapons/items persist after Heat clear
5. **Air Fan → Item-1 platforms past s4** — deploy Item-1 (weapon menu + B) at
   prog ~984; clear camera ≥5 Clean
6. Optional: FCEUX human cloud-stand RAM pin (parallel residual; see below)

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
| **rr-808** | Heat mid/late toward boss | PARTIAL — dual-green through screen 5; pre-boss open |
| **rr-809** | Heat boss + Item-1 / Atomic Fire pin | open (blocked by late Heat) |
| **rr-810** | Air + Item-1 cam ≥5 | open |
