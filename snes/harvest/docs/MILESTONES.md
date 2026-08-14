# Milestones — Harvest Moon (SNES)

Short gate board. Verified evidence lives in [STATUS.md](STATUS.md).  
Structural debt: [CODE_QUALITY_REVIEW.md](CODE_QUALITY_REVIEW.md).  
Future work narrative: [plan.md](plan.md).

**Last board sync:** 2026-08-14

## Maturity snapshot

| Field | Value |
|-------|-------|
| Program maturity | **M3** (calendar multi-day); crop economy past fixture Gate A, short of continuous M4 |
| Runtime / intervention | Bronze / Clean |
| Active tip | D2 grape+shop closed (`rr-zmss`); Gate B `rr-5in` (water/income) → epic `rr-20w` |

## Gate table

| Gate | Definition of done | Primary beads | Status |
|------|--------------------|---------------|--------|
| **M3 calendar** | Spring multi-day shell, Clean overnights | historical soak | **Met** (calendar-only; money was $100) |
| **Gate A economy** | Multi-day money > $100 + harvest/establish phases | `rr-y8n`, `rr-53g` | **Closed** (Day09 successor) |
| **Empty-can natural** | Dry fixture + continuous refill without RAM poke | `rr-3q27`, thrash kids | **Mostly closed** |
| **Gate B continuous** | Power-on → Summer D1, money > 100, Clean, no mid-run load | `rr-5in` (return_home/`rr-uru1` closed); tip **CROP_WATER refill** | **Open** (21 ovn partial) |
| **Gate C calendar richness** | Festival / Sunday / rain ordering | `rr-1vc` | Open |
| **M4 natural summer** | Resume live `Y1_Summer_D1_Morning` domain work | `rr-hheu` | Open (after Gate B end-state) |
| **A4 coop skills** | Multi-adult feed/collect; skill composition | `rr-rbk` | Partial — nav skills wired (`rr-h280`); multi-adult open |
| **Crop arch tax** | `crop_planter` thin composer; refill/corridor modules | `rr-ds3` + `rr-7f54` + `rr-e6fw` | **Near bar** (~1.16k mono; typed dual-FSM; thrash data-driven; skill-composer residual) |
| **Nav / farm_ops promote** | Pathfinder + TileScanner/tools out of `farm_clearer` | `rr-fjbk`, `rr-vwrc` | **Closed** (`nav`, `farm_ops`) |
| **Day-plan soft fails** | Early ENSURE_CAN / CROP_WATER after plant | `rr-rzpd` | Open |
| **M5 domain** | Cow/barn, gifts, multi-seed, stamina closed | `rr-y80y` (mixins done), `rr-zcd3`, `rr-pzw`, `rr-buo1` | Cow mono shrink done; skill rewrite residual |
| **A6 D1 structure** | Thin handoff skills | `rr-7js5` | Open |
| **Test suite split** | day_plan sequences by domain | `rr-ufml` | **Closed** |
| **Campaign** | Multi-year / marriage / ending | deferred | — |

## Ready order (agents)

1. **`rr-rzpd`** — D2 same-day clear/hoe/plant is GREEN (`rr-20w.1` closed); sparse one-cell `CROP_WATER` still skips with `dry_crops=1`.
2. **`rr-3ae8` / `rr-5in`** — Gate B re-soak after CROP_WATER refill exhaust (power-on continuous; money>$100 to Summer).
3. `rr-rbk` / `rr-pzw` / `rr-1vc` after Gate B or in parallel if tip blocked on long soak only.
4. `rr-hheu` M4; P3 (`rr-7js5`, `rr-zcd3`, `rr-buo1`).

## Explicit non-goals (now)

- LLM plan apply polish  
- Editor file split  
- Multi-year campaign objective  

## Evidence anchors

| Claim | Evidence |
|-------|----------|
| Gate A | `recordings/run_spring_gate_a_day09.json` |
| Empty-can 3/3 | `recordings/empty_can_refill_probe.json` |
| Power-on partial Gate B | `recordings/power_on_spring_to_summer.json` (~21 ovn, money $400, terminal return_home) |
| Ship path re-verify | `recordings/rr_9xyy_ship_money_day09.json` (24 ship, $1260→$3180) |
| Power-on D2 handoff | `recordings/power_on_d1_handoff_d2.json` |
| D2 grape→shop→plant | `logs/d2_full_watch/watch_20260814_rr_20w_1_1_hoe_target.log` (`CROP_ESTABLISH` complete) |
