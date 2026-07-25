# Baseline metrics — TMNT IV full hard run

Recorded from continuous power-on dry-run (low-assist).  
Source: `recordings/tmnt_iv_full_hard_dry_run.json`  
Date: 2026-07-24 (tank charge + continuous wall recovery).

## Headline

| Metric | Value |
|--------|-------|
| Power-on → credits | **01:05:41.709** |
| Credits complete frame | **236,892** |
| Total damage taken | **6,851** |
| Emergency HP heals (HP≤16→80) | **93** |
| Form-2 iframe guard frames | **3,887** |
| Life losses | **0** |
| Min HP seen | **2** |
| Lives start / peak / end | **2 / 6 / 6** |
| Hard WRAM | **2** |
| State loads / stage writes / A-special | **0** |

## vs previous baselines

| Metric | Full HP spam | Low-assist + whiplash Slash | Previous re-probe | **Tank + wall fixes** |
|--------|--------------|------------------------|----------------------------|------------------------|
| Time | 01:28:49.024 | 01:15:34.050 | 01:09:46.389 | **01:05:41.709** |
| Damage | 12,309 | 8,085 | 7,959 | **6,851** |
| HP interventions | 1,302 (→96 every hit) | 110 (≤16→80) | 108 | **93** (≤16→80) |
| I-frame guard | 5,042f | 7,467f | 4,482f | **3,887f** |
| Life losses | 0 | 0 | 0 | **0** |

Δ vs previous re-probe: **−4:04.680**, **−1,108 damage**, **−15 heals**,
and **−595 iframe-guard frames**.

## Damage by stage

| Stage byte | Name | Damage | Share | Δ vs pre-Slash | Δ vs previous run |
|------------|------|--------|-------|----------------|-----------------|
| 0 | Big Apple | 322 | 4.7% | 0 | 0 |
| 1 | Alleycat Blues | 288 | 4.2% | 0 | 0 |
| 2 | Sewer Surfin' | 466 | 6.8% | 0 | 0 |
| **3** | **Technodrome** | **1,412** | **20.6%** | −838 | **−988** |
| 4 | Prehistoric | 982 | 14.3% | **−3,810** | −56 |
| 5 | Skull & Crossbones | 970 | 14.2% | −162 | −8 |
| 6 | Wounded Knee | 916 | 13.4% | −158 | −41 |
| 7 | Neon Night Riders | 407 | 5.9% | −80 | +261 |
| 8 | Starbase | 1,088 | 15.9% | −404 | **−276** |
| 9 | Final Shell Shock | 0 | 0% | 0 | 0 (iframe guard) |

## Stage split times (power-on clock)

| Stage | Elapsed | Split Δ vs previous run |
|-------|---------|------------------------|
| Big Apple | 00:00:34.643 | same |
| Alleycat Blues | 00:05:45.681 | same |
| Sewer Surfin' | 00:10:48.915 | same |
| Technodrome | 00:15:20.551 | same start |
| Prehistoric | 00:26:36.304 | **−4:32 vs previous** |
| Skull & Crossbones | 00:33:46.646 | −4:56 |
| Wounded Knee | 00:39:28.649 | −5:07 |
| Neon Night Riders | 00:45:19.671 | −5:07 |
| Starbase | 00:49:57.830 | −3:49 (Neon variance) |
| Final Shell Shock | 00:59:16.443 | −3:55 |
| Credits complete | 01:05:41.709 | **−4:04.680** |

Technodrome segment: **11:15.754** (was 15:47.939). Prehistoric:
**7:10.341**. Starbase: **9:18.613**.

## Biggest remaining damage targets

Ranked by absolute damage still taken (best ROI for next policy work):

1. **Technodrome (1,412 / 20.6%)** — still the largest bucket, but −988
2. **Starbase (1,088 / 15.9%)** — long wave chain + Super Shredder form 1
3. **Prehistoric (982 / 14.3%)** — now below 1,000
4. **Skull & Crossbones (970 / 14.2%)** — ship + Bebop/Rocksteady
5. **Wounded Knee (916 / 13.4%)** — train + Leatherhead

## Tokka/Rahzar + tank probe

`CombatPositionStall` no longer jump-escapes during duo bosses. Emergency
heal HP≤16→80.

Pink Foot / tank throw fix (2026-07-24): pure-run charge ≥34f (no Y, no
align), then 10f toward+Y; align only on retreat/grab. Old 2f Y-tap after
early dx<16 whiffed ~75% of stun cycles (FullHardTank: 1 shredder chip /
8k f → clear).

| State | Stall-suppress | Pre-charge-fix | **Charge fix** |
|-------|----------------|----------------|----------------|
| FullHardTank | — | timeout 20k / 708 dmg / 10 heals | **9,366f / 232 dmg / 3 heals** |
| Boss4 (→stage 4) | 3,218f duo-only | 15,345f / 470 / 7 | **16,422f / 468 / 7** |
| Boss6_hp80 | **3,236f / 176 / 2** | 3,888f / 176 / 2 | (unchanged) |

Tank segment alone: **−476 dmg, −7 heals**, now clears. The first continuous
attempt then exposed a right-door duo pin (`x=224`); 37 frames of targeted
`duo_wall_escape` cleared it in the successful run. Whole Technodrome dropped
to **1,412 damage / 11:15.754**.

`probe_boss_metrics` now supports `--heal emergency|none` (default: emergency)
to match the production low-assist run.

## Slash fight known facts

- Char `0x50`, spawn HP **160**, stage byte **4**, event `0x0A`
- States: `Boss5`, `Boss5_mid`, `FullHardBoss5`, `FullHardBoss5_hp48`
- Entity status (EnemyState.animation): spin `0xEE`; punish windows often after spin settles to `0x3E`; hitstun `0x17` / multi `0x2E`
- Production policy: `SlashTactics` hybrid whiplash (approach → jump-cross → toward+Y; spin dodge)
- FullHardBoss5 probe: **13,651f / 616 dmg / 10 heals** (was ~33k / 1820 / 29)

## Regression commands

```bash
# Full continuous dry-run
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.record_full_hard_run --dry-run

# Slash-only probe
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_boss_metrics FullHardBoss5 \
  --max-frames 40000 --stop-stage-gt 4
```

## Improvement goals

1. Cut Technodrome below **1,000** (now 1,412)
2. Cut Starbase below **1,000** (now 1,088)
3. Keep Prehistoric below **1,000** (now 982)
4. Remove form-2 iframe guard without life losses
5. Do not regress continuous zero life-loss dry-run

## Slash probe progress (emergency heal HP≤16→80)

| Version | State | Frames | Damage | Heals | Outcome |
|---------|-------|--------|--------|-------|---------|
| Pre-pass thrash | FullHardBoss5 | ~33,482 | ~1,820 | ~29 | CLEAR |
| Status-wait only | FullHardBoss5 | 40,000 | 1,908 | 31 | TIMEOUT @ HP44 |
| Lab `hybrid_whiplash` | FullHardBoss5 | 18,570 | 918 | 13 | CLEAR |
| **Production whiplash** (2026-07-23) | **FullHardBoss5** | **13,651** | **616** | **10** | **CLEAR** |
| Production whiplash | Boss5 | 13,315 | 550 | 9 | CLEAR |

Δ FullHardBoss5 vs pre-pass: **−59% frames**, **−66% damage**, **−19 heals**.

Research artifacts:
- `docs/SLASH_VULN_MAP.md` — status lexicon, claw vs punish
- `scripts/probe_slash_vuln.py`
- `scripts/slash_pattern_lab.py`
- `docs/SLASH_PATTERN_LAB.md`
