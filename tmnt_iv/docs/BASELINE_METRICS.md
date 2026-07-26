# Baseline metrics — TMNT IV full hard run

Recorded from continuous power-on dry-run (low-assist).  
Source: `recordings/tmnt_iv_full_hard_dry_run.json`  
Date: 2026-07-25 (Stage 2–3 pass + Big Apple pizza-scope fix).

## Headline

| Metric | Value |
|--------|-------|
| Power-on → credits | **01:04:07.131** |
| Credits complete frame | **231,208** |
| Total damage taken | **6,869** |
| Emergency HP heals (HP≤16→80) | **91** |
| Form-2 iframe guard frames | **3,824** |
| Life losses | **0** |
| Min HP seen | **2** |
| Lives start / peak / end | **2 / 6 / 6** |
| Hard WRAM | **2** |
| State loads / stage writes / A-special | **0** |

## vs previous baselines

| Metric | Full HP spam | Low-assist + whiplash Slash | Previous re-probe | Tank + wall fixes | **Stage 2–3 pass** |
|--------|--------------|-----------------------------|-------------------|-------------------|--------------------|
| Time | 01:28:49.024 | 01:15:34.050 | 01:09:46.389 | 01:05:41.709 | **01:04:07.131** |
| Damage | 12,309 | 8,085 | 7,959 | 6,851 | **6,869** |
| HP interventions | 1,302 (→96 every hit) | 110 (≤16→80) | 108 | 93 | **91** (≤16→80) |
| I-frame guard | 5,042f | 7,467f | 4,482f | 3,887f | **3,824f** |
| Life losses | 0 | 0 | 0 | 0 | **0** |

Δ vs the tank + wall baseline: **−1:34.578**, **−5,684 frames**,
**+18 damage**, **−2 heals**, and **−63 iframe-guard frames**. Zero life
losses held.

## Damage by stage

| Stage byte | Name | Damage | Share | Δ vs pre-Slash | Δ vs tank + wall |
|------------|------|--------|-------|----------------|------------------|
| 0 | Big Apple | 324 | 4.7% | +2 | +2 |
| 1 | Alleycat Blues | 346 | 5.0% | +58 | +58 |
| **2** | **Sewer Surfin'** | **240** | **3.5%** | **−226** | **−226** |
| **3** | **Technodrome** | **1,262** | **18.4%** | **−988** | **−150** |
| 4 | Prehistoric | 1,144 | 16.7% | −3,648 | +162 |
| **5** | **Skull & Crossbones** | **760** | **11.1%** | **−372** | **−210** |
| 6 | Wounded Knee | 1,159 | 16.9% | +85 | +243 |
| 7 | Neon Night Riders | 418 | 6.1% | −69 | +11 |
| 8 | Starbase | 1,216 | 17.7% | −276 | +128 |
| 9 | Final Shell Shock | 0 | 0% | 0 | 0 (iframe guard) |

## Stage split times (power-on clock)

| Stage | Elapsed | Split Δ vs tank + wall |
|-------|---------|------------------------|
| Big Apple | 00:00:32.896 | −0:01.747 |
| Alleycat Blues | 00:05:53.884 | +0:08.203 |
| Sewer Surfin' | 00:11:10.047 | +0:21.132 |
| Technodrome | 00:14:18.935 | **−1:01.616** |
| Prehistoric | 00:23:05.901 | **−3:30.403** |
| Skull & Crossbones | 00:30:49.255 | −2:57.391 |
| Wounded Knee | 00:36:44.686 | −2:43.963 |
| Neon Night Riders | 00:43:31.133 | −1:48.538 |
| Starbase | 00:48:18.543 | −1:39.287 |
| Final Shell Shock | 00:57:42.398 | −1:34.045 |
| Credits complete | **01:04:07.131** | **−1:34.578** |

Largest segment time gains: Sewer Surfin' **3:08.889** (**−1:22.747**)
and Technodrome **8:46.965** (**−2:28.788**). Later-route variance returned
some of that gain: Prehistoric was **7:43.354** (+0:33.013), Wounded Knee
**6:46.447** (+0:55.425), and Starbase **9:23.855** (+0:05.242).

## Stage 1 Clean-track probes (2026-07-25)

Not a whole-run baseline replacement. Segment evidence after pizza seek
(`0x30`) + Baxter left-lane + post-pickup disengage. Source:
`recordings/stage1_clean_track/stage1_probes.json` (3/3 stable).

| Probe | Heal | Outcome | Frames | Damage | E-heals | Notes |
|-------|------|---------|--------|--------|---------|-------|
| Continuous Big Apple (current) | emergency | clear | 19,291 | **324** | in whole-run 91 | power-on context |
| `Stage1` segment | **none** | **stage_advance** | **14,921** | **130** | **0** | Clean Stage 1; entry HP 76 |
| `Stage1` segment | emergency | stage_advance | 14,921 | 130 | 1 | same route |
| `Boss` (Baxter) | none | stage_advance | 4,293 | **64** | **0** | Clean Baxter |

Publication target remains **Bronze / Clean** (maturity stays M8).

## Biggest remaining damage targets

Ranked by absolute damage still taken (best ROI for next policy work):

1. **Technodrome (1,262 / 18.4%)** — still the largest bucket
2. **Starbase (1,216 / 17.7%)** — long wave chain + Super Shredder form 1
3. **Wounded Knee (1,159 / 16.9%)** — train + Leatherhead variance
4. **Prehistoric (1,144 / 16.7%)** — regressed above 1,000
5. **Skull & Crossbones (760 / 11.1%)** — now below 1,000
6. **Big Apple (324 → segment Clean 130)** — checkpoint gain did not transfer

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
`duo_wall_escape` cleared it in that run. The current whole-run Technodrome
bucket is **1,262 damage / 8:46.965**.

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

1. Cut Technodrome below **1,000** (now 1,262)
2. Cut Starbase below **1,000** (now 1,216)
3. Return Prehistoric below **1,000** (now 1,144)
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
