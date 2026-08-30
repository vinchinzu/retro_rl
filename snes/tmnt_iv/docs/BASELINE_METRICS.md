# Baseline metrics — TMNT IV full hard run

Recorded from continuous power-on dry-run (low-assist).  
Source: `recordings/tmnt_iv_full_hard_dry_run.json`  
Date: 2026-07-25 (sub-hour Raphael route).

The file currently sitting at that path is **not** the published table
below. It contains **210,082f / 5,801 damage**. The table remains the
historical 206,718 / 4,667 snapshot; it is not rewritten as if it were
the current file, and this is not a new official baseline.

## Headline

| Metric | Value |
|--------|-------|
| Power-on → credits | **00:57:19.635** |
| Credits complete frame | **206,718** |
| Total damage taken | **4,667** |
| Emergency HP heals (HP≤16→80) | **65** |
| Form-2 iframe guard frames | **4,635** |
| Life losses | **0** |
| Min HP seen | **2** |
| Lives start / peak / end | **2 / 6 / 6** |
| Hard WRAM | **2** |
| State loads / stage writes / A-special | **0** |

## vs previous baselines

| Metric | Full HP spam | Low-assist + whiplash Slash | Tank + wall fixes | Stage 2–3 pass | **Sub-hour Raphael** |
|--------|--------------|-----------------------------|-------------------|----------------|------------------------|
| Time | 01:28:49.024 | 01:15:34.050 | 01:05:41.709 | 01:04:07.131 | **00:57:19.635** |
| Damage | 12,309 | 8,085 | 6,851 | 6,869 | **4,667** |
| HP interventions | 1,302 (→96 every hit) | 110 (≤16→80) | 93 | 91 | **65** (≤16→80) |
| I-frame guard | 5,042f | 7,467f | 3,887f | 3,824f | **4,635f** |
| Life losses | 0 | 0 | 0 | 0 | **0** |

Δ vs the Stage 2–3 baseline: **−6:47.496**, **−24,490 frames**,
**−2,202 damage**, and **−26 emergency heals**. The form-2 iframe hold
increased by 811 frames; zero life losses held.

## Damage by stage

| Stage byte | Name | Damage | Share | Δ vs Stage 2–3 pass |
|------------|------|--------|-------|---------------------|
| 0 | Big Apple | 334 | 7.2% | +10 |
| 1 | Alleycat Blues | 376 | 8.1% | +30 |
| **2** | **Sewer Surfin'** | **202** | **4.3%** | **−38** |
| **3** | **Technodrome** | **1,022** | **21.9%** | **−240** |
| 4 | Prehistoric | 861 | 18.4% | −283 |
| **5** | **Skull & Crossbones** | **306** | **6.6%** | **−454** |
| 6 | Wounded Knee | 579 | 12.4% | −580 |
| 7 | Neon Night Riders | 238 | 5.1% | −180 |
| 8 | Starbase | 749 | 16.0% | −467 |
| 9 | Final Shell Shock | 0 | 0% | 0 (iframe guard) |

## Stage split times (power-on clock)

| Stage | Elapsed | Split Δ vs Stage 2–3 pass |
|-------|---------|----------------------------|
| Big Apple | 00:00:35.309 | +0:02.413 |
| Alleycat Blues | 00:05:44.216 | −0:09.668 |
| Sewer Surfin' | 00:10:38.382 | −0:31.665 |
| Technodrome | 00:13:34.226 | −0:44.709 |
| Prehistoric | 00:22:23.654 | −0:42.247 |
| Skull & Crossbones | 00:29:11.233 | −1:38.022 |
| Wounded Knee | 00:33:40.173 | −3:04.513 |
| Neon Night Riders | 00:38:26.868 | **−5:04.265** |
| Starbase | 00:42:16.573 | **−6:01.970** |
| Final Shell Shock | 00:50:44.619 | **−6:57.779** |
| Credits complete | **00:57:19.635** | **−6:47.496** |

Raphael's real-menu route and tighter Wounded Knee cadence produced the
largest cumulative gain. Wounded Knee fell from **6:46.447** to
**4:46.695** while its damage fell from 1,159 to 579. The Starbase launch
guard then preserved the intended opening lane and prevented the faster
entry from entering an enemyless stall.

## Stage 1 Clean-track probes (2026-07-25)

Not a whole-run baseline replacement. Segment evidence after pizza seek
(`0x30`) + Baxter left-lane + post-pickup disengage. Source:
`recordings/stage1_clean_track/stage1_probes.json` (3/3 stable).

| Probe | Heal | Outcome | Frames | Damage | E-heals | Notes |
|-------|------|---------|--------|--------|---------|-------|
| Continuous Big Apple (current) | emergency | clear | 18,565 | **334** | in whole-run 65 | power-on context |
| `Stage1` segment | **none** | **stage_advance** | **14,921** | **130** | **0** | Clean Stage 1; entry HP 76 |
| `Stage1` segment | emergency | stage_advance | 14,921 | 130 | 1 | same route |
| `Boss` (Baxter) | none | stage_advance | 4,293 | **64** | **0** | Clean Baxter |

Publication target remains **Bronze / Clean** (maturity stays M8).

## Biggest remaining damage targets

Ranked by absolute damage still taken (best ROI for next policy work):

1. **Technodrome (1,022 / 21.9%)** — largest continuous bucket. Leo
   `blocker_hit_frames=8` probe win **failed** continuous transfer (→1,131)
2. **Prehistoric (861 / 18.4%)** — grind only on a **Raphael** Slash state
3. **Starbase (749 / 16.0%)** — waves + Super Shredder form 1
4. **Wounded Knee (579 / 12.4%)** — stall-only thrash escape for elevated `0xb0`
5. **Alleycat Blues (376 / 8.1%)** — checkpoint gain still does not transfer
6. **Big Apple (334 → segment Clean 130)** — reconcile power-on context

## Tokka/Rahzar + tank probe

`CombatPositionStall` no longer jump-escapes during duo bosses. Emergency
heal HP≤16→80.

Pink Foot / tank throw fix (2026-07-24): pure-run charge ≥34f (no Y, no
align), then 10f toward+Y; align only on retreat/grab. Old 2f Y-tap after
early dx<16 whiffed ~75% of stun cycles (FullHardTank: 1 shredder chip /
8k f → clear).

| State | Stall-suppress | Pre-charge-fix | **Charge fix (production)** | hit_frames=8 (Leo only) |
|-------|----------------|----------------|------------------------------|--------------------------|
| FullHardTank | — | timeout 20k / 708 / 10 | **9,366f / 232 / 3** | 6,325f / 184 / 2 |
| FullHardStage4 | — | — | 31,380f / 1,232 / 17 | 29,982f / 1,024 / 14 |
| Boss4 (→stage 4) | 3,218f duo-only | 15,345f / 470 / 7 | **16,422f / 468 / 7** | — |
| Boss6_hp80 | **3,236f / 176 / 2** | 3,888f / 176 / 2 | (unchanged) | — |

Tank segment (charge fix): **−476 dmg, −7 heals**. hit_frames=8 is a Leo
probe win that **regressed continuous Raph Technodrome** (1,022→1,131) and
is **not** production. Whole-run Technodrome remains **1,022 / 8:49.428**.

`probe_boss_metrics` now supports `--heal emergency|none` (default: emergency)
to match the production low-assist run.

## Slash fight known facts

- Char `0x50`, spawn HP **160**, stage byte **4**, event `0x0A`
- **Prefer** `RaphFullHardBoss5` (char 8, continuous-faithful)
- Legacy Leo: `FullHardBoss5`, `Boss5`, `Boss5_mid`, `FullHardBoss5_hp48`
- Entity status (EnemyState.animation): spin `0xEE`; punish windows often after spin settles to `0x3E`; hitstun `0x17` / multi `0x2E`
- Production policy: `SlashTactics` hybrid whiplash (approach@48 → jump-cross 22/16 → toward+Y; spin_dodge_adx **52**)
- **RaphFullHardBoss5** production: **11,386f / 478 dmg / 6 heals**
- Raph probe KEEP spin_dodge_adx **40**: **6,765f / 226 / 3** (5/5) — continuous
  dry-runs regressed total (5,474 dmg); **not** production
- Leo FullHardBoss5: **13,651f / 730 / 10** (do not grind for continuous)

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

## Improvement goals (toward Bronze / Clean)

Fewer emergency heals is a function of lower damage + natural pizza, not
a softer HP threshold. Ranked next steps:

1. **Re-dry-run** after tank `blocker_hit_frames=8` + Wounded Knee thrash harden
2. **Technodrome** below **1,000** continuous damage (probe already 1,024 on FullHardStage4)
3. **Raph Slash state** then re-grind approach/cross knobs (Leo KEEP does not transfer)
4. **Alleycat / Big Apple entry context** — checkpoint Clean gains still
   fail to transfer to power-on (376 and 334 continuous)
5. **Form-2 iframe guard → 0** without life losses (last protection assist)
6. Hold **0 life losses** and sub-hour continuous clear while assists fall

## Slash probe progress (emergency heal HP≤16→80)

| Version | State | Frames | Damage | Heals | Outcome |
|---------|-------|--------|--------|-------|---------|
| Pre-pass thrash | FullHardBoss5 | ~33,482 | ~1,820 | ~29 | CLEAR |
| Status-wait only | FullHardBoss5 | 40,000 | 1,908 | 31 | TIMEOUT @ HP44 |
| Lab `hybrid_whiplash` | FullHardBoss5 | 18,570 | 918 | 13 | CLEAR |
| Leo production whiplash | FullHardBoss5 (Leo) | 13,651 | 730 | 10 | CLEAR |
| **Raph production (spin 52)** | **RaphFullHardBoss5** | **11,386** | **478** | **6** | **CLEAR** |
| Raph spin_dodge 44 | RaphFullHardBoss5 | 8,957 | 298 | 4 | CLEAR 3/3 |
| **Raph spin_dodge 40 KEEP** | **RaphFullHardBoss5** | **6,765** | **226** | **3** | **CLEAR 5/5** |

Continuous dry-runs (same policy, spin only):

| spin_dodge_adx | Time | Damage | Heals |
|----------------|------|--------|-------|
| **52 (production)** | **00:57:19.635** | **4,667** | **65** |
| 44 | 00:57:31.316 | 5,152 | 74 |
| 40 | 00:57:52.248 | 5,474 | 78 |

Park spin-40 for a full-route re-tune (Skull/WK regressed hardest).

Research artifacts:
- `docs/SLASH_VULN_MAP.md` — status lexicon, claw vs punish
- `lab/slash_vuln.py`
- `lab/slash_lab.py`
- `docs/SLASH_PATTERN_LAB.md`
- `recordings/local_grind_agent/summary.json` — original approach_band KEEP
