# Wiki beginner Ice/Wave/Spazer charge-only Phantoon (rr-7lc5)

Public policy: https://wiki.supermetroid.run/Phantoon#Any.25_KPDR_.28Ice.2FWave.2FSpazer_Charge_Only.29

Pin: `scratch/post_ws_basement_to_phantoon.state` — room `0xCD13` ~(39,124) p81
gs=8, HP 2500, health 299, missiles 20, supers 5, items `0x3105`, beams `0x1007`.
Assist ON. Product body **not** rewritten.

## Wiki mapping

Beginner KPDR Phantoon-second: Charge+Wave+Spazer (+Ice equipped here,
`0x1007`). 2500 HP. Left-corner seat. Charge when the eye opens, then two more
so he disappears (~300 dmg/charge, 3 per round); repeat ~4 rounds. Crouch/snipe
rain. Never Super except a last-round finisher that actually kills (enrage is 8
flame waves). Eye opens after 1 / 6 / 11 s (fast/mid/slow), left or right.
Round-1 has 6 positions; later rounds another 6. **300+ dmg in one barrage
makes him disappear**; space shots if you want 3 in one window.

Product body (`combat/phantoon.py`) is left-corner charge, never Super. Probe
default is `shots_per_window=1`; spine hop is `shots_per_window=3`. This bench
measures that discrepancy without wiring a change.

## Bench (assist, pin reload between rows)

Probe: `scripts/probe/phantoon_wiki_charge.py bench`. Every row through
`format_segment_time`. Dual is the faster successful policy on a second reload
(tie → `probe_default`).

| policy | shots_per_window | frames | seconds | clock | success | shots | windows | HP | boss bit | assist writes/deaths |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---|
| probe_default | 1 | **20537** | 341.721 | 05:42.28 | yes | 9 | 9 | 0 | 1 | 25 / 0 |
| spine_three | 3 | **20537** | 341.721 | 05:42.28 | yes | 9 | 9 | 0 | 1 | 25 / 0 |
| dual probe_default | 1 | **20537** ×2 | 341.721 | 05:42.28 | yes | 9 | 9 | 0 | 1 | 25 / 0 |

Chips (both policies, same frames): 8×300 then 100 to kill.

| # | frame | HP |
|---|---:|---|
| 1 | 1566 | 2500→2200 |
| 2 | 4682 | 2200→1900 |
| 3 | 6495 | 1900→1600 |
| 4 | 8644 | 1600→1300 |
| 5 | 10258 | 1300→1000 |
| 6 | 12443 | 1000→700 |
| 7 | 15545 | 700→400 |
| 8 | 18515 | 400→100 |
| 9 | 19506 | 100→0 (body 0 at 19507, `$D82B` bit 0 at 20537) |

Entry after settle: `0xCD13` (39,128) p81 gs=8 health 299 selected=2 (supers —
policy selects beam before the first open). Final: HP 0, boss bit 1, gs=8,
health 299, supers still 5, selected=0. Method `left_corner_charge_beam`.

## Δ vs 20537f

Baseline assist dual **20537f** ×2 (rr-tlaq, `phantoon_assist_kill.json`).

| policy | Δ frames | Δ seconds | Δ clock |
|---|---:|---:|---|
| probe_default | 0 | 0.000 | +00:00.00 |
| spine_three | 0 | 0.000 | +00:00.00 |
| dual | 0 ×2 | 0.000 | +00:00.00 |

`spine_three` vs `probe_default` is also **0f**. Deterministic match of the
published assist kill.

## `shots_per_window=3` measured behavior

**Does not land 3 chips.** Both policies fire **1 charge per window** (mean
1.0, max 1, `three_chips_landed=false`, `disappear_after_300=true`).

Ice/Wave/Spazer charge is 300. Wiki Eye Open: 300+ in one barrage closes the
eye. Charge recharge is ~60–70f (`CHARGE_FULL=60`); the close after a 300 chip
is faster than a second full charge, so `_fire_window` sees `not hittable` and
breaks even when `shots_per_window=3`. Wiki “charge + two more, ~4 rounds” is
the *spaced* 3-chip barrage; this controller cannot space a second charge
inside that window. Result is **9 one-chip rounds** (8×300+100), not 4×3.

One extra full-charge dump at f6654 (`phan_farm_snipe`, drop 0) during a skip
park — not a HP chip.

## Rain / right-park skip

Product already skips rain except (48,96) and skips right fig-8 (`x≥155`).
Measured: **16704 / 20537** frames are `phan_farm_snipe` (left-seat UP+tap X
through `$D82A` / right parks). Charge 2193f, fire 18f, death-anim 1030f.
No Super (supers stay 5; `missile_writes=0`). Assist energy 500 restored / 25
writes, max hit 20, deaths 0 — `$D82A` is tanked by assist, not by sitting in
rain.

## Recommendation (do not wire)

**Keep the product body.** Left-corner charge, never Super, is the wiki
beginner seat and the measured 20537f kill. `shots_per_window=3` on the spine
hop is wiki-aspirational and a **no-op** versus probe `=1`: same 9 chips, same
frames. Do not rewrite `combat/phantoon.py`. Optional later honesty: spine
could pass `=1` so the hop matches the probe; it would not change the fight.
Do not Super-spray a last-round finisher unless a measured window actually
kills (remaining 100 is a charge chip here).

## Files

- `scripts/probe/phantoon_wiki_charge.py` — `strategy` / `bench`
- `scratch/phantoon_wiki_charge_only.json` — both policies + dual block
- `scratch/phantoon_wiki_charge_only_dual.json` — `probe_default` ×2
- `tests/test_phantoon_wiki_charge.py` — helpers, no emulator

Did **not** edit `combat/phantoon.py`, `routes/kpdr/k6/phantoon_fight.py`,
STATUS, AGENTS, hops, or `recordings/phantoon.json`. Did **not** clobber
`post_phantoon_poweron.state` / `post_phantoon_defeated.state`.

## Honest reds

- No-assist full kill still red (energy floor at `$D82A`). Not re-run.
- Wiki 3-charges-per-round is not achieved with this charge combo; 300-disappear
  + recharge makes `shots_per_window=3` identical to `1`.
- Most of the clock is skip-park farm-snipe, not shooting. A real 4-round
  would need either spaced non-charge shots (missiles) or a 2-round X-factor /
  doppler — out of scope.
- Last-round Super finisher not used (and not needed; last chip is 100).
- Did not STATUS-promote. Did not run power-on `--to phantoon` (rr-8g2u).
