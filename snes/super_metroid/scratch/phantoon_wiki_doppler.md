# Wiki missile-doppler Phantoon (rr-7lc5)

Scratch probe. Do **not** wire, STATUS-promote, or clobber
`post_phantoon_poweron.state`.

## Wiki mapping

- https://wiki.supermetroid.run/Phantoon
- https://wiki.supermetroid.run/Phantoon#Any.25_PRKD
- https://wiki.supermetroid.run/Phantoon#Any.25_KPDR_.28Charge_Plus_Missiles.29
  (KPDR Room Strategies: “Dopplers” / Charge+Missiles mini-doppler)

PRKD doppler: vanish-damage still needs ~10f to close the eye. Missile
cooldown 9f; wiki spacing **10 frames** so extra missiles land (and can
interrupt the close). Super deals 600 and **enrages** if he lives.

Adapted 2-2-N on KPDR pin inventory (missiles 20, supers 5, beams `0x1007`,
items `0x3105`):

1. Left-corner seat. Skip x≥155 right park; skip rain unless (48, 96)
   y 88–104. Rain reuses product corner snipe (do not jump under the body).
2. Optional uncharged beam tap when already on beam (do not dump a charge).
3. Two missiles, wait 45f, two missiles, 40f gap, then chase extras at 10f
   (cap 6). Hits counted by missiles decreasing + `enemy0_hp` delta.
4. Super only if remaining HP ≤ 600.

Pin: `scratch/post_ws_basement_to_phantoon.state` — `0xCD13` ~(39,124) p81
gs=8 HP 2500 health 299.

## Window proof

`phantoon_wiki_doppler_window.json` — **GREEN** (assist ON).

| | |
|--|--|
| missiles spent | **5** (counted per-frame; assist refill) |
| HP | 2500 → 2200 (**300** chip) |
| recipe | 2-2-1 |
| close-eye extra | 0 |
| halt_miss | true (5th spend did not chip within 48f) |
| frames | 1909 (`00:31.82`) |

Missiles spend **and** HP chips before any 20k fight. Halted at first extra
that did not land.

## Full fight vs 20537f

`phantoon_wiki_doppler.json` + `_dual.json` — assist ON, `$7E:D82B` bit 0
via `read_bank7e_wram` (never `get_ram()[0xD82B]`).

| | product charge | wiki doppler |
|--|--:|--:|
| frames | **20537** ×2 | **12118** ×2 |
| clock | 05:41.66 | **03:21.97** |
| body 0 | 19507 | 11093 |
| boss bit | 20537 | 12118 |
| HP | 0 | 0 |
| `$D82B` bit 0 | yes | yes |
| gs / health | 8 / 299 | 8 / 299 |
| missiles spent | 0 | 25 |
| max barrage | — | 5 |
| rounds | 9 charge chips | 8 |
| Super used? | no | **yes** (4 spent; 2 misses then kill) |
| close-eye extra | n/a | **0** |
| vs 20537 | — | **−8419f** |

Windows: 2500→2200 (2-2-1, 300), then four 400-chip 2-2-1s to 600, then Super
miss / miss / 2-Super 600 kill. No enrage (death-anim 1025f, same ballpark as
product 1030f). Assist: energy_restored 700 / 35 writes, missile_writes 25,
super_writes 4, max hit 20, deaths 0.

## Extra missiles during close-eye delay

**No.** `close_eye_extra=0` on window and fight. Recipe landed **2-2-1**
(3–4 hits of 5 spends per barrage), not wiki 2-2-6. The 5th missile often
missed or arrived after the eye started closing without interrupting it.
Doppler extras 3–6 never counted.

## Super used?

**Yes**, gated on HP ≤ 600. Two seat/rain Super spends missed; last window
spent 2 and chipped 600 (kill). Not a Super-spray enrage.

## Recommendation

**Keep as scratch; do not wire.** Faster than left-corner charge (**12118f**
vs **20537f**, dual-exact this session) and a real HP-0 + boss-bit kill, but:

- Wiki doppler extras did **not** land during the close-eye delay.
- Super finisher is sloppy (2 misses + 2-spend kill).
- Do not replace the product charge policy, STATUS, `recordings/phantoon.json`,
  or `--to phantoon`. Do not clobber `post_phantoon_poweron.state`.

## Files

- `snes/super_metroid/combat/phantoon_doppler.py` (500 lines)
- `snes/super_metroid/tests/test_phantoon_doppler.py` (4 passed)
- `snes/super_metroid/scratch/phantoon_wiki_doppler.json`
- `snes/super_metroid/scratch/phantoon_wiki_doppler_dual.json`
- `snes/super_metroid/scratch/phantoon_wiki_doppler_window.json`
- `snes/super_metroid/scratch/phantoon_wiki_doppler.md`

## Honest reds

- Close-eye doppler extras: **no** (count 0).
- Super: used, not clean (4 spent for one 600 chip).
- Dual: two matching 12118f assist kills this session; not a `--dual` CLI
  rewrite of the 20k product pin.
- Did not STATUS-promote, wire, or touch `combat/phantoon.py`.
