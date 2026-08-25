# Ice-on X-Factor / popTOON Phantoon — measured red

**Bead:** rr-7lc5. Do **not** wire. Do **not** replace product 20537f.

Wiki:
- https://wiki.supermetroid.run/Phantoon#Any.25_KPDR_.28popTOON_-_X-Factor_Plus_Missiles.29
- https://wiki.supermetroid.run/Phantoon#Any.25_KPDR_.28X-Factor_Only.29
- https://wiki.supermetroid.run/Charge_Beam_Combos#Wave_Shield

Wave Shield (X-Factor) = Charge + Wave + Power Bomb, `$0CD0` = 120.
True X-Factor needs Ice **off**. Pin has Ice on. No pause-menu beam-toggle helper.

## Pin (measured)

`scratch/post_ws_basement_to_phantoon.state` → room `0xCD13` `(39,128)` p81 gs=8 HP **2500**.

| field | pin | measured |
|-------|-----|----------|
| beams `$09A6` | `0x1007` | **`0x1007` Charge+Ice+Wave+Spazer** |
| Ice bit | `0x0002` | **set** |
| Spazer `0x0004` | maybe | **set** |
| items `$09A2` | `0x3105` | **`0x3105`** |
| max PB / missiles / supers | 5 / 20 / 5 | 5 / 20 / 5 |
| true Wave Shield | no | **no** |

## Window (required)

`probe/phantoon_xfactor.py window --assist`

Report: `scratch/phantoon_wiki_xfactor_window.json`

| | |
|--|--|
| first open | f1513 func `$D4A8` eye `(120,108)` ilist `$CC57` |
| seat | `(37,187)` p1 face RIGHT |
| charge peak | **120** (combo threshold) |
| PB spent | **0** (5→5, never dipped) |
| `enemy0_hp` | **2500 → 2500** (drop **0**) |
| SBA types `$0C04` | none of `$001C..$001F` (Ice/Wave/Spazer/Plasma shield) |
| slots seen | stale `$0002` / `$0012` at x=36871 y=0 — not live particles |
| outcome | `ice_on_xfactor_miss` |
| time | 1818f / 00:30.30 |

Ice-on combo **does not chip**. Charge 120 with PBs selected did not consume a PB and did not spawn Wave Shield (X-Factor) particles. Wiki SBA needs *one* other beam; pin has Ice+Wave+Spazer together.

## popTOON

Recipe 2+2+XF / 2+2+S is **not feasible** on this inventory:

1. Ice-on SBA did not chip (window red).
2. No existing pause-menu Ice unequip helper — do not invent one.
3. Super only if HP ≤ 600. Never reached.

Full fight **not run**. No `phantoon_wiki_xfactor.json`. No dual vs 20537f.

## Recommendation

**Do not wire.** Product left-corner charge-only assist **20537f** ×2 stays. Ice-on X-Factor is a measured miss, not a 2-round kill.
