# Status: Super Metroid

Glossary: [`CONTEXT.md`](../CONTEXT.md). Working board: [plan.md](plan.md).

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M5 (matrix sticker; rungs are the board) |
| Best verified result | Continuous power-on → **Phantoon defeat + basement leave** (KPDR K6) |
| Last verification | 2026-08-24 (STATUS promote 2026-08-25, `rr-b926`) |
| Runtime class | Bronze |
| Intervention class | Survival (resource-assisted: energy + unlocked ammo) |
| Target | Power-on through end of credits with an RTA (any% KPDR, noob loadout) |
| Current assists | Current energy on Zebes + naturally unlocked current ammo |
| Default CLI tip | `phantoon` |
| Acceptance | Ordinary WS Basement `0xCC6F` `(1240,139)` p10 gs=8 + `$D82B` bit 0 |
| Machine report | `recordings/phantoon.json` + `phantoon_dual.json` (**195,336f** each, exact) |
| Save-state loads | 0 |
| Progression / capacity writes | 0 |
| Deaths | 0 |
| Video | `recordings/phantoon.mp4` (Ceres on tape; frames match) |
| Shared ROM SHA-256 | `12b77c4bc9c1832cee8881244659065ee1d84c70c3d29e6eaf92e6798cc2ca72` |
| Milestone board | [routes/MILESTONES.md](routes/MILESTONES.md) |
| Ready work | [Speed optimization toward the 198,000-frame budget and tape-to-policy conversion](plan.md#start-here-next-session) |
| Clean track | **Grapple Skip v5** through the final credits screen in **379,723 frames** from power-on and **`1:38:33.70`** from Ceres control to ship. Guarded replay and the independent vanilla ROM audit both passed. [Clean track](#clean-track) |

## Current verified tip: Phantoon (K6)

Two matching `--to phantoon` runs (2026-08-24, `rr-8g2u`) reached ordinary
Wrecked Ship Basement `0xCC6F` after doppler kill + loot/exit. Exact frame
match both runs (**195,336f**). Integrity green: known transitions, ordered
splits, **0** loads / progression / capacity writes / deaths. Beams **`0x1007`**
(Charge+Spazer+Wave+Ice). Items **`0x3105`**. Video frames match. Planner
STATUS promote `rr-b926` (2026-08-25): this is the **one living tip**. Ice /
Wave / Speed / Moat / WS are prefix CI, not extra products.

| Metric | Value |
|--------|------:|
| Total frames | **195,336** (~54.26 min @ 60 fps), ×2 exact |
| WO → WS | 175,967 @ `0xCA08` |
| Entrance | 176,402 @ `0xCAF6` |
| Main | 177,636 @ `0xCC6F` |
| Basement → room | 178,300 @ `0xCD13` |
| Fight | 195,000 @ `0xCD13` |
| Loot-exit | 195,168 @ `0xCC6F` |
| Final room | `0xCC6F` ordinary gameplay `(1240,139)` p10 gs=8 |
| Beams | **`0x1007`** (Charge+Spazer+Wave+Ice) |
| Items | **`0x3105`** |
| Outcome | `phantoon_defeated` |
| Report | `recordings/phantoon.json` + `phantoon_dual.json` |
| Video | `recordings/phantoon.mp4` |

Previous living tip (still valid prefix): Ice Beam `--to ice` **148,167f** ×2
(`recordings/ice.json` + `ice_dual.json`, 2026-08-10, room `0xA890`, beams
`0x1007`). Superseded as default CLI tip. Not false.

Previous prefix: Wave Beam `--to wave` **136,361f** ×2. Speed Booster
`--to speed` **130,388f** ×2. Frog Save `--to frog` **114,923f** ×2 (side).
Non-Spazer Bat Cave **122,304f** ×2 remains valid history.

Gravity (`rr-kw8t`) is the next spine bead. The residual owns the pin.
Do not treat that pin as this gate.
[`tasks/rr-kw8t-residual.md`](tasks/rr-kw8t-residual.md).
Work: `bd ready -l super_metroid -l spine`.

## Continuous prefix tips (frames only)

| Tip | CLI | Frames | Notes |
|-----|-----|-------:|-------|
| Morph | `morph` | **24,187** | K0; TAS-close Ceres spine, dual-exact (`morph.json`, `morph_dual.json`; was 24,475) |
| Bombs / Torizo | `bombs` | 47,132 | K0 |
| Spore exit | `spore` | 73,216 | K0 |
| Spore Supers | `supers` | 73,251 | K0 |
| Red Tower | `red_tower` | 80,445 | K1 |
| Bat Room | `bat` | 81,652 | K2 |
| Below Spazer | `below_spazer` | 82,300 / **84,880** w/ Charge | K2; Charge mainline `below_spazer_with_charge.json` |
| Warehouse | `warehouse` | 83,512 / 85,992 Charge / **89,416+90,904** Spazer dual | K2; Spazer mainline dual (below) |
| Hi-Jump | `hijump` | 87,696 | K2 |
| Kraid entry | `kraid` | 97,170 | K2 |
| Varia | `varia` | 101,954 best / 104,382 re-verify | K3; keep best published |
| Business return | `business` | 113,723 | K3→K4 |
| Frog Save | `frog` | 114,923 | K4.0 side tip |
| Bat Cave | `bat_cave` | **122,304** | K4.4 previous default tip (non-Spazer dual) |
| Speed Booster | `speed` | **130,388** | K4.5 previous default (Spazer dual) |
| Wave Beam | `wave` | **136,361** | K4.10 prefix CI |
| Ice Beam | `ice` | **148,167** | K4 previous living tip |
| **Phantoon** | **`phantoon`** | **195,336** | **K6 living tip** (rr-8g2u dual exact match; STATUS `rr-b926`) |

All listed tips are integrity-green continuous (0 loads / progression / capacity).
Detail and history: [routes/MILESTONES.md](routes/MILESTONES.md).

## Spazer prefix

Warehouse Spazer is a prefix of the Speed tip, not a second gate. Dual
reports: `recordings/warehouse_with_spazer.json` (**89,416f**) and
`recordings/warehouse_with_spazer_dual.json` (**90,904f**), room `0xA6A1`,
beams `0x1004`. Frame delta is Spore combat variance. The older single
`--to bat_cave` with Spazer (**127,806f**, `recordings/bat_cave_spazer_cwu.json`)
is folded into the Speed dual at **130,388f**.

## Clean track

The default Clean replay and benchmark use Grapple Skip. The program gate
above tracks the assisted route separately.

The V7 Early Supers candidate crosses all three shutters with a mockball
from the natural V6 opening. It collects Early Supers, Bridge Missile,
Charge, Spazer, Wave, Ice, and Gravity. The power-on replay and independent
vanilla ROM audit reach Botwoon at row 204 in **177,723 frames**,
**7:23.45** ahead of V5, with zero loads, RAM writes, or deaths. Samus
enters with 287/399 energy, 20 Missiles, three Supers, and three Power
Bombs.
[V7 splits and replay](SPEED_SPLITS.md#v7-early-supers)

The v5 finish takes **`1:38:33.70` Ceres→Ship** (354,822 frames at 60 fps),
**30.77s faster** than v4 and **43:33.70** above 55:00. Five live Space Jump
rooms carry forward through rebuilt Ridley and Mother Brain fights. The
guarded replay matched all 315 room transitions through row 339 with zero
state loads, RAM writes, or deaths. The independent audit matched snapshot
`1507fd3b589d…` on the final credits screen at frame 379,723, with items
`0x332D` and 376/499 energy. [V5 splits and replay](SPEED_SPLITS.md#v5-through-credits)

The V6 Moonfall opening reaches Big Pink (row 40) in **40,228 frames**
from power-on. Ceres splits at **1:28.65**, 2.66s ahead of the KPDR
56:28.88 reference, and Morph Ball at **3:10.62**, 3.51s ahead. Samus skips
the second Missile pack and beats Bomb Torizo, returning to Parlor **2:09.02
ahead of V5**. The guarded replay and independent vanilla ROM audit passed
with zero loads, RAM writes, or deaths. Big Pink entry has 109/199 energy and
0/5 Missiles.
[V6 opening splits and replay](SPEED_SPLITS.md#v6-moonfall-opening)

The Red Tower and Phantoon candidate collects Gravity Suit in **`45:16.02`** from Ceres
control, **60.75s faster** than v4 at the pickup. The guarded replay and
independent vanilla ROM audit both passed through row 148 in **174,633
frames** from power-on with Phantoon defeated, Gravity collected, and zero
loads, RAM writes, or deaths.
Samus ends in `0xCE40` with 13/399 energy and three Power Bombs.
[Gravity candidate splits and replay](SPEED_SPLITS.md#v5-through-gravity-suit)

The reactive opening clears Ceres Station in **1:32.70** and returns to
Parlor after Bomb Torizo in **7:24.55**, **92.37s** faster than the v4 opening.
The power-on proof and independent audit reached the Green Brinstar elevator
(row 38) with zero loads, RAM writes, or deaths. [Opening splits](SPEED_SPLITS.md#reactive-opening-on-grapple-skip)

The `no_crocomire` policy (no Crocomire, Grapple, or Spring Ball; items
`0x332D`) is the v3 Grapple Skip baseline at **384,844 frames
from power-on** through the credits screen (272 rooms cleared by the policy
build, 13,417f credits tail), **`1:39:59.05` Ceres→Ship** (359,943f @ 60 fps).
The `policy build no_crocomire` proof passed on the vanilla ROM with zero
state loads, RAM writes, and deaths, `endpoint_matches` and `credits_complete`
true. The independent zero-harness audit (`audit_master_tape.py --stem
speed_benchmarks/master_tape_no_crocomire_clean`, all 384,844f) passed:
snapshot `c60e1eb9d86a…` matches the build report and the credits screen is
verified in WRAM (2026-11-19). The tape ends at 476/499 energy. Benchmark
`v3_no_crocomire`: `+43:30.17` vs the `0:56:28.88` reference, `-18:14.90` vs
`v0_baseline`, 15.1% Skill coverage
([v3 report](../recordings/speed_benchmarks/v3_no_crocomire.md)).
[Grapple Skip route and replay](GRAPPLE_SKIP_SCOPE.md) · [Split history](SPEED_SPLITS.md)

| Fact | Value |
|------|-------|
| Morph on clean bombs path | **26,824f** (split; older than assisted 24,187 tip) |
| Clean bombs tip | **GREEN** **49,321f** ×2 (2026-08-06). Parlor `0x92FD`, items `0x1004`. |
| Clean integrity | 0 energy/ammo writes; 0 loads/progression/capacity; dual reverify |
| Evidence | `recordings/bombs_clean*.json`. Residual purged. |
| Next | Superseded by the master tape; see [plan.md](plan.md#start-here-next-session). |

Contract: [CLEAN_TRACK.md](CLEAN_TRACK.md) · [ASSIST_CONTRACT.md](ASSIST_CONTRACT.md).
No Clean dual. The master tape's full-game Clean finish claim is in the program-gate table above.

## Reproduce

```bash
uv run python snes/super_metroid/scripts/record/continuous.py --to phantoon --no-video
uv run python snes/super_metroid/scripts/record/continuous.py --no-video   # default tip = phantoon
uv run python snes/super_metroid/scripts/record/continuous.py --to ice --no-video     # previous living tip
uv run python snes/super_metroid/scripts/record/continuous.py --to wave --no-video
uv run python snes/super_metroid/scripts/record/continuous.py --to speed --no-video
uv run python snes/super_metroid/scripts/record/continuous.py --to morph --clean
```

## Pointers

| Doc | Role |
|-----|------|
| [routes/MILESTONES.md](routes/MILESTONES.md) | Prefix / tip names |
| [ASSIST_CONTRACT.md](ASSIST_CONTRACT.md) | Allowed assists |
| [CLEAN_TRACK.md](CLEAN_TRACK.md) | Clean intervention track |
| [plan.md](plan.md) | Future work only |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Layers |
| [routes/ROUTE_KPDR.md](routes/ROUTE_KPDR.md) · [routes/KPDR_TRACKER.md](routes/KPDR_TRACKER.md) | KPDR spine / progress chart |
