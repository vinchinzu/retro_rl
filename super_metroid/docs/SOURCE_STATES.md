# Super Metroid source-state catalog

Continuous-like pure entry states for executor cards. Prefer these over
ad-hoc warps. All paths are under
`custom_integrations/SuperMetroid-Snes/` unless noted.

**Rules**

- Pure geometry cards must name **exact path + expected room id**.
- Prefer `scratch/continuous_like_*` and natural post-collect anchors over
  door-warp-only fixtures for spine probes.
- After a pure green, consider dumping a fresh continuous-like state at the
  exit room for the next hop (`SM-*-SRC` cards).
- States are local/gitignored noise unless promoted to a named anchor —
  keep the **index** here even when the binary is only on the machine.
- Dev anchors (`dev_*`, `dev_route_anchor_*`) are topology-only probes.
  Do not STATUS-promote from them.
- `natural_post_spore_spawn` is the shared boot fixture for most K0–K2
  practice and is **not** a continuous tip in itself.

## Spine / KPDR continuous-like

| ID | Path (under SuperMetroid-Snes/) | Room | Capabilities (approx) | Use for |
|----|---------------------------------|------|------------------------|---------|
| `natural_post_spore_spawn` | `natural_post_spore_spawn.state` | `0x9B5B` Spore Super | Morph/Bombs/Supers 0 | pure room practice, K0–K3 probes; **not** continuous tip |
| `post_varia_collected` | `scratch/post_varia_collected.state` | `0xA6E2` Varia | post-Varia collect | pure `varia-to-kraid` |
| `post_varia_to_kraid` | `scratch/post_varia_to_kraid_pure.state` | `0xA59F` Kraid | post-Varia return | pure `kraid-to-eye-return` |
| `post_kraid_to_eye` | `scratch/post_kraid_to_eye_return.state` | `0xA56B` Eye Door | post pure K3.3 | pure `eye-to-baby-return` |
| `post_eye_to_baby` | `scratch/post_eye_to_baby_return.state` | `0xA521` Baby Kraid | post pure K3.4 | pure `baby-to-kihunter-return` |
| `post_baby_to_kihunter` | `scratch/post_baby_to_kihunter_return.state` | `0xA4DA` Kihunter | post pure K3.5 | pure `kihunter-to-zeela-return` |
| `business_climb_entry` | `scratch/continuous_like_business_climb_entry.state` | `0xA7DE` Business Center floor band | Hi-Jump, continuous-like | pure `business-to-warehouse` |
| `continuous_like_bat` | `scratch/continuous_like_bat.state` | `0xA3DD` Bat room | pre-Kraid | bat pure / dwell isolation |
| `red_to_warehouse` | `scratch/red_to_warehouse_controller.state` | Red Tower → Warehouse path | post-Supers | pure `warehouse-hijump-kraid` |
| `natural_bomb_torizo_active` | `scratch/natural_bomb_torizo_active.state` | `0x9804` Bomb Torizo | continuous-like entry | pure bomb-torizo combat verification |

### Post-Varia K4 reverse chain

Pure green hops (controller_dev) in the reverse direction Kraid→Eye→Baby→Kihunter→Zeela→Warehouse→Business.
Sources captured at each exit for the next hop:

| ID | Path | Room | Use for |
|----|------|------|---------|
| `post_varia_to_kraid` | `scratch/post_varia_to_kraid_pure.state` | `0xA59F` Kraid | pure `kraid-to-eye-return` |
| `post_kraid_to_eye` | `scratch/post_kraid_to_eye_return.state` | `0xA56B` Eye Door | pure `eye-to-baby-return` ✓ green |
| `post_eye_to_baby` | `scratch/post_eye_to_baby_return.state` | `0xA521` Baby Kraid | pure `baby-to-kihunter-return` ✓ green |
| `post_baby_to_kihunter` | `scratch/post_baby_to_kihunter_return.state` | `0xA4DA` Kihunter | pure `kihunter-to-zeela-return` ✓ green (~1716f) SM-K4-R-CLIMB-REDESIGN |
| `post_kihunter_to_zeela` | `scratch/post_kihunter_to_zeela_return.state` | `0xA471` Zeela | pure `zeela-to-warehouse-return` ✓ green (~1800f) SM-K4-R-ZEELA-REDESIGN |
| `post_zeela_to_warehouse` | `scratch/post_zeela_to_warehouse_return.state` | `0xA6A1` Warehouse right ledge x≈728 | pure `warehouse-to-business` (SM-K4-R-04 RED — reverse stack) |

## Dev / topology anchors (not continuous evidence)

### K0–K3 KPDR probes

| ID | Path | Room | Capabilities (approx) | Use for |
|----|------|------|------------------------|---------|
| `dev_kpdr_ghz` | `dev_kpdr_ghz.state` | `0x9E52` Green Hill Zone | Supers, Morph, Bombs | pure `ghz-to-noob` |
| `dev_kpdr_noob` | `dev_kpdr_noob.state` | `0x9FBA` Noob Bridge | post-GHZ | pure `noob-to-red` |
| `dev_kpdr_red_tower` | `dev_kpdr_red_tower.state` | `0xA253` Red Tower | post-Noob | pure red-tower / dwell probes |
| `dev_kpdr_warehouse` | `dev_kpdr_warehouse.state` | `0xA6A1` Warehouse Entrance | post-Red | warehouse pure / dwell |
| `dev_kpdr_business` | `dev_kpdr_business.state` | `0xA7DE` Business Center | post-Warehouse | pure business climb |
| `dev_kpdr_hj_shaft` | `dev_kpdr_hj_shaft.state` | `0xAA41` Hi-Jump Shaft | post-Business | pure HJ shaft climb |
| `dev_kpdr_kraid_eye` | `dev_kpdr_kraid_eye.state` | `0xA56B` Kraid Eye | post-Warehouse reverse | pure eye→kraid |
| `dev_kpdr_kraid_entry` | `dev_kpdr_kraid_entry.state` | `0xA59F` Kraid | natural entry | pure kraid fight → Varia |
| `dev_kpdr_varia` | `dev_kpdr_varia.state` | `0xA6E2` Varia Room | post-Kraid | varia probes |
| `dev_hijump_room_entry` | `dev_hijump_room_entry.state` | `0xA9E5` Hi-Jump Room | door-warp entry | HJ collect geometry (no boots) |
| `dev_hijump_collected_dev` | `dev_hijump_collected_dev.state` | `0xA9E5` Hi-Jump Room | **granted** HJ boots | HJ exit / return |
| `dev_red_tower_stable` | `dev_red_tower_stable.state` | `0xA253` Red Tower | Supers | red tower stable dwell |
| `dev_kraid_room_natural` | `dev_kraid_room_natural.state` | `0xA59F` Kraid | natural Kraid entry | kraid fight pure |
| `dev_kraid_eye_at_eye` | `dev_kraid_eye_at_eye.state` | `0xA56B` Eye Door | natural eye entry | eye room pure |
| `dev_kraid_defeated` | `dev_kraid_defeated.state` | `0xA59F` Kraid | Kraid dead (dev spray) | post-fight door / Varia approach |
| `dev_varia_equipped_dev` | `dev_varia_equipped_dev.state` | `0xA6E2` Varia | **granted** Varia | varia exit / return |
| `dev_power_bombs_collected` | `dev_power_bombs_collected.state` | `0xA3AE` Alpha PB | warp-collected PB (dev) | Crateria / Moat topology |
| `eye_hj_kraid_entry` | `scratch/eye_hj_kraid_entry.state` | `0xA59F` Kraid | post-HJ/Zeela chain | kraid fight pure |
| `eye_hj_kraid_varia_collected` | `scratch/eye_hj_kraid_varia_collected.state` | `0xA6E2` Varia | post-Kraid fight | Varia collected (items `0x1105`) |
| `dev_b1_supers_natural` | `dev_b1_supers_natural.state` | `0x9B5B` Spore Super | Spore defeated | super collect geometry |

### Boss / late (dev route anchors)

Door-warp topology anchors from the full 22-leg completion chain. All
are **developmentOnly** with granted full loadout + boss bits.

| ID | Path | Room | Leg | Use for |
|----|------|------|-----|---------|
| `dev_phantoon_entry` | `dev_phantoon_entry.state` | `0xCD13` Phantoon | K6 | Phantoon fight probe; ship route |
| `dev_route_anchor_gravity_suit` | `dev_route_anchor_gravity_suit.state` | `0xCE40` Gravity Suit | K6 | gravity entry / WS clear |
| `dev_route_anchor_botwoon` | `dev_route_anchor_botwoon.state` | `0xD95E` Botwoon | K7 | botwoon fight / Maridia |
| `dev_route_anchor_draygon` | `dev_route_anchor_draygon.state` | `0xDA60` Draygon | K7 | draygon fight / Space Jump |
| `dev_route_anchor_ridley` | `dev_route_anchor_ridley.state` | `0xB32E` Ridley | K7 | ridley fight / LN |
| `dev_route_anchor_statues` | `dev_route_anchor_statues.state` | `0xA66A` Statues | endgame | G4 statue room (all bits set) |
| `dev_route_anchor_tourian_elevator` | `dev_route_anchor_tourian_elevator.state` | `0xDAAE` Tourian elev | endgame | Tourian entry |
| `dev_route_anchor_mother_brain` | `dev_route_anchor_mother_brain.state` | `0xDD58` Mother Brain | endgame | MB fight / escape |
| `dev_route_mother_brain_entry` | `dev_route_mother_brain_entry.state` | `0xDD58` MB (east door) | endgame | MB natural-ish entry |
| `dev_route_anchor_tourian_escape_4` | `dev_route_anchor_tourian_escape_4.state` | `0xDEDE` Escape 4 | endgame | escape room 4 |
| `dev_route_anchor_landing_site_finish` | `dev_route_anchor_landing_site_finish.state` | `0x91F8` Landing Site | endgame | escape finish |
| `dev_route_full` | `dev_route_full.state` | (any) | 22-leg finish | full route topology (boss bits set) |
| `dev_route_late_full` | `dev_route_late_full.state` | (any) | 9-leg late | late skeleton topology |
| `dev_escape_room1` | `dev_escape_room1.state` | `0xDE4D` Escape 1 | endgame | escape room 1 geometry |

Door-warp anchors under `dev_*` prove topology only. Do not STATUS-promote
from them. The `dev_route_*` states use `grant_route_loadout` + `mark_all_major_bosses`
and are **not** representative of real continuous loadout.

## Capture recipe (for SM-*-SRC cards)

1. Run continuous (or pure chain) to the predecessor exit.
2. Dump state into `scratch/<label>.state` via existing probe dump helpers
   or session env `write_state_bytes`.
3. Verify room id + pose band on load (one pure probe that only asserts entry).
4. Add a row to this catalog; reference the row id from the geometry card.

## Gaps (need capture)

| Needed for | Expected room | Blocker | Next card |
|------------|---------------|---------|-----------|
| pure `zeela-to-warehouse` (K4 reverse) | `0xA471` → `0xA6A1` | **GREEN** ~1800f SM-K4-R-ZEELA-REDESIGN; graph `controller_dev` | next R-04 warehouse reverse |
| pure `warehouse-to-business` (K4 reverse) | `0xA6A1` right ledge → `0xA7DE` | **RED** elevator-only controller; reverse stack pin x≈325 | SM-K4-R-04B planner redesign |
| pure HJ shaft mid-climb isolation | `0xAA41` band | `SM-HJ-SRC` partial (ensure_morph RED) | SM-HJ-SRC follow-up or continuous dump |
| pure business climb post-Varia entry | `0xA7DE` floor band | no continuous-like source at Business floor after Varia return; `business_climb_entry` is pre-Varia | SM-SRC-BUSINESS |
| pure bubble mountain entry (K4 Speed) | `0xACB3` Bubble Mountain | needs continuous-like capture after Business→Frog Speedway; no source exists | SM-SRC-BUBBLE |
| pure moat entry (K6) | `0x95FF` Moat | needs capture after Crateria elev + Kihunter; loadout: Speed, Hi-Jump, PB | SM-SRC-MOAT |
| pure west ocean / WS entry (K6) | `0x93AA` West Ocean | needs capture after Moat; loadout: Speed, HJ, PB | SM-SRC-WS |
| pure crateria Kihunter entry | `0x948C` | needs capture after Crateria elev descent | SM-SRC-CRKIHUNTER |

Update this table when residuals report "blocked on source."
