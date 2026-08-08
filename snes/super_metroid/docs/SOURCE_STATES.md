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

**Planned registry hardening** (see [`plan.md`](plan.md) / [`ARCHITECTURE.md`](ARCHITECTURE.md)):

- Provenance (capture command, parent tip, capabilities) per row.
- Fingerprint validation on pure load: room + pose + x/y band must match.
- Dispatch auto-suggest `--source` from room id + required capabilities.
- On pure RED: auto-capture last pin (+ short clip when tooling lands).

## Spine / KPDR continuous-like

| ID | Path (under SuperMetroid-Snes/) | Room | Capabilities (approx) | Use for |
|----|---------------------------------|------|------------------------|---------|
| `natural_post_spore_spawn` | `natural_post_spore_spawn.state` | `0x9B5B` Spore Super | Morph/Bombs/Supers 0 | pure room practice, K0–K3 probes; **not** continuous tip |
| `post_varia_collected` | `scratch/post_varia_collected.state` | `0xA6E2` Varia | post-Varia collect | pure `varia-to-kraid` |
| `post_varia_continuous` | `scratch/post_varia_continuous.state` | `0xA6E2` Varia (x=119/y=126/pose=81) | full zero-load `--to varia` candidate, items `0x1105` | revalidate the reverse chain against persistent route state |
| `post_varia_continuous_to_kraid` | `scratch/post_varia_continuous_to_kraid.state` | `0xA59F` Kraid | natural Varia-lineage Varia→Kraid exit | revalidate Kraid→Eye |
| `post_varia_continuous_to_eye` | `scratch/post_varia_continuous_to_eye.state` | `0xA56B` Eye Door | natural Varia-lineage Kraid→Eye exit | revalidate Eye→Baby |
| `post_varia_continuous_to_baby` | `scratch/post_varia_continuous_to_baby.state` | `0xA521` Baby Kraid | natural Varia-lineage Eye→Baby exit | revalidate Baby→Kihunter |
| `post_varia_to_kraid` | `scratch/post_varia_to_kraid_pure.state` | `0xA59F` Kraid | post-Varia return | pure `kraid-to-eye-return` |
| `post_kraid_to_eye` | `scratch/post_kraid_to_eye_return.state` | `0xA56B` Eye Door | post pure K3.3 | pure `eye-to-baby-return` |
| `post_eye_to_baby` | `scratch/post_eye_to_baby_return.state` | `0xA521` Baby Kraid | post pure K3.4 | pure `baby-to-kihunter-return` |
| `post_baby_to_kihunter` | `scratch/post_baby_to_kihunter_return.state` | `0xA4DA` Kihunter | historical fixture chain | fixture-only `kihunter-to-zeela-return` (not route-ready) |
| `post_varia_continuous_to_kihunter` | `scratch/post_varia_continuous_to_kihunter.state` | `0xA4DA` Kihunter (after 5f settle: x=461/y=395/pose=165) | natural-input successors of `post_varia_continuous` through Baby→Kihunter | authoritative revalidation source for `kihunter-to-zeela-return` |
| `post_varia_continuous_to_zeela` | `scratch/post_varia_continuous_to_zeela.state` | `0xA471` Zeela | natural Varia-lineage Kihunter→Zeela exit | authoritative revalidation source for `zeela-to-warehouse-return` |
| `post_varia_continuous_to_warehouse` | `scratch/post_varia_continuous_to_warehouse.state` | `0xA6A1` Warehouse right ledge | natural Varia-lineage Zeela→Warehouse exit | authoritative revalidation source for reverse `warehouse-to-business` |
| `post_business_continuous` | `scratch/post_business_continuous.state` | `0xA7DE` Business Center | integrity-green `--to business` endpoint | authoritative source for `business-to-frog-save` / `business-to-ice-gate` |
| `post_business_to_ice_gate_pure` | `scratch/post_business_to_ice_gate_pure.state` | `0xA815` Ice Gate ~(1752,651) p2 | pure dual GREEN Business→Ice Gate (rr-fg3, 894f ×2); pre-Speed loadout | geometry pin only |
| `post_business_to_ice_gate_wave_speed_pure` | `scratch/post_business_to_ice_gate_wave_speed_pure.state` | `0xA815` ~(1752,651) Wave+Speed | Gate pin with Speed (Boost Blocks) + Wave | pure `ice-gate-to-acid` |
| `post_ice_gate_to_acid_pure` | `scratch/post_ice_gate_to_acid_pure.state` | `0xA75D` Acid ~(470,139) p2 | pure dual GREEN Gate→Acid (rr-9t4, 370f ×2) | pure Acid→Snake |
| `post_frog_continuous` | `scratch/post_frog_continuous.state` | `0xB167` Frog Savestation (reload: x=60/y=139/pose=1) | integrity-green `--to frog` **side tip** endpoint | authoritative source for `frog-save-to-speedway` (parked post-Speed) |
| `post_frog_save_to_speedway_pure` | `scratch/post_frog_save_to_speedway_pure.state` | `0xB106` Frog Speedway (reload: x=39/y=139/pose=11; door_transition=0) | continuous-like pure successor of Frog Save from `post_frog_continuous` (pure GREEN, frames=295) | pure `speedway-to-farm` / K4.2; **not** continuous tip |
| `post_business_to_frog_save_pure` | `scratch/post_business_to_frog_save_pure.state` | `0xB167` Frog Savestation (reload: x=60/y=139/pose=1) | recorded pure handoff from accepted Business; superseded by the accepted Frog checkpoint | probe record only |
| `post_bat_cave_continuous` | `scratch/post_bat_cave_continuous.state` | `0xB07A` Bat Cave | integrity-green `--to bat_cave` **primary tip** endpoint (122,304f ×2) | authoritative source for Bat → Speed Hall pure |
| `post_bubble_to_bat_pure` | `scratch/post_bubble_to_bat_pure.state` | `0xB07A` Bat Cave ~(39,395) p11 | pure GREEN R19 Bubble→Bat successor | alternate pure source for Bat → Speed Hall |
| `post_bat_cave_to_speed_hall_pure` | `scratch/post_bat_cave_to_speed_hall_pure.state` | `0xACF0` Speed Hall | pure GREEN Bat→Hall successor | pure `speed-hall-to-speed` |
| `post_speed_hall_pre_speed_with_spazer` | `scratch/post_speed_hall_pre_speed_with_spazer.state` | `0xACF0` Speed Hall ~(54,125) | HJ/Varia + beams **`0x1004`**, **no Speed** | human / pure right before Speed Booster; geometry from pure Hall, Charge+Spazer OR'd (full power-on blocked at Ceres) |
| `post_speed_hall_to_speed_pure` | `scratch/post_speed_hall_to_speed_pure.state` | `0xAD1B` Speed Room ~(169,123) p2 | pure GREEN Hall→Speed collect; items `0x3105` | next pure / human record |
| `post_speed_collected` | `scratch/post_speed_collected.state` | `0xAD1B` Speed Room standing | **human-record save point** after Speed (alias of pure handoff) | `guided_human --from speed` → Wave/Ice/Moat |
| `business_climb_entry` | `scratch/continuous_like_business_climb_entry.state` | `0xA7DE` Business Center floor band | Hi-Jump, continuous-like | pure `business-to-warehouse` |
| `continuous_like_bat` | `scratch/continuous_like_bat.state` | `0xA3DD` Bat room | pre-Kraid | bat pure / dwell isolation |
| `post_below_spazer_for_spazer_pure` | `scratch/post_below_spazer_for_spazer_pure.state` | `0xA408` Below Spazer (x=39/y=395/pose=9; reload: x≈49/y=395/pose=1) | Morph/Bombs, beams 0x0000 | pure `below-spazer-to-spazer` / `spazer-collect-return`; source capture from continuous-like Bat handoff (668f) |
| `post_below_spazer_with_charge_continuous` | `scratch/post_below_spazer_with_charge_continuous.state` | `0xA408` Below Spazer ~(49,395) | Morph/Bombs + **Charge** `0x1000`, supers 5 | integrity-green continuous `--to below_spazer` **with Charge on spine** (84,880f); climb source for Spazer fold |
| `post_warehouse_with_spazer_continuous` | `scratch/post_warehouse_with_spazer_continuous.state` | `0xA6A1` Warehouse ~(50,121) | beams **`0x1004`** Charge+Spazer | Charge continuous pin → pure `spazer-detour` → West→Glass→East→Warehouse (skip full power-on while Ceres flaky); mainline Spazer handoff |
| `pre_spazer_door_with_charge` | `scratch/pre_spazer_door_with_charge.state` | `0xA408` top ledge ~(460,139) | Charge + supers 5 | **pre green Super door** for Spazer entry pure; parent continuous-with-Charge + place (geometry developmentOnly; inventory continuous-legal) |
| `post_spazer_entry_pure` | `scratch/post_spazer_entry_pure.state` | `0xA447` Spazer ~(39,139) | Charge held | pure `below-spazer-to-spazer` GREEN from pre-door |
| `post_spazer_collect_pure` | `scratch/post_spazer_collect_pure.state` | `0xA447` Spazer ~(171,171) | beams `0x1004` Charge+Spazer | pure `spazer-collect` GREEN |
| `post_spazer_return_pure` | `scratch/post_spazer_return_pure.state` | `0xA408` Below Spazer top ~(380,155) | beams `0x1004` | pure `spazer-return-to-below` GREEN; handoff clear of open Super door (x≲400); TOP-MID / `spazer-top-to-west` source |
| `post_spazer_west_pure` | `scratch/post_spazer_west_pure.state` | `0xCF54` West Tunnel ~(39,108) | beams `0x1004` | pure `spazer-top-to-west` GREEN **1281f** ×2 from return handoff |
| `red_to_warehouse` | `scratch/red_to_warehouse_controller.state` | Red Tower → Warehouse path | post-Supers | pure `warehouse-hijump-kraid` |
| `natural_bomb_torizo_active` | `scratch/natural_bomb_torizo_active.state` | `0x9804` Bomb Torizo | continuous-like entry | pure bomb-torizo combat verification |

### Post-Varia K4 reverse chain

Historical fixture results for reverse controller development. A controller is
not route-ready until it also clears from its real predecessor state.

| ID | Path | Room | Use for |
|----|------|------|---------|
| `post_varia_to_kraid` | `scratch/post_varia_to_kraid_pure.state` | `0xA59F` Kraid | pure `kraid-to-eye-return` |
| `post_kraid_to_eye` | `scratch/post_kraid_to_eye_return.state` | `0xA56B` Eye Door | pure `eye-to-baby-return` ✓ green |
| `post_eye_to_baby` | `scratch/post_eye_to_baby_return.state` | `0xA521` Baby Kraid | pure `baby-to-kihunter-return` ✓ green |
| `post_baby_to_kihunter` | `scratch/post_baby_to_kihunter_return.state` | `0xA4DA` Kihunter | fixture GREEN only; superseded by full-continuous revalidation |
| `post_varia_continuous_to_kraid` | `scratch/post_varia_continuous_to_kraid.state` | `0xA59F` Kraid | natural Varia chain ✓ green → source for K3.3 |
| `post_varia_continuous_to_eye` | `scratch/post_varia_continuous_to_eye.state` | `0xA56B` Eye Door | natural Varia chain ✓ green → source for K3.4 |
| `post_varia_continuous_to_baby` | `scratch/post_varia_continuous_to_baby.state` | `0xA521` Baby Kraid | natural Varia chain ✓ green → source for K3.5 |
| `post_varia_continuous_to_kihunter` | `scratch/post_varia_continuous_to_kihunter.state` | `0xA4DA` Kihunter after 5f settle x=461/y=395/pose=165 | natural predecessor of K3.6 ✓ green |
| `post_varia_continuous_to_zeela` | `scratch/post_varia_continuous_to_zeela.state` | `0xA471` Zeela | K3.6 natural green → source for K3.7 |
| `post_varia_continuous_to_warehouse` | `scratch/post_varia_continuous_to_warehouse.state` | `0xA6A1` Warehouse right ledge | K3.7 natural green → K3.8 reverse-stack source |
| `post_business_continuous` | `scratch/post_business_continuous.state` | `0xA7DE` Business Center | two matching integrity-green `--to business` runs → source for K4 Frog save (side) |
| `post_frog_continuous` | `scratch/post_frog_continuous.state` | `0xB167` Frog Savestation | two matching integrity-green `--to frog` runs → source for K4 Speedway (side tip; not primary) |
| `post_bat_cave_continuous` | `scratch/post_bat_cave_continuous.state` | `0xB07A` Bat Cave | two matching integrity-green `--to bat_cave` runs (122,304f) → primary tip; Bat → Speed Hall source |
| `post_kihunter_to_zeela` | `scratch/post_kihunter_to_zeela_return.state` | `0xA471` Zeela | historical fixture only |
| `post_zeela_to_warehouse` | `scratch/post_zeela_to_warehouse_return.state` | `0xA6A1` Warehouse right ledge x≈728 | historical fixture only |

The chained reverse fixtures do not by themselves prove persistent room state
from the power-on run (for example, the Warehouse Super-block stack). Capture
a fresh accepted Varia checkpoint with:

```bash
uv run python snes/super_metroid/scripts/record/continuous.py --to varia --no-video \
  --state-output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_varia_continuous.state
```

The 2026-08-01 Varia candidate is GREEN at 104,382f. Its natural return chain
now clears Varia→Kraid→Eye→Baby→Kihunter→Zeela→Warehouse→Business in 9,343
controller frames from that accepted checkpoint. Two power-on `--to business`
runs then reached ordinary Business at 113,723f with zero state loads,
progression/capacity writes, and deaths; the accepted endpoint is
`post_business_continuous`.
The Business elevator descent and blue-door Frog exit then cleared in 1,190
pure frames from that source. Two power-on `--to frog` runs reached ordinary
Frog Savestation at 114,923f with the same zero-write/zero-death integrity;
the accepted endpoint is `post_frog_continuous`.

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

## Cathedral first Bubble (no Speed) — pure sources

Authoritative first Bubble path is Cathedral climb, not Frog Speedway.

| ID | Path (under SuperMetroid-Snes/) | Room | Use for |
|----|---------------------------------|------|---------|
| `post_rising_tide_to_bubble_pure` | `scratch/post_rising_tide_to_bubble_pure.state` | `0xACB3` Bubble entry | source for `bubble-to-bat-cave` (CATH-04) |
| `post_bubble_to_bat_pure` | `scratch/post_bubble_to_bat_pure.state` | `0xB07A` Bat Cave ~(39,395) p11 | **R19 pure GREEN successor**; next hop Bat→Speed |
| `post_bubble_phase_d_pure_r19` | `scratch/post_bubble_phase_d_pure_r19.state` | `0xACB3` ~(305,141) | Phase D pin (door recon); not hop GREEN alone |
| `post_bubble_mid_climb_pure` | `scratch/post_bubble_mid_climb_pure.state` | `0xACB3` mid pin | mid-iso only |
| `post_bubble_right_contact_pure` | `scratch/post_bubble_right_contact_pure.state` | `0xACB3` Phase C band | **dev handoff** via `--dump-phase-c`; climb-only |
| `bubble_human_runway` | `scratch/bubble_human_runway.state` | `0xACB3` ~(27,395) p2 | **dev isolation** Phase D (R15); not pure proof |
| `post_bubble_fire_seat_live_r18` | `scratch/post_bubble_fire_seat_live_r18.state` | fire seat | lucky live isolation (tops without wait); not hop GREEN |

Phase ladder + capture commands:
[`tasks/HARD_ROOM_SPLITS.md`](tasks/HARD_ROOM_SPLITS.md) ·
techniques [`tasks/BUBBLE_TECHNIQUES.md`](tasks/BUBBLE_TECHNIQUES.md).
R19 pure Bubble → Bat is green (see [STATUS.md](STATUS.md)).

```bash
# Full pure GREEN Bubble → Bat (R19)
uv run python snes/super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bubble_to_bat_pure.state \
  --pin-json snes/super_metroid/debug/bubble_to_bat_pure_pin_r19.json --no-red-diag
# success=true room=0xB07A frames=2012

# Primary continuous tip endpoint (Bat Cave)
uv run python snes/super_metroid/scripts/record/continuous.py --to bat_cave --no-video \
  --state-output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bat_cave_continuous.state
```

## Capture recipe (for SM-*-SRC cards)

1. Run continuous (or pure chain) to the predecessor exit.
2. Dump state into `scratch/<label>.state` via existing probe dump helpers
   or session env `write_state_bytes`.
3. Verify room id + pose band on load (one pure probe that only asserts entry).
4. Add a row to this catalog; reference the row id from the geometry card.

## Gaps (need capture)

| Needed for | Expected room | Blocker | Next card |
|------------|---------------|---------|-----------|
| pure HJ shaft mid-climb isolation | `0xAA41` band | `SM-HJ-SRC` partial (ensure_morph RED) | SM-HJ-SRC follow-up or continuous dump |
| pure business climb post-Varia entry | `0xA7DE` floor band | no continuous-like source at Business floor after Varia return; `business_climb_entry` is pre-Varia | SM-SRC-BUSINESS |
| pure bubble mountain entry (K4 **Speed** shortcut) | `0xACB3` Bubble Mountain | post-Speed only (Frog→Speedway→Farm); **not** first Bubble path | park until Speed |
| pure Speed Hall after Bat (K4) | Speed Hall / Speed Room | next hop pure from `post_bat_cave_continuous` / `post_bubble_to_bat_pure` | **SM-K4.4-GRAPH** / Bat→Speed |
| pure moat entry (K6) | `0x95FF` Moat | needs capture after Crateria elev + Kihunter; loadout: Speed, Hi-Jump, PB | SM-SRC-MOAT |
| pure west ocean / WS entry (K6) | `0x93FE` West Ocean | **pins:** Moat handoff `scratch/post_moat_west_ocean_spark.state` ~(49,1163); mid-right door pure `scratch/post_west_ocean_door_spark.state` → `0xC98E` (not green WS); green Super `0xCA08` still open | [SHINE_PRACTICE.md](tasks/SHINE_PRACTICE.md) / SM-SRC-WS |
| Landing Site shine practice | `0x91F8` | `scratch/landing_site_speed_practice.state` ~(899,1163) items `0x3105` (not escape `0xF32F`) | `shine_practice.py` drill/human |
| pure crateria Kihunter entry | `0x948C` | needs capture after Crateria elev descent | SM-SRC-CRKIHUNTER |
| practice SEG-08 Gravity Suit | `0xC98E` Bowling Alley → door `0xA1A4` | needs a controllable post-Phantoon, pre-Gravity state (`boss_bits[3] & 0x01`, `collected_items & 0x0020 == 0`) so the powered Gravity PLM is live | SM-ROOM-SEG-08-R1 / SM-ROOM-SEG-08-SRC |
| practice SEG-10 WS West Super | `0xCAF6` WS Main Shaft → door `0xA210` | needs a controllable post-Phantoon state with the West Super PLM still uncollected; the ship is powered only after `boss_bits[3] & 0x01` | SM-ROOM-SEG-10-R1 / SM-ROOM-SEG-10-SRC |
| practice SEG-20 Crab Tunnel | `0xD21C` Crab Hole → door `0xA4F8` | needs a controllable natural state with Gravity (`collected_items & 0x0020`) and at least one Super pack for the underwater green-gate branch | SM-ROOM-SEG-20-R1 / SM-ROOM-SEG-20-SRC |
| practice SEG-21 Spring Ball | `0xD8C5` Shaktool → door `0xA8D0` | needs a controllable natural state with Gravity + Bombs (`0x0020` and `0x1000`) and Spring Ball still clear (`collected_items & 0x0002 == 0`); X-Ray is only for the alternate suitless route | SM-ROOM-SEG-21-R1 / SM-ROOM-SEG-21-SRC |

The current state inventory contains no valid source for these rows. An
emulator scan of all 1,735 `*.state` files found 13 post-Phantoon and 18
Gravity-capable candidates, all development full-loadout anchors
(`items=0xF32F`, `beams=0x100B`); several are input-frozen. They must not be
used to fabricate a practice green. Capture the listed source from real play,
then bootstrap the same doorway and verify ordinary gameplay plus the listed
inventory/event conditions before changing policy geometry.

Update this table when residuals report "blocked on source."
