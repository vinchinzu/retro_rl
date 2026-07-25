# Final Fight — RAM map

Addresses are **WRAM offsets** for stable-retro `get_ram()` / `data.json`
(USA). Primary source: [TCRF Notes: Final Fight (SNES)](https://tcrf.net/Notes:Final_Fight_(SNES)).
Probe scripts should confirm before trusting combat policies.

## Game / stage

| Name | Addr | Type | Notes |
|------|------|------|-------|
| `game_status` | `0x0CA0` | u8 | `00` char select, `02`/`04` open stage, `06` play, `08` clear area (sub-section), `0A` clear round (stage done), **`0E` Break Car / bonus gameplay** |
| `round` | `0x0CB0` | u8 | `00` Slum, **`01` Subway**, **`02` West Side**, `03` Industrial, `04` Bay, `05` Up Town, **`06` Break Car**, `07` Break Glass |
| `area` | `0x0CB1` | u8 | Sub-area within round (Slum 00–02; Subway 00–03; West Side starts 00) |
| `rounds_cleared` | `0x0CB2` | u8 | Incremented on clear-round handler (**3** at West Side entry) |
| `level_end` | `0x0CD0` | u8 | Write `01` forces level end (stuck if boss not dead) — TCRF |
| `boss_dead_flag` | `0x0CD2` | u8 | `01` when certain bosses die; ends level script / disables player hitbox — TCRF |
| `go_flashing` | `0x0CD7` | u8 | `01` when GO! arrow flashes |
| `char_select` | `0x008F` | u8 | `00` Cody, `01` Haggar |
| `camera_x` | `0x0E07` | u16 LE | Scroll X in round |

## Player 1 (`base 0x0D00`, entity layout)

| Name | Addr | Type | Notes |
|------|------|------|-------|
| `player_active` | `0x0D00` | u8 | `00` inactive, `01` active |
| `player_x` | `0x0D07` | u16 LE | World X |
| `player_y` | `0x0D0D` | u16 LE | Ground / body Y (also `0x0D0A` jump Y). **UP increases Y** |
| `player_hp` | `0x0D14` | u8 | Current HP (max typically `0x80` at `0x0D18`) |
| `player_lives` | `0x0D6E` | u8 | Lives remaining (HUD is −1) |

## Enemies / boss

Entity layout mirrors the player (status / X / Y / HP at the same offsets).
Slots:

| Slot | Base | Stride notes |
|------|------|----------------|
| Enemy 0 | `0x1000` | |
| Enemy 1 | `0x10B0` | `+0xB0` |
| Enemy 2 | `0x1140` | `+0xB0` |
| Boss (Damnd/Thrasher) | `0x11E0` | status `00` none, `01` present undrawn, `03` drawn |

Per-slot fields: `+0x00` status, `+0x07` X u16, `+0x0D` Y u16, `+0x14` HP.
Boss HP also at `0x11F4` (large vs regular thug HP ≤ `0x80`).

### Damnd spawn signals (Slum)

Cam **1536** is the alley lock, not the Damnd door. After enough room-1
clears the camera unlocks past 1536 (observed **1650→1718+**); Damnd/Thrasher
still has not spawned (`0x11E0` stayed `00`). FAQ: Thrasher breaks a wooden
door after further street clears past the alley.

1. **Boss status** `0x11E0` → `01` then `03` (primary)
2. **Boss HP** `0x11F4` large / not thug-sized (0 while status `01`)
3. **Camera** past alley unlock (~1700+) then door fight (not the 1536 lock)
4. Do **not** treat `game_status` `0x08` (CLEAR_AREA) as Damnd/stage clear
5. After `01`, **regular-slot door thugs** (peaks ~82 / 60 / 95) fight first;
   Damnd draw (`03`) was not observed until those clear — treat `01` as
   door-entry, not fight-ready boss

## Probe status

| Check | Result |
|-------|--------|
| TCRF layout in `data.json` / `final_fight/ram.py` | done |
| Headless confirm player X/Y/HP | **confirmed** — walk R/L/U/D moves `0x0D07` / `0x0D0D`; HP `0x0D14` |
| Camera scroll `0x0E07` | **confirmed** (moves once screen unlocks) |
| Enemy slot 0 HP/X/Y | **confirmed** on first Slum wave (`status=3`, HP≈64) |
| Entity `status` byte | living fighters **`0x03`**; junk `0x02`; spawn/despawn `0x01` (do not chase) |
| Inactive junk slots | require status `0x03` and `0 < hp <= 0x80` |
| On-screen band | camera −128 … camera +256+48 (left spawns need wide margin) |
| Screen-lock dedicated flag | tentative: living enemies ∧ ¬`go_flashing` |
| Vertical axis | **UP increases Y** (DOWN decreases); policy uses `invert_vertical` |
| Boss slot | included in `enemies` when combat-active so nearest-target works |
| Boss spawn | **yes** — `0x11E0=01` at cam **2304** / room 2 (`Boss.state`, HP **40**) |
| Boss drawn (`03`) / Damnd HP drop | **yes** — draw HP **44** at cam≈2675; kill → underflow |
| Door thugs before Damnd | peaks **36, 60, 64, 42, 82** (all @ player HP40) |
| Door kick-band close | tough (>50): JD dx 40–103; peak≤50: **park-bait / retreat** |
| Post-kill HP0 / underflow `st=03` | **damages** — flee dx<36; plant-punch; `threat_enemies` |
| Kick band at door | idle/bait at dx≈40–103 chips / one-shots (esp. peak 42); do not sit |
| Damnd draw trigger | after thug5 clear, **creep right** (cam→~2675); do not repark left |
| Damnd fight | kite ~60f after draw, then spam-Y dx 24–40 |
| Damnd kill-frame | HP underflow (~237) at st=`03`; **`0x0CD2` stays 0** |
| Stage clear bridge | `set_value(game_status, CLEAR_AREA)` → `0x0A` → open Subway (`round=01`) |
| Subway entry | round `01` area `00`; HP refresh to 80; first lock cam≈**537** |
| Subway wave-1 | peak **34**; kick dx≈51 one-shots idle/hold; use door kick-band + punch 28–38 |
| Subway status `01` spawn | living HP near cam (sx≈−50) still chips — treat as combat-active |
| Subway cam844 clear | from `Stage2_Clear_w2_cam537`: dual pack = HP0 + **living HP148** |
| Subway HP148 | living max **224** (West Andore ≈216); behind: face-Y (HP>72) then ground walk_past (dx≤103) |
| Subway HP148 evidence | clear **HP67/L1** (`stage2_push_994b`); JD-past burned to HP12 |
| Subway cam847 mid | mid-fight `Stage2_Mid_hp148_p54_e108_cam847`; far `Stage2_Far_hp67_L1_cam900` |
| Subway cam848 ghost-free | **scroll softlock** — skip save; cam≥840 distant HP0 unlock |
| Subway cam≥840 | scroll_mash RIGHT+Y; plant-punch kick-band HP0/UF only |
| Subway cam994 area0 | **scroll softlock** — CLEAR_AREA poke → area1 cam1792 |
| Subway area1 dual-pack | hit-and-run (pulse JD/retreat); nearest in-band; clears cam1969+ |
| Subway cam2561 area1 | **scroll softlock** — CLEAR_AREA poke → area2 cam3840 |
| Subway area1→2 scroll | plain RIGHT (no Y); keep past far-behind; HP54 entry |
| Subway area2 | e69 early **JD90+toward+Y** → clear @HP54; plant UF; HP134→clear @37 |
| Subway cam4130 area2 | **scroll softlock** — CLEAR_AREA poke → area3 cam≈4864 / Sodom |
| Subway Boss2 / Sodom | `0x11E0=01` @cam4864; drawn `03` HP44; UP+Y throw 41→1 (dmg40) + grab mash → UF; `Stage2_Clear` |
| Stage2→3 CLEAR_AREA | poke → Break Car (`round=06`, status `0x0E`) → West Side (`round=02`); HP refresh 80 |
| West Side entry | **`Stage3`** cam **619** / HP80 / area00; first lock cam **640** |
| West Side wave1 | dual 78+91; **sx>170 JD-left** (not punch-band) → clear @**HP80** / 0 pdmg |
| West Side cam640 | post-clear ~90f B+LEFT entry then mid space/Y; Mid_p66 |
| West Side wave2 | clear from Mid @**HP31** (`Clear_w2`); continuous still dies mid-dual |
| West Side wave3 | crumb mash from chip e58 → **`Clear_w3` @HP31** (prefer) |
| West Side wave4 | Andore living HP **216**; right-edge JD → **`Clear_w4` @HP31** (sx≈232) |
| West Side post-w4 | dual 142+96; **cleared** via split+heal crumb → true1v1 LEFT+Y wait-KD → **`Clear_w5_real_p48`**; cam931 CLEAR_AREA → Area1 |
| West Side Area1 | cam**2560**; living thug peaks **HP≈250** (`ENTITY_HP_MAX=255`); Boss3 open |
| Living HP max | **252** (Area1 ≤250); UF ghosts **≥253** (wave5 kill-frame ~254) |
| Subway post-kill ghosts | true HP0/UF≥200; flee dx<40; area2 unlocked plant to dx160 |
| Screen-lock walk band | player X roughly cam+40 … cam+170; past ~cam+170 overshoots |
| Right-edge park | wait cam+100 (tough: cam+72); never idle at hold in kick dx 45–95 |
| Punch connect | probes: HP drops at dx≈28–35 (≈8–10 HP); wider left-edge poke whiffs |
| Throw at right wall | alley policy punch-only (throw_gap trades); no throw_right on lock |
| Room-1 food | **none observed** — wave4_instrument heals=0 through unlock |
| Unlock lives / HP | **lives 2 / HP 38** (`Stage1_PostUnlock_L2`); Damnd via L2 / w3 |
| Continue after door death | `game_status=0x12` — START/A/B/Y probes failed; need lives≥2 |

`Stage1.state` is saved at first living enemy wave (fight-ready), not bare spawn.
Preferred mid clear: `Stage1_Clear_w8_cam777` (HP 80 / lives 3).
Preferred room-1 entry: `Stage1_Room1_Healthy` (HP 80 / lives 2 / cam 1536).
Post-unlock lives≥2: `Stage1_PostUnlock_L2` (HP **38** / lives 2).
Healthy post-unlock: `Stage1_PostUnlock_Healthy` (cam 1600 / HP 42 / lives 1).
Soft: `Stage1_PostUnlock` (cam 1650 / HP 9 / lives 1).
Boss entry: **`Boss.state`** (HP **40** / lives 1 / cam 2304 / `0x11E0=01`).
Mid door: `Boss_ThugMid` (thug 36 / player 40). After thug 1: `Boss_PostThug1`.
Segment runner writes clears with HP ≥ 40, plus `PostUnlock_L2` on unlock
(and `PostUnlock_L2_Healthy` when unlock HP ≥ 40).
Stage 2: **`Stage2`** (subway park, cam 537 / HP 80 / round 01). Prefer mid:
`Stage2_Clear_w2_cam537` (HP80/L2). HP148 mid-fight:
`Stage2_Mid_hp148_p54_e108_cam847` (HP54/L2). Far clear:
`Stage2_Clear_w1_cam848_threat` (HP67/L1). Far train:
`Stage2_Far_hp67_L1_cam900`. Avoid cam848 ghost-free (scroll softlock).
Living HP max **224** (subway 148; West Andore ≈216; UF ≥225 / Damnd ≈237).
Cam≥840: distant HP0 does not lock.
