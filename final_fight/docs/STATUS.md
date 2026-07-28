# Final Fight — Status


## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M3 |
| Best verified result | Stages 1–2 segment clears; Stage 3 wave5 clear; Area1 chip in progress |
| Last verification | 2026-07-27 |
| Runtime class | Bronze |
| Intervention class | Clean (heal poke when documented) |

**Approach:** save-state + segment scripts first; continuous title-to-credits
later. Retries and mid-stage `.state` files are expected.

| Item | State |
|------|--------|
| Integration `FinalFight-Snes` | wired (`data.json` / metadata / scenario / rom) |
| ROM from shared zip | ready (`scripts/setup_rom.py` / `setup_all_roms`) |
| Dev save past char select | **`Stage1.state`** at first enemy wave |
| Healthy resume (preferred) | **`Stage1_Clear_w8_cam777`** — HP 80 / lives 3 / cam 777 / room 0 |
| Healthy room-1 entry | **`Stage1_Room1_Healthy`** — HP 80 / lives 2 / cam 1536 |
| Post-unlock lives≥2 | **`Stage1_PostUnlock_L2`** — unlock HP **38** / lives 2 |
| Boss entry (preferred) | **`Boss.state`** — HP **40** / lives 1 / cam 2304 / `0x11E0=01` |
| Mid door fight | **`Boss_ThugMid`** — thug HP 36 / player 40 / dx≈31 |
| After door thug 1 | **`Boss_PostThug1`** — HP 40 / cam 2332 |
| After door thug 2 | **`Boss_PostThug2`** — HP **40** / cam 2365 (jump-dash) |
| After door thug 3 | **`Boss_PostThug3`** — HP **40** / cam 2449 (ghost punch + JD) |
| After door thug 4 | **`Boss_PostThug4`** — HP **40** / cam 2449 (park-bait) |
| After door thug 5 | **`Boss_PostThug5`** — HP **40** / cam 2496 (rise+delay JD) |
| Damnd drawn | **`Boss_Drawn`** — HP **40** / cam 2678 / `0x11E0=03` / boss HP **44** |
| Stage 1 clear (kill frame) | **`Stage1_Clear`** — HP **40** / Damnd HP underflow / cam 2751 / `0x0CD2=00` |
| Stage 2 entry (subway) | **`Stage2.state`** — HP **80** / lives 1 / cam **537** / round **01** / area **00** |
| Stage 2 mid clear (prefer) | **`Stage2_Clear_w2_cam537`** — HP **80** / lives 2 / cam 537 |
| Stage 2 HP148 mid-fight | **`Stage2_Mid_hp148_p54_e108_cam847`** — HP **54** / L2 / tough 108 / cam847 |
| Stage 2 L2 HP148 clear | **`Stage2_Clear_L2_hp54_cam847_threat`** — HP **54** / L2 / cam847 |
| Stage 2 far clear | **`Stage2_Clear_w1_cam848_threat`** — HP **67** / lives 1 / cam **848** (post-HP148; scroll-mash) |
| Stage 2 far (train approach) | **`Stage2_Far_hp67_L1_cam900`** — HP67 / L1 / cam900 |
| Stage 2 cam994 mid | **`Stage2_Mid_cam994_p80_e34`** / **`Stage2_Clear_cam994_p60_L1`** |
| Stage 2 area1 entry | **`Stage2_Area1_hp60_L1_cam1792`** — CLEAR_AREA poke from cam994 clear |
| Stage 2 area1 dual clear | **`Stage2_Area1_clear_hp60_L1_cam1969`** / **`…_hp54_L1_cam1998`** / **`…_hp54_L1_cam2415`** |
| Stage 2 area1 far | **`Stage2_Far_hp54_L1_cam2434`** / **`Stage2_Area1_clear_hp54_L1_cam2454`** |
| Stage 2 area1→2 prebridge | **`Stage2_Area1_clear_hp38_L1_cam2561`** (scroll softlock) |
| Stage 2 area2 entry | **`Stage2_Area2_hp54_L1_cam3840`** — prefer (scroll chip fixed); old HP38 still valid |
| Stage 2 area2 1v1 mid | **`Stage2_Area2_mid_p54_e79_cam3891`** / **`…_1v1_p54_e69_cam3900`** / **`…_1v1_p54_e28_cam3969`** / **`…_1v1_p15_e3_cam3914`** |
| Stage 2 area2 pack clear | **`Stage2_Area2_clear_hp54_L1_cam3968`** — e69 early JD90+Y (prefer); old HP15 still valid |
| Stage 2 area2 post pack | **`Stage2_Clear_w2_cam3968_threat`** — HP **37** after HP134; **`…_mid_p37_e80_cam4130`** |
| Stage 2 Sodom entry | **`Boss2`** — HP **37** / L1 / cam **4864** / area **3** / `0x11E0=01` |
| Stage 2 Sodom drawn | **`Boss2_Drawn`** — HP **37** / cam≈5097 / `0x11E0=03` / boss HP **44** |
| Stage 2 Sodom mid chip | **`Boss2_Mid_b42_p37`** — after 2 spaced-Y hits + LEFT flee (dx≥150) |
| Stage 2 Sodom post-throw | **`Boss2_Mid_b1_p37`** — after UP+Y 40-dmg grab (41→1, dx≈−118) |
| Stage 2 clear (Sodom UF) | **`Stage2_Clear`** — HP **37** / L1 / cam≈5198 / boss HP **254** UF |
| Stage 3 entry (West Side) | **`Stage3`** — HP **80** / L1 / cam **619** / round **02** / area **00** |
| Stage 3 wave1 clear (prefer) | **`Stage3_Clear_w1_cam640`** — HP **80** / L1 / cam **640** (0 pdmg) |
| Stage 3 wave2 mid | **`Stage3_Mid_w2_p66_cam640`** — HP **66** / L1 during dual |
| Stage 3 wave2 clear | **`Stage3_Clear_w2_cam640`** — HP **31** / L1 / cam **640** (from Mid) |
| Stage 3 wave3 chip | **`Stage3_Mid_w3_chip_p31_e58_cam640`** — dual leftover HP5+53 behind |
| Stage 3 wave3 crumb | **`Stage3_Mid_w3_e25_p24_cam640`** / **`…_e43_p31`** — 1v1 finish |
| Stage 3 wave3 clear (prefer) | **`Stage3_Clear_w3_cam640`** / **`…_hp31`** — HP **31** / L1 / cam **640** unlocked |
| Stage 3 wave4 clear (Andore) | **`Stage3_Clear_w4_hp31_cam640`** — HP **31** / L1 / cam **640** (Andore 216 edge-JD; right-edge — dual already spawning) |
| Stage 3 wave5 dual entry | **`Stage3_Mid_w5_entry_p31_cam640`** — HP **31** / sx≈110 after JD-left from Clear_w4 |
| Stage 3 wave5 dual chip | **`Stage3_Mid_w5_chip_p31_w59_t142_cam640`** (prefer) / **`…_w80_…`** / **`…_w72_…`** — weak chipped, tough 142 |
| Stage 3 wave5 deep chip | **`Stage3_Mid_w5_chip_p20_w22_t142_cam640`** / **`…_p9_w14_…`** — finish crumbs (fragile HP) |
| Stage 3 wave5 true 1v1 | **`Stage3_Mid_w5_true1v1_p60_e142_cam640`** — weak UF; LEFT+Y wait-KD |
| Stage 3 wave5 clear (prefer) | **`Stage3_Clear_w5_real_p48_cam640`** / **`…_hp48`** — verified empty 60f @HP48 |
| Stage 3 far (area0 softlock) | **`Stage3_Far_hp50_L1_cam931`** — poke CLEAR_AREA → Area1 |
| Stage 3 Area1 entry | **`Stage3_Area1_hp50_L1_cam2560`** — HP50 / area1 / cam2560 |
| Stage 3 Area1 best mid (prefer) | **`Stage3_Area1_mid_p70_e101_cam2560`** — face-Y chip **250→101** (~149 dmg) with heal pokes |
| Stage 3 Area1 other mids | `…_e109` / `…_e142` / `…_e197` / older `…_e187` / `…_e189` |
| Stage 3 Area2 / Boss3 (dev) | **`Boss3`** cam **3072** / `0x11E0=01` — via Area1 kill + CLEAR_AREA (legit HP250 kill still open; force-map with `--force-enemy-hp`) |
| Stage 3 wave5 dual mid | **`Stage3_Mid_w5_p31_e238_cam640`** / **`…_p20_e238`** — dual 142+96 in fight (pre-chip) |
| Stage 2 far (scroll softlock) | old **`Stage2_Clear_w*_cam848`** ghost-free — do not use |
| Stage 2 far clear (poisoned) | **`Stage2_Clear_w1_cam844_threat`** — HP **80** / lives 2 / cam **844** (HP0 + living HP148 misread as UF) |
| RAM map (player / enemies) | confirmed; combat = status `0x03` (+ near `0x01` spawn **living or UF**) + (HP 1–**224** living **or** HP0/UF≥225 ghost) + on-screen |
| Segment policy (fight / walk) | `final_fight/policy.py` + `snes_oneshot.combat` (kick-band retreat, tough patient, punch-only alley, post-unlock gap=6, door **park-bait≤50** / **JD>50** + Damnd **spam-Y**; **subway area0** always JD kick-band + dual **focus HP148** + face-dir Y + **behind walk_past** + **cam≥840 scroll_mash**; **area1+** mid-screen hit-and-run / near-target / no unlock-mash / scroll_edge sx>170 / plant ghosts; **area1+ scroll** plain RIGHT + past far-behind; **area2** JD-pass under cam3915 + stall JD-left close + face-Y punch-band / no area2 scroll_edge hijack / ultra-dual space sx≤100 or tough adx<55 / sx≥55) |
| Multi-wave chain | `WaveChainTracker` + per-wave `damage_dealt` / `player_damage` |
| Slum alley unlock | **yes** — from `Stage1_Room1_Healthy`, cam **1536→1729** |
| Damnd / Thrasher | **spawned + drawn + HP-underflow** — `01`→`03` HP44→underflow |
| Damnd defeated / stage clear | **partial** — kill-frame `Stage1_Clear`; **natural `0x0CD2` / CLEAR_ROUND not observed** (see bridge) |
| Stage 2 (subway) reached | **yes** — via CLEAR_AREA bridge; Sodom cleared |
| Stage 3 (West Side) reached | **yes** — via CLEAR_AREA → Break Car → `Stage3` |
| Full stage 1 / continuous run | later (Room1→Damnd one-shot still dies mid post-unlock) |

## Current milestone

**Stage 3 West Side — wave5 cleared (verified); Area1 @cam2560; Boss3 open.**

### Stage2 → Stage3 bridge

`Stage2_Clear` is Sodom HP-underflow at `game_status=06` with
**`0x0CD2=00`** (same as Damnd). CLEAR_AREA poke runs:

1. Load `Stage2_Clear`
2. `env.set_value("game_status", CLEAR_AREA)` (`0x08`)
3. Natural pipeline: `0x08` → `0x0A` → **Break Car** (`round=06`,
   `game_status=0x0E` bonus) → open West Side (`round=02` area=00) →
   play (`0x06`) at cam≈256, HP refreshed to **80**
4. Walk right until first thug engage dx≤110 → save **`Stage3`**
   (cam **619**, HP80, L1)

Evidence: `recordings/stage3_bridge_probe/`, `recordings/stage3_push_final/`,
`scripts/stage3_advance.py`. Prefer continuous from `Stage2_Clear` for
wave1 (loading `Stage3.state` has ~1px engage drift and dies).

### Stage 3 combat notes

- First lock cam **640**; wave1 dual peaks **78+91**. **JD-left when
  sx>170** (not in punch band) keeps mid-screen — clear @**HP80 / 0
  pdmg** (`Stage3_Clear_w1_cam640`). Old right-edge park (sx≈232) took
  pdmg≈49 → HP31 and failed wave2 entry.
- Post-clear: ~90f **`west_w2_entry`** B+LEFT to sx≈90, then mid-screen
  space / pulse-Y / JD (`west_pack_*`). Mid save
  **`Stage3_Mid_w2_p66_cam640`**. Wave2 clear from Mid @**HP31**
  (`Stage3_Clear_w2_cam640`). Continuous Stage2_Clear still dies mid
  wave2 after Mid (same dual); resume Mid to finish.
- **Wave3** (cam640 sandwich): front + behind; chip mid
  **`Stage3_Mid_w3_chip_p31_e58_cam640`** → crumb mash →
  **`Stage3_Clear_w3_hp31_cam640`** @**HP31** unlocked (prefer over old
  HP15). Do not clobber preferred mids/clears on resume.
- **Wave4 Andore:** living HP **216** (was misread as UF at max192).
  `ENTITY_HP_MAX=224`. Right-edge JD kick band clears Andore @**HP31** →
  **`Stage3_Clear_w4_hp31_cam640`** (player sx≈232; HP142 already
  spawning). Andore park is HP≥200 only (not dual 142+96).
- **Post-w4 dual 142+96:** Clear_w4 right-edge chips immediately — JD-left
  ~90f to mid → prefer **`Stage3_Mid_w5_entry_p31_cam640`** or chip
  **`Stage3_Mid_w5_chip_p31_w59_t142_cam640`**. Entry: ~55f setup walk-past
  (DOWN toward smaller memory Y) then **alt60_3**. Chip resumes **skip**
  setup55. **Working finish (heal poke):** from crumb
  `…_p9_w14…` with `--heal-hp 60|70` + `--w5-tactic split` → weak dies →
  **`Mid_w5_true1v1_p60_e142`**. Then grounded **LEFT+Y** (not JD) with
  **wait mid on st01 flyaway** (dx≲−70) — chips 142→0 through KD windows.
  Verified clear: **`Stage3_Clear_w5_real_p48_cam640`** (living=0 for 90f;
  UF254 corpse still threats — plant-punch before scroll or HP chips).
  Do **not** trust brief st01 empties (`Clear_w5_p22` is false — tough
  returns at HP102). Softlock cam**931** (not 990): CLEAR_AREA →
  **Area1 cam2560** (`Stage3_Area1_hp50_L1_cam2560`). Old
  **`Clear_w5_hp13`** poisoned. Heal poke documented when used; prefer
  natural food if found.
- Softlock guard: cam≥**920** area0 no living → CLEAR_AREA.

### Stage 3 Area1 (HP≈250) — in progress

Entry: **`Stage3_Area1_hp50_L1_cam2560`**. First living thug peaks
**HP≈250** behind the player (dx≈−52). Probe:
`scripts/stage3_area1_probe.py`.

| Finding | Detail |
|---------|--------|
| Continuous `LEFT+Y` | **0 damage** (animation lock) |
| Face then pulsed `Y` | Chips ~5/hit; first hit ~f120–140 after spawn |
| Best chip (heal assist) | **250→101** (~149 dmg) → mid `…_p70_e101_cam2560` |
| Holding LEFT while punching | Walks into left gutter → death |
| Post-kill ghost | st=03 HP0 still chips — plant-punch before scroll |
| Post-kill scroll | Camera stays 2560; **CLEAR_AREA** → room2 cam3072 / **Boss3** undrawn |
| Legit full kill | **Open** (needs ~250 dmg without dying) |

Dev-only Boss3 map (not a Clean segment claim):

```bash
uv run python final_fight/scripts/stage3_area1_probe.py --force-enemy-hp 5
# → Stage3_Area1_Clear + Boss3 (bst=01, cam3072)
```

Evidence: `recordings/stage3_area1_facey_exact/`,
`recordings/stage3_area1_force_map/`, `recordings/stage3_area1_probe/`.

**Stage 2 subway — Sodom defeated; `Stage2_Clear` saved (HP37 / UF).**

Kill from cold **`Boss2_Drawn`**: spaced **UP+Y** @dx≤65 chips 44→41, then
a grab/throw deals **40** (41→1 @dx≈−118); mash grab dirs
(`UP/DOWN/LEFT/RIGHT+Y`, `Y`, `B+Y`) → HP underflow (~254). Repro **3/3**.
`0x0CD2` stays 0 (same as Damnd). Evidence:
`recordings/sodom_upy_finish/`, `scripts/sodom_probe.py --mode kill`.

Prior blockers: plain spaced-Y ~1/hit; chains `a1≥8` one-shot @dx≈80–90;
cold LEFT flee saves Mid_b42 but woken Mid matches flee. JUMP/duck/throw-far
/vertical park alone did not finish.

Scroll chip **54→38** was RIGHT+Y mash at cam≈2523 (no living parse) plus
parking into a cam2488 far-behind HP134. Fix: area1+ unlocked scroll uses
plain **RIGHT** and keeps scrolling past far-behind leftovers into the
cam2561 CLEAR_AREA poke → **`Stage2_Area2_hp54_L1_cam3840`**.

Area2: front punches often whiff at sx≈136. **Early leftover open (prefer
e69 @HP54):** 90f `B+RIGHT` then grounded **toward+Y** (slot-tracked) while
cam<3960 — kills mid-HP 1v1 + dual, plant UF ghosts → clean
**`Stage2_Area2_clear_hp54_L1_cam3968`**. (HP≤8 crumb path still works from
e3 @HP15.) Next HP134 pack clears @**HP37** (`Stage2_Clear_w2_cam3968_threat`).
Cam**4130** softlock: CLEAR_AREA → area **2→3** cam≈4864 / `0x11E0=01` →
**`Boss2`**. Evidence: `recordings/stage2_push_area2_e69j/` / `…_e69i/` /
`stage2_push_boss2_b/`.

### Stage1 → Stage2 bridge

`Stage1_Clear` is Damnd HP-underflow at `game_status=06` with **`0x0CD2=00`**.
Idling / fleeing the corpse never reaches `CLEAR_ROUND` (`0x0A`) — the
underflow ghost keeps status `03` and chips until continue (`0x12`).

**Working segment bridge** (`scripts/stage2_advance.py`):

1. Load `Stage1_Clear`
2. `env.set_value("game_status", CLEAR_AREA)` (`0x08`)
3. Natural pipeline: `0x08` → `0x0A` → open round (`0x02`/`0x04`) with
   **`round=01` area=00** → play (`0x06`) at cam≈256, HP refreshed to **80**
4. Walk right until first thug enters engage dx≤110 → save **`Stage2`**

Until Damnd death sets `0x0CD2=01` legitimately, treat HP-underflow as a
kill-frame only and use the CLEAR_AREA poke to advance.

### Stage 2 combat notes

- First lock cam **537**; thug peak **34** at sx≈285 then walks left.
- Alley `edge_wait` / door `hold_left` at dx≈51 **one-shots** (Sid/J kicks).
- Policy: subway area0 (`stage>=1`, `room==0`) uses door kick-band with punch
  dx **28–38**, nudge ≤45, **always JD** the kick band.
- UF / HP0 ghosts are threats (adapter normalizes UF→0); flee dx<40 then
  plant-punch. Status `01` living spawners near cam also count.
- From `Stage2_Clear_w2_cam537`: next wave reaches **cam 844–849**. The
  “dual UF ghost” at clear is **HP0 corpse + living HP≈148** — parser
  `ENTITY_HP_MAX` was 128, so 148 was normalized to a ghost and plant-punched.
  Fix: living HP max **192**; dual-living focuses the tough thug; face-dir Y.
- **HP148 behind method:** face-Y while HP>72 and player>40; then ground
  **walk_past** across the full kick band (dx≤103) / **throw_behind**.
  JD-past at dx≈−66…−73 ate Sid/J and burned clear to HP12; ground walk
  past keeps player ~54 through chips → clear **HP67**. Evidence
  (`recordings/stage2_push_994b/` / `stage2_push_wp103/`): behind deltas
  stay at player HP54 while 148→57; wave1 clear **HP67/L1** cam848
  (pdmg 13; one life burned earlier). Prior HP12 clear:
  `stage2_push_behind7`.
- Mid-fight save: **`Stage2_Mid_hp148_p54_e108_cam847`** (L2). Prefer
  continuous from cam537 for HP67 clear. **L2 clear:**
  `Stage2_Clear_L2_hp54_cam847_threat`.
- **Cam848 ghost-free softlock:** do not wait / save ghost-free when
  cam≥840. Policy **scroll_mash** (RIGHT+Y); distant HP0 no longer hold
  `screen_locked`. Old `Stage2_Clear_w*_cam848` ghost-free are poisoned.
- **Cam994 area-0 softlock:** scroll never advances past 994. Same
  Damnd poke — `set_value(game_status, CLEAR_AREA)` — advances subway
  **area 0→1** (~31f → open-stage → cam **1792**). Saved
  `Stage2_Area1_hp60_L1_cam1792`. Cam994 pack clear:
  `Stage2_Clear_cam994_p60_L1` (arrive L1; life often burned 848→994).
- **Area1 dual-pack:** always-JD suicided (peak cam1950). Fix: area1+ uses
  mid-screen **hit-and-run** (pulse JD / retreat, sx≥55), fights **nearest
  in-band** (not far tough), plants ghosts before mash, **scroll_edge**
  when sx>170. Clears: cam1969 HP60, cam1998/2415/2454 HP54. Evidence:
  `recordings/stage2_push_area1_hr2/`, `stage2_push_sodom3/`.
- **Cam2561 area-1 softlock:** scroll never past 2561. CLEAR_AREA poke →
  **area 1→2** cam **3840**. Prefer HP54 entry
  (`Stage2_Area2_hp54_L1_cam3840`) — see scroll notes below.
- **Pre-engage scroll chip (fixed):** RIGHT+Y at cam≈2523 chips **54→38**
  with living=0. Area1+ scroll_mash is plain RIGHT. Also keep RIGHT through
  cam2488 when only a far-behind thug holds the lock.
- Peak **cam3969** area2. Dual HP112/134 → mid 1v1 e79 @ cam3891
  (`Stage2_Area2_mid_p54_e79_cam3891`). Sodom / `Boss2` not lit.

### Progress

- **Damnd report:** `recordings/survive_r1_postunlock_l2h/stage1_segment.json`
- **Door fight evidence:** `recordings/door_clear_post5/` (+ `door_jump_clear/`)
- **Stage2 advance:** `recordings/stage2_advance/stage2_advance.json`
- **Stage2 waves:** `recordings/stage2_waves/stage2_waves.json`
- **Stage2 push:** `recordings/stage2_push_994b/` (HP67) /
  `recordings/stage2_push_area1_bridge/` (area1) /
  `recordings/stage2_push_sodom3/` (cam2561) /
  `recordings/stage2_push_area2_hp54b/` (HP54 area2) /
  `recordings/stage2_push_area2_dual_d/` (112→79 behind @3924) /
  `recordings/stage2_push_area2_1v1e/` (behind face-Y e79→56) /
  `recordings/stage2_push_area2_1v1g/` / `…_e28c/` (mid/e28 deaths) /
  `recordings/stage2_push_area2_e69/` (e69→3 + dual chips) /
  `recordings/stage2_push_area2_e28/` (e28 behind resume) /
  `recordings/stage2_push_area2_e3f/` (crumb JD90+Y + dual clear) /
  `recordings/stage2_push_area2_e69j/` (early JD90 → clear HP54) /
  `recordings/stage2_push_area2_e69i/` (wave2 @HP37 → cam4130) /
  `recordings/stage2_push_boss2_b/` (cam4130 CLEAR_AREA → Boss2) /
  `recordings/leftover_kill_probe/` + `leftover_finish_probe/` /
  `recordings/leftover_kill_jd90_early/`
- **Unlock HP / lives:** **38 / 2** (`Stage1_PostUnlock_L2`)
- **Damnd spawned?** **yes** (`0x11E0=01`, cam 2304, room 2)
- **Boss.state?** **yes** — HP 40 from `Stage1_Clear_w3_cam1728`
- **Damnd HP underflow?** **yes** — `Stage1_Clear.state`
- **Natural CLEAR_ROUND / `0x0CD2`?** **no** — spaced-Y from `Boss_Drawn`
  still UF with `cd2=0`; bridge required
- **Stage2 reached?** **yes** — `Stage2.state` (round 01 / area 00 / HP80)
- **Stage2 waves cleared?** area0 packs + **area1 dual-pack** + area1 far
  clears through cam2561; **area2** entered cam3840 at **HP54**; **area2
  pack clear @HP54** (e69 early JD90); HP134 wave2 @HP37; cam4130 bridge
- **Stage2 boss (Sodom)?** **defeated** — `Boss2` / `Boss2_Drawn` →
  **`Stage2_Clear`** (player HP37, boss UF~254, cam≈5198, area3). Method:
  spaced **UP+Y** 44→41 then **40-dmg throw** 41→1 @dx≈−118 + grab-dir
  mash. Mid chip path still valid (`Boss2_Mid_b42_p37` / `…_b1_p37`).
  `0x0CD2` stays 0. Evidence:   `recordings/sodom_upy_finish/`,
  `scripts/sodom_probe.py --mode kill` (repro 3/3).
- **Stage3 reached?** **yes** — `Stage3.state` (round 02 / area 00 / HP80 /
  cam619) via CLEAR_AREA → Break Car (`round=06`) → West Side. Evidence:
  `recordings/stage3_bridge_probe/`, `recordings/stage3_push_final/`.
- **Stage3 waves?** wave1 pack clear @**HP80** cam640 (0 pdmg;
  `Stage3_Clear_w1_cam640`); mid **`Stage3_Mid_w2_p66_cam640`**; wave2
  clear @**HP31** from Mid (`Stage3_Clear_w2_cam640`); wave3 →
  **`Stage3_Clear_w3_hp31_cam640`** @**HP31**; wave4 Andore HP216 →
  **`Stage3_Clear_w4_hp31_cam640`** @**HP31** (edge-JD). Post-w4 dual
  entry **`Stage3_Mid_w5_entry_p31_cam640`**; setup55+alt60_3 chips
  weak ≈96→59 @HP31 (chip mids w80/w72/w59); deep crumbs **w22@p20** /
  **w14@p9**. **Wave5 cleared** via heal+`--w5-tactic split` → true1v1
  → LEFT+Y wait-KD → **`Clear_w5_real_p48`**; cam→931 → CLEAR_AREA →
  **Area1 cam2560** (`Stage3_Area1_hp50_L1_cam2560`). Area1 HP250 thug
  open (Boss3 not reached). Continuous still dies mid-wave2 after Mid.
  Evidence: `recordings/stage3_push_w5_1v1_waitkd2/`,
  `stage3_w5_lefty_clear/`, `stage3_push_area1_hp250/`.

- **Subway cam844 pack HP table** (from `Stage2_Clear_w2_cam537`):

| Frame-ish | Enemy | Player HP | Notes |
|-----------|-------|-----------|-------|
| spawn dual | weak 20 + **148** | 80 | 148 was misread as UF ghost |
| behind face-Y | 148→72 | 80→54 | dx≈−73; ground walk_past after |
| walk_past finish | 72→0 | **54** L2 / **67** L1 | L2: `Clear_L2_hp54`; L1: life burn |
| clear | ≥3 kills | **54**/L2 or **67**/L1 | cam847–848 threat |
| cam994 clear | pack | **60**/L1 | then CLEAR_AREA → area1 |
| area1 dual clear | 63+42 | **60**/L1 | hit-and-run; `clear_cam1969` |
| area1 far | packs | **54**/L1 | clears cam1998 / 2415 / 2454 |
| cam2561 | softlock | **54**/L1 | plain RIGHT scroll; CLEAR_AREA → area2 |
| area2 entry | — | **54**/L1 | `Stage2_Area2_hp54_L1_cam3840` |
| area2 dual | 112→79 @cam3924 then +134 | **54→41→death** | behind ~7/hit; HP134 @cam≈3928 |
| area2 mid 1v1 | e79 front dx≈31 | **54→death** | JD-scroll summons 134; front Y whiffs |
| area2 e28 chip | e28 behind dx≈−52 | **54→death** | behind close slow; 134 arrives |
| area2 e69 path | **69→3** then dual | **54→15→death** | UP+Y25 + policy; e3 stalls |
| area2 dual peak | 79+134 → **26+8** | **54** then death | sx≤155 chips; sx>170 trap |
| area2 e28 behind | 28→8 | **54→37** | face-Y @dx−52; e26 wakes |
| area2 e3 crumb | **3→UF** + dual **134→0** | **15** clear | JD90+toward+Y; `clear_hp15` |
| area2 e69 early | **69→UF** + dual | **54** clear | JD90 early; plant UF; `clear_hp54` |
| area2 HP134 wave | 134→0 | **54→37** | behind face-Y≤70; `Clear_w2` |
| area2 cam4130 | softlock e80 | **37**/L1 | CLEAR_AREA → area3 / Boss2 |
| Boss2 / Sodom | `0x11E0=01→03` HP44 | **37** | UP+Y throw kill → UF; `Stage2_Clear` |

- **Door thug HP deltas** (reproducible):

| Kill | Peak HP | Player HP | Notes |
|------|---------|-----------|-------|
| 1 | 36 | 40→40 | punches dx≈34, 0 chips |
| 2 | 60 | 40→40 | jump-dash kick band + punches |
| 3 | **64** | **40→40** | plant-punch HP0 then JD 40–103 |
| 4 | **42** | **40→40** | park-bait / retreat; no hop_in |
| 5 | **82** | **40→40** | rise Y70 + delay40 then JD |

- **Boss HP deltas** (from `Boss_PostThug5` / `Boss_Drawn`):

| Frame-ish | Boss HP | Player HP | Notes |
|-----------|---------|-----------|-------|
| draw | 0→44 | 40 | cam≈2675 / status `03` |
| hits | 44→34→25→9→237 | **40** | kite60 then spam-Y; underflow = kill-frame |
| post | 237 / st=03 | chips | `0x0CD2` stays 0; corpse hostile |

- **Boss HP deltas (Sodom from `Boss2_Drawn` — kill):**

| Frame-ish | Boss HP | Player HP | Notes |
|-----------|---------|-----------|-------|
| draw | 44 | 37 | cam≈5097 / `0x11E0=03` |
| spaced UP+Y @dx65 | 44→43→42→41 | **37** | `a1=2` chips |
| UP+Y grab/throw | **41→1** | **37** | dmg40 @dx≈−118; `Boss2_Mid_b1_p37` |
| grab-dir mash | 1→**254** UF | **37** | UP/DOWN/L/R+Y cycle; `Stage2_Clear` |
| LEFT flee (cold only) | 44→42 | **37** | `Boss2_Mid_b42_p37` dx≥150 |
| woken Mid flee / dodge | ≤39 | death | chains `a1≥8` @dx≈80–90 |

- **Blockers / next:**
  1. Continuous `Stage2_Clear`→wave2 still dies after Mid_p66 — need
     same Mid→Clear_w2 path without resume, or healthier Mid.
  2. **Area1 HP≈250** thug @cam2560 — chips player while trading;
     need healthier entry / Andore-style edge-JD / food. Boss3 open.
  3. Carry **L2** through cam994 + area bridges (848→994 and scroll
     chips still burn a life / HP).
  4. Find legitimate Damnd death that sets `0x0CD2=01` (spaced-Y still UF).
  5. Continuous Room1→Damnd one-shot still burns a life (use L2 / w3).
  6. Wave5 path still uses **heal poke** (60–70) on crumb resumes —
     try natural food / healthier Clear_w4 for unassisted clear.

## Damnd / stage addresses

| Signal | Addr | Expect |
|--------|------|--------|
| Boss status | `0x11E0` | `00` none → **`01` loaded** → **`03` drawn** |
| Boss HP | `0x11F4` | **44** at draw; underflow (~237) on kill-frame |
| Boss-dead flag | `0x0CD2` | **`01`** should end level script (TCRF); **stays 0** after UF |
| Level-end force | `0x0CD0` | write `01` forces end (stuck if boss not dead) |
| Camera | `0x0E07` | alley unlock ~1700; Damnd door ~**2304+**; draw ~**2675** |
| Round | `0x0CB0` | `00` Slum → **`01` Subway** → **`02` West Side**; bonus **`06` Break Car** between S2→S3 |
| Area / room | `0x0CB1` | Slum room `2` at boss; Subway **`00`→`01`** @cam994, **`01`→`02`** @cam2561, **`02`→`03`** @cam4130 via CLEAR_AREA; West Side starts **`00`** |
| Game status | `0x0CA0` | bonus Break Car uses **`0x0E`** (`BONUS_GAMEPLAY`) |
| Rounds cleared | `0x0CB2` | increments on clear-round; **3** at West Side entry |

## Commands

```bash
# ROM
uv run python -m snes_oneshot.setup_all_roms final_fight

# Headless boot → Stage1.state
SDL_VIDEODRIVER=dummy uv run python final_fight/scripts/boot_probe.py

# Multi-wave Stage1 chain (JSON + PNGs under recordings/)
SDL_VIDEODRIVER=dummy uv run python final_fight/scripts/run_stage1_segment.py

# Post-thug-5 → Damnd draw + kill (preferred short segment)
SDL_VIDEODRIVER=dummy uv run python final_fight/scripts/door_jump_clear.py \
  --state Boss_PostThug5 --trials 2 \
  --out-dir final_fight/recordings/door_clear_post5

# Stage1_Clear → subway Stage2 (+ early waves)
SDL_VIDEODRIVER=dummy uv run python final_fight/scripts/stage2_advance.py \
  --state Stage1_Clear --waves 4 \
  --out-dir final_fight/recordings/stage2_advance

# Stage2 mid resume (preferred)
SDL_VIDEODRIVER=dummy uv run python final_fight/scripts/stage2_advance.py \
  --state Stage2_Clear_w2_cam537 --waves 10 \
  --out-dir final_fight/recordings/stage2_push

# Area1 resume / dual-pack
SDL_VIDEODRIVER=dummy uv run python final_fight/scripts/stage2_advance.py \
  --state Stage2_Area1_hp60_L1_cam1792 --waves 16 \
  --out-dir final_fight/recordings/stage2_push_sodom

# Area1 clear → cam2561 → area2 (auto CLEAR_AREA)
SDL_VIDEODRIVER=dummy uv run python final_fight/scripts/stage2_advance.py \
  --state Stage2_Area1_clear_hp54_L1_cam2415 --waves 20 \
  --out-dir final_fight/recordings/stage2_push_area2

# Area2 resume toward Sodom (prefer HP54)
SDL_VIDEODRIVER=dummy uv run python final_fight/scripts/stage2_advance.py \
  --state Stage2_Area2_hp54_L1_cam3840 --waves 16 \
  --out-dir final_fight/recordings/stage2_push_sodom_boss

# Area2 1v1 mid (first thug ~HP79 before second spawn)
SDL_VIDEODRIVER=dummy uv run python final_fight/scripts/stage2_advance.py \
  --state Stage2_Area2_mid_p54_e79_cam3891 --waves 16 \
  --out-dir final_fight/recordings/stage2_push_area2_1v1

# Prefer e69 early JD90 open (e28 @cam3969 skips scripted JD)
SDL_VIDEODRIVER=dummy uv run python final_fight/scripts/stage2_advance.py \
  --state Stage2_Area2_1v1_p54_e69_cam3900 --waves 24 \
  --out-dir final_fight/recordings/stage2_push_area2_e69
# After HP54 clear / wave2 / cam4130 → Boss2
SDL_VIDEODRIVER=dummy uv run python final_fight/scripts/stage2_advance.py \
  --state Stage2_Area2_clear_hp54_L1_cam3968 --waves 24 \
  --out-dir final_fight/recordings/stage2_push_area2_sodom
SDL_VIDEODRIVER=dummy uv run python final_fight/scripts/stage2_advance.py \
  --state Stage2_Area2_mid_p37_e80_cam4130 --waves 8 \
  --out-dir final_fight/recordings/stage2_push_boss2
# Leftover instrument (includes jd90_toward_y)
SDL_VIDEODRIVER=dummy uv run python final_fight/scripts/leftover_kill_probe.py \
  --recipe jd90_toward_y \
  --out-dir final_fight/recordings/leftover_kill_jd90_early

# Sodom kill (UP+Y throw + grab mash) → Stage2_Clear
SDL_VIDEODRIVER=dummy uv run python final_fight/scripts/sodom_probe.py \
  --mode kill --state Boss2_Drawn \
  --out-dir final_fight/recordings/sodom_upy_finish

# Stage2_Clear → West Side Stage3 (+ early waves; prefer continuous)
SDL_VIDEODRIVER=dummy uv run python final_fight/scripts/stage3_advance.py \
  --state Stage2_Clear --waves 8 \
  --out-dir final_fight/recordings/stage3_advance

# Wave2 mid resume (after continuous Mid_p66)
SDL_VIDEODRIVER=dummy uv run python final_fight/scripts/stage3_advance.py \
  --state Stage3_Mid_w2_p66_cam640 --waves 8 \
  --out-dir final_fight/recordings/stage3_push_w2

# Post-w4 dual (prefer split+heal crumb → true1v1 LEFT+Y wait-KD)
SDL_VIDEODRIVER=dummy uv run python final_fight/scripts/stage3_advance.py \
  --state Stage3_Mid_w5_chip_p9_w14_t142_cam640 --waves 8 \
  --w5-tactic split --heal-hp 70 \
  --out-dir final_fight/recordings/stage3_push_w5
SDL_VIDEODRIVER=dummy uv run python final_fight/scripts/stage3_advance.py \
  --state Stage3_Mid_w5_true1v1_p60_e142_cam640 --waves 8 \
  --w5-tactic split --heal-hp 70 \
  --out-dir final_fight/recordings/stage3_push_w5_1v1
SDL_VIDEODRIVER=dummy uv run python final_fight/scripts/stage3_advance.py \
  --state Stage3_Clear_w5_real_p48_cam640 --waves 10 --heal-hp 50 \
  --out-dir final_fight/recordings/stage3_push_post_w5
SDL_VIDEODRIVER=dummy uv run python final_fight/scripts/stage3_advance.py \
  --state Stage3_Area1_hp50_L1_cam2560 --waves 12 \
  --out-dir final_fight/recordings/stage3_push_area1

# Wave3 from Clear_w2 / chip mid (prefer chip → Clear_w3)
SDL_VIDEODRIVER=dummy uv run python final_fight/scripts/stage3_advance.py \
  --state Stage3_Clear_w2_cam640 --waves 8 \
  --out-dir final_fight/recordings/stage3_push_w3
SDL_VIDEODRIVER=dummy uv run python final_fight/scripts/stage3_advance.py \
  --state Stage3_Mid_w3_chip_p31_e58_cam640 --waves 6 \
  --out-dir final_fight/recordings/stage3_push_w3_chip

# Cold Drawn spaced-Y chip + LEFT flee → Mid_b42
SDL_VIDEODRIVER=dummy uv run python final_fight/scripts/sodom_probe.py \
  --mode chip --state Boss2_Drawn --hits 2 \
  --out-dir final_fight/recordings/sodom_probe

# Pure policy / segment tests
uv run --frozen pytest final_fight/tests \
  snes_oneshot/tests/test_combat.py \
  snes_oneshot/tests/test_segment_runner.py -q
```
