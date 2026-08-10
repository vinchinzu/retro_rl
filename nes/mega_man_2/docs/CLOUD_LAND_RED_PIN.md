# rr-54ui RED pin — Thunder Chariot land (2026-08-10 night)

## Acceptance: NOT MET

No `camera_x_screen ≥ 5`. No sustained cloud stand. M3 AirScreen2→4 still GREEN.

## Breakthrough (this session)

fpd6 residual was “~28px short in X at apex”. Clean progress:

| Metric | fpd6 residual | This session best |
|--------|---------------|-------------------|
| X at apex (no kill) | sx≈135 vs LL x≈163 (**dx≈28**) | dx≈5–7 contact class |
| Rider kill | never (HP watch wrong / no pulse B) | **Yes** — type `0x3D` 20→13→6→despawn |
| Y-meet after kill | n/a | **dx≈5–10, \|dy\|≤4** (still no stand) |
| Max prog | ~1070–1086 | ~1084–1089 class |
| Camera ≥5 | no | no |

### Rider kill (verified Clean)

- Target **`0x3D`** (`kaminari_goro_move`), not body `0x3E` (body HP stays 20).
- Buster needs **pulse** (tap B period 3–8, 1f press) — hold-B under-fires.
- Hits require **air height** (grounded shots miss high rider y≈12–16).
- 3 hits (wiki): live ≈7 dmg/hit (20→13→6→dead).
- On kill: rider slot frees; types **6** + **118** flash; body `0x3E` remains.

### Geometry residual (updated)

1. **X gap closable** by waiting for LL approach + edge jump + pulse shoot.
2. **If kill when player still high above cloud (dy≳20):** player and cloud sink at ~same rate → dy stays ~20 forever → never land.
3. **If kill near Y-meet:** best Y-meet after kill still **falls through** at dx≈5–10, dy≈0 (sy≈by).  
   Example: `c976_dx35-65` kill f122, +4f: sx128 sy49 bx140 by49 dx12 dy0 ft=0 → continue freefall.
4. Therefore residual is **deeper than X**: empty Thunder Chariot **object-solid / stand condition** not yet achieved under Clean.

### Best kill+Y-meet class recipes (from AirFanPlatform)

- Camp prog **~970–976**, jump when LL **dx 35–70**, jh **12**, hang **20–28**, B tap **period 3–4**.
- Probes: `scripts/cloud_land_probe.py`, v2–v7 one-shots under `recordings/air_post4_cloud*/`.
- Evidence JSON: `air_post4_cloud/cloud_land_grid.json`, `air_post4_cloud_v2/summary.json`,  
  `air_post4_cloud_v5/dps_search.json`, `air_post4_cloud_v6/summary.json`,  
  `air_post4_cloud_v7/summary.json`.

## Solid decode (2026-08-10 session) — still RED

Probe: `scripts/cloud_solid_decode.py` → `recordings/air_post4_cloud_solid/`.
708 Clean recipes + deep dumps + diagnostic pokes (poke = not Clean evidence).

| Field | Decode |
|-------|--------|
| `aobject_tsa` ($4E0) | **AI timer**, not solid type. Counts down (~41→0) then resets; not `tsa.asm` block enum |
| flag 128→192 | `objects_exist` → `exist\|objects_right` (facing). **Not** solid enable. Appearing-block bit `$10` never set on empty cloud |
| type 6 | Confirmed `objects_killed` death anim on rider slot (~12f then free). |
| type 118 | **Not observed** on kill path this session (prior “118 flash” likely misread). Enum = `objects_large_life_capsule` |
| Body post-kill | Stays **`0x3E`**, HP 20, tsa cycles, `ys≈255` signed motion; **no** type→platform rewrite |
| feet-on-top | With MM_H=24, **`feet_dy=0` + dx≤2** after kill achieved many times — still freefall `ft=0` |
| Co-sink | After contact, player+cloud match vertical rate; `feet_dy` locks ≈−3…−4 |
| Screen | Body often `scr=4` while cam still 3 at meet — possible collision gate |
| Diagnostic poke | Place sy=by−24, zero yspeed: still falls through empty `0x3E` (suggests solid path inactive, or needs edge-cross descent the AI never arms) |

### Geometry note

At typical kill (sy≈34, by≈48): player **top** is above cloud, but feet (sy+24) are already through cloud volume. Jump apex min_sy~34 vs cloud y≈32–50 → almost no “clearly above then drop” window. One-way solid would need a thin surface cross while descending **relative to cloud**.

## Disasm + screen-align session (2026-08-10) — still RED

Probe: `scripts/cloud_screen_align.py` → `recordings/air_post4_screen_align/`
(247 Clean recipes + focused pin_diag; 40+ kills, 0 stand, 0 cam≥5).

### Body AI (`objects_kaminari_goro` 0x3E) — lsmmega bank14 `14_19.bin`

| Finding | Detail |
|---------|--------|
| Spawn | Body AI `LDA #$3D / JSR $F159` creates rider child; stores parent link `$0120` |
| Rider attach | Rider AI: if parent type==`$3E` and exist, lock y=`by-0x14`, same x/screen |
| On rider death | **No type rewrite, no solid flag arm** in body AI — body stays `0x3E` HP20 |
| Flag bit `$08` | Body sets/clears AI phase bit (`ORA #$08` / `AND #$F7`) — **not** solid |
| Appearing-block solid | Engine solid for objects uses `objects_appearing_block=$10` (`appearing_block.asm`) — **never set** on live empty cloud |
| Full PRG scan | Only **4×** `CMP #$3E` (AI/timer only); **0×** `CMP #$3D` — no type-whitelist solid path |
| OAM geometry | `oamcoord_3e` spans y=−16…+16 around body y → cloud **top ≈ by−16** |

### Screen-align live pin

| Metric | Result |
|--------|--------|
| Kill contact window | Almost always **cam=3, body scr=4** (misaligned) |
| Best cloud-top approach | `top_dy≈−8` (feet already through estimated top) @ dx≈12, cam3 |
| When cam finally 4 | `top_dy≈−19` — already deep through cloud volume |
| Same-scr + empty body | Still freefall `ft=0`, status 3/6/7 air — **no Y lock** |
| Higher hang 40–56 | Does not create clean above-top then drop window |

### Diagnostic pokes (not Clean; prove solid path inactive)

Via fceumm state WRAM poke (base offset 93):

| Poke | Result |
|------|--------|
| `fall_top` place feet near by−16, zero yspeed | **top_dy≈+1 achieved** — still freefall, no stand |
| `fall_center` place feet near by | still freefall (prior session) |
| Force body flag `\|$08` (AI bit) | no stand |
| Force body flag `\|$10` (appearing_block) | no stand under our contact window |

**Engine residual (precise):** Under fceumm/stable-retro Clean play, empty
`0x3E` after rider kill does **not** present a working object-solid / stand path
in any tested geometry (center, top band, same-screen, force-place above top).
Body AI has no post-kill solid arm. Appearing-block collision is the only
decoded object-solid path and is never enabled on the chariot.

### Status $2C during contact

| Value | When seen |
|-------|-----------|
| 6 | freefall / air (typical kill+meet) |
| 7 | brief air variant |
| 3 | late fall / some ground-exit — **not** sustained cloud stand |
| Never | status that freezes Y to body for ≥4f with `ft` or yvar lock |

## Alt path + appear-mask session (2026-08-10) — still RED (PARTIAL)

Acceptance still **not met** (`cam ≥ 5` no). Residual child: **rr-f3nr**.

### Human / TAS routes (documented)

| Route | Requirement | Air-first Clean? |
|-------|-------------|------------------|
| Kill LL → ride Thunder Chariot ×5 | Buster only | Intended; solid residual blocks it |
| Item-1 platforms skip cloud waits | Heat first → Item-1 | **No** — `AirFanPlatform` weapons=`$00` |
| Jump/damage-boost past gap | — | **No** — max prog ~1050–1070 class; gap ~296px |

Sources: MMKB / StrategyWiki walkthroughs; TAS 2881S + megamanrta (Heat-before-Air + Item-1).

### Appear-mask disasm pin (new)

| Finding | Detail |
|---------|--------|
| Sole `LDA #$90 / STA flag,X` | appearing_block AI only (`14/14_23.bin`) |
| `ORA #$10` in full PRG bins | **0×** |
| Real appear setup | flag=`$90`, `ys` (`$640`)=`#$04`, xs/xsf from shifted pos, tsa=solid type |
| Body `0x3E` | Never arms appear; stays exist/facing only |
| Diag zero-mask force | flag\|`$10` + tsa=1 + xs=ys=xsf=ysf=0 → **global solid** under fceumm (ft=1, Y lock) — solid *path* works when configured |
| Diag localized masks | tile16 / wide_y / 14_23-style after Clean kill → still freefall (wrong mask geometry and/or AI overwrite of speed regs) |

### Engine residual (updated)

1. Empty cloud after Clean rider kill never presents stand (prior RED pin).
2. Only decoded object-solid path = appearing_block; body never configures it.
3. fceumm **can** solid via appear when fully forced (zero masks) — residual is **arming/config**, not a total emulator solid blackout.
4. No Air-first Clean alternate past s4 without cloud stand or external Item-1.

Evidence: `recordings/air_post4_altpath/` (`appear_mask_probe.json`, `appear_mask_geom.json`, `altpath_summary.json`).

## Next experiments (do not re-run)

**Do not:** goblin-solid, pure-RIGHT only, “LL never spawns”, hold-B spam without pulse,
re-grid feet_dy=0 alone, re-grid screen-align alone, re-poke fall_top/appear/flag08,
re-grid zero-mask global solid, pure X-chase (already negative).

**Do (rr-f3nr):**

1. **FCEUX/human RAM pin** — on a frame where feet **stick** on empty cloud, dump
   sy/by/`$2C`/body fl/tsa/xs/ys/xsf/ysf/cam vs freefall dumps here.
2. **Heat→Air Item-1 Clean segment** as alternate past s4 (new milestone; not Air-first).
3. If human pin shows missing RAM/type arm: implement under Clean.
4. Chain mapset 5–6 LLs only after first stand freezes a state.

## Smoke

AirScreen2 → target 4 still expected GREEN (~502f). Units: `uv run pytest nes/mega_man_2/tests -q`.
