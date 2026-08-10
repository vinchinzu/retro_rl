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

## Next experiments (do not re-run)

**Do not:** goblin-solid, pure-RIGHT only, “LL never spawns”, hold-B spam without pulse.

**Do:**

1. **Object-solid decode** — `aobject_tsa=$4E0`, flag bits after kill (128→192), type 118 role;  
   stand pose: compare sy vs by when human/TAS is on cloud (likely feet offset, not sy==by).
2. **Land window** — after kill, approach from **above** with feet-on-top geometry (sy ≈ by − sprite_h), not body-center align.
3. Chain mapset 5–6 LLs once first cloud stand freezes a state.
4. Freeze AirScreen2→5 (3/3) only after stand+ride advances cam.

## Smoke

AirScreen2 → target 4 still expected GREEN (~502f). Units: `uv run pytest nes/mega_man_2/tests -q`.
