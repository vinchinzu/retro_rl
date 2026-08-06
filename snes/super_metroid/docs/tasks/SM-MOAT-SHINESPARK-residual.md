## Residual — rr-hhj / SM-MOAT-SHINESPARK

### Result
GREEN

Pure store→spin→UP unspin→spark + post-spark `RIGHT+X` door open reaches West
Ocean `0x93FE` from pin (probe `hop` + controller `pure` both green).

### Files changed
- `routes/kpdr/moat.py` — hop-unspin-activate band + post-spark blue-door open
  into West Ocean (`play_moat_shinespark`)
- `routes/skills/shinespark.py` — reusable charge/store/activate skill surface
- `routes/skills/__init__.py` — export shinespark helpers
- `scripts/probe/moat_spark_watch.py` — hop / pure / sweep / watch / record probe
- `scripts/probe/landing_shine_practice.py` — Landing Site dual-track measure
- `tests/test_shinespark_skill.py` — unit tests for skill knobs (no emulator)
- `scratch/post_kihunter_pre_moat_spark.state` — pin (Kihunter clear, local)
- `scratch/post_moat_west_ocean_spark.state` — West handoff after green hop
- `docs/plan.md` — K6 pure Moat checkbox marked done (pure only)

### Verify paste
```bash
uv run python snes/super_metroid/scripts/probe/moat_spark_watch.py hop \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_kihunter_pre_moat_spark.state
# exit 0
# boot  xy=(39,139) room=0x948C
# boost xy=(503,178) echoes=4
# store armed=$0A68=179 xy=(511,192) pose=53
# hop stand=4 run=2 hop_f=14 … unspin=UP@3 act=RIGHT+A
# travel moat=True west=True moat_max_x=490 …
# final room=0x93FE xy=(39,1163) pose=11 spark=0
# GREEN

uv run python snes/super_metroid/scripts/probe/moat_spark_watch.py pure \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_kihunter_pre_moat_spark.state
# exit 0 · GREEN room=0x93FE xy=(49,1163) frames=721
# saved …/scratch/post_moat_west_ocean_spark.state

uv run pytest snes/super_metroid/tests/test_shinespark_skill.py snes/super_metroid/tests/test_moat_scaffold.py -q
# 10 passed
```

### Acceptance
- [x] Read residual + moat.py + shinespark skill for failure mode
- [x] Pure green: store→spin→spark clears Moat into West Ocean from pin
- [x] Post-West scratch captured (`post_moat_west_ocean_spark.state`)
- [x] Residual updated to PROCESS schema; stale “not West yet” superseded
- [x] Units for skill knobs (`test_shinespark_skill.py`)
- [x] Close rr-hhj + bd sync + commit (no push)

### Residual risks
- Spark still **dies mid-Moat** (~x475); West is spark + blue-door open, not a
  continuous spark through the door.
- Natural-entry from predecessor segment not re-verified this round (pin-only).
- Continuous STATUS / continuous route not claimed.
- Optional: elevated screenshot-faithful charge (path B) if full spark-through
  door (no walk) is required later.
- Optional refactor: `moat.py` → `store_then_spin_unspin_activate` from skill.

### Next action (required)
- **Next card ID:** PLANNER-GATE (K6 Moat → West Ocean → Wrecked Ship by play)
- **One change:** natural-entry / continuous compose after pure predecessor, or
  pure West Ocean room traversal — planner picks
- **Source state:** `scratch/post_moat_west_ocean_spark.state` (West `0x93FE`)

### Non-claims
- Did **not** STATUS-promote continuous Kihunter→Moat→West
- Did **not** claim pure spark opens Moat right door without X
- Did **not** re-verify natural-entry from Speed/Ice predecessor
- Did **not** edit continuous.py / catalog tip wiring

### Probe pin (if pure/geometry)
**Source pin:** room=`0x948C` pose clear xy≈(39,139) door_transition=0
last_pin=`scratch/post_kihunter_pre_moat_spark.state`

**Green path:**
```text
RIGHT+B until sc≥4 pose 9 y≥170     # ~x503 y178 trench
DOWN ×18                             # arm $0A68≈179
idle stand ~4f
RIGHT+B micro-run ~2f                # leave crouch pose 39 → pose 9 (REQUIRED)
RIGHT+B+A hop ~13–16f                # spin over x555; hop_a continuous
UP ~3–4f                             # unspin; often → pose 199 windup
RIGHT+A activate + travel            # pose 201 horizontal into Moat
# Moat: spark to ~x475 (timer stuck 14, vx→0) then dies ~moat_f 108
RIGHT+X pulse walk                   # open blue door → West Ocean 0x93FE
```

**Post-green West:** room=`0x93FE` xy≈(39–49,1163) pose=11/1 spark=0 frames≈721
(controller) / hop travel≈141f after activate · handoff
`scratch/post_moat_west_ocean_spark.state`

**Best params (band):** stand 4–8 (def 4); micro_run **≥2** (0 fails: A from
crouch early-shines into wall); hop_f 13–16 (def 14); hop_a_f −1; unspin **UP**
1–5f (def 3); travel RIGHT+A; door_open post-spark on.

### Harness buttons (do not swap)

| Role | Harness | VOD / SM default |
|------|---------|------------------|
| Dash / speed charge | **B** | A |
| Jump / shine activate | **A** | B |
| Store | DOWN | DOWN |
| Shoot (blue door) | **X** | X |

### Geometry facts (measured)

| Fact | Value |
|------|-------|
| Trench charge | x≈503 y≈178 pose **9** sc=4 (~90f `RIGHT+B`) |
| Crouch-store from pose 9 | arms `$0A68`≈179 → pose 53; settles pose 39 |
| **Do not store elevated** | require **y≥170**; elevated hop jams ~x619 |
| Wall trap | x≈555 — pure horizontal spark dies here |
| Door lip Kihunter | x≈720–768 |
| Moat room | `0x95FF`; spark corridor jam max_x≈**475** |
| West door | blue; need **RIGHT+X** after spark dies |
| West settle | `0x93FE` x≈39–55 y≈1163 |

### What failed (do not re-chase blindly)

1. Trench store + pure `RIGHT+A` → pose 201 dies x≈555 wall
2. Spin over wall then store → DOWN from 166 wipes echoes
3. 1f A-taps while charging kill echo build
4. Re-charge on upper after hop — no runway left of door
5. Travel-hold variants alone still jam mx=475 — need door shoot
6. micro_run=0 → crouch `RIGHT+B+A` shines into wall @ x555
7. Store elevated y&lt;170 → jam Kihunter ~x619
8. Old yt `moat_shinespark_timing.json` unreliable vs WRAM + pin

### Quick commands

```bash
uv run python snes/super_metroid/scripts/probe/moat_spark_watch.py hop \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_kihunter_pre_moat_spark.state

uv run python snes/super_metroid/scripts/probe/moat_spark_watch.py hop --sweep \
  'stand=4,8;hop=13:16:1;run=2;unspin_f=3,4;travel=RIGHT+A'

uv run python snes/super_metroid/scripts/probe/moat_spark_watch.py pure \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_kihunter_pre_moat_spark.state

uv run python snes/super_metroid/scripts/probe/moat_spark_watch.py watch
```

### Artifacts
- `debug/moat_spark/hop_measure.json` — last green hop + Moat samples
- `debug/moat_spark/hop_height_only.json` / `hop_travel_sweep.json`
- `scratch/post_moat_west_ocean_spark.state` — West pin
- `recordings/moat_shinespark_practice_hud.mp4` (optional practice HUD)

### Shared shine skill
- Module: `routes/skills/shinespark.py`
- Probe: `scripts/probe/landing_shine_practice.py`
- Pin: `scratch/landing_site_speed_practice.state` (room `0x91F8` + Speed)
- Moat controller still room-specific open-loop (not yet on skill recipe)
