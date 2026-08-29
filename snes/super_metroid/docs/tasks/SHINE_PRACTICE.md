# Shinespark practice + K6 West Ocean

Human learning gym and pure probes for Speed-booster store/spark.
**Product pure:** Moat handoff → over-ocean spark → green Super WS `0xCA08`
(natural-entry from post-Moat pin). Continuous STATUS still waits on
compose with predecessor stack.

## Scripts (findable entry points)

Leftover practice is the A/B loop + Skill owner:

| Script | Role |
|--------|------|
| `scripts/probe/kpdr.py` | **A/B loop** — `pure west-ocean-to-ws` / `pure west-ocean-to-bowling` / `compose moat-to-ws` |
| `routes/skills/shinespark.py` | Shared charge / store / activate Skill + `diagnose_trace` |
| `routes/kpdr/moat.py` | `play_moat_shinespark`; `play_moat_to_ws` compose → `0xCA08` |
| `routes/kpdr/west_ocean.py` | `play_west_ocean_over_ocean_spark` (product), `play_west_ocean_edge_spark` |
| `scripts/record/guided_human.py` | `--from west-ocean` · `--from ws-entrance` (Phantoon ship) · `--from pre-moat` |
| `scripts/record/practice_takes.py` | `--segment west-ocean-to-ws` · `--segment ws-entrance` · `--segment moat-to-ws` |

Also listed in `AGENTS.md` (Commands) and `scripts/README.md`.

## Harness buttons (never swap with VOD A/B)

| Role | Harness | Typical VOD label |
|------|---------|-------------------|
| Dash / speed charge | **B** | A |
| Jump / shine activate | **A** | B |
| Store | **DOWN** | DOWN |

## Landing Site practice pin

- Path: `custom_integrations/SuperMetroid-Snes/scratch/landing_site_speed_practice.state`
- Room `0x91F8` ~(899,1163), items `0x3105` (Speed), **not** escape-finish `0xF32F`
- Bootstrap from pre-Moat Kihunter (walk left a few rooms). Practice via:

```bash
uv run python snes/super_metroid/scripts/probe/kpdr.py pure kihunter-to-moat \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_kihunter_pre_moat_spark.state
```

Skill surface: `routes/skills/shinespark.py` (`charge_store_activate`, `diagnose_trace`).

## Human store drill (preferred if store never arms)

**Measured trap (2026-08-06 takes `ls_edge_v1`):**  
Releasing **RIGHT** while keeping **B**, or going **idle**, dumps `echoes` **4→0 in one frame**.  
Crouching after that is ordinary crouch (`pose` 39/53, `$0A68=0`) — not a shine-store.

**Correct store:** while still holding **RIGHT** (+B ok) and echoes=4, **also press DOWN**.  
`DOWN+RIGHT+B` arms store. Do **not** release direction first.

```bash
uv run python snes/super_metroid/scripts/probe/kpdr.py pure kihunter-to-moat \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_kihunter_pre_moat_spark.state
```

`diagnose_trace` lives on `routes/skills/shinespark.py` (not a gym CLI).

Units: `tests/test_shine_practice_diagnose.py`, `tests/test_shinespark_skill.py`.

### Diagnosis grades

| Grade | Meaning |
|-------|---------|
| GREEN | Spark pose + travel |
| YELLOW | Charge + store armed; activate missed |
| ORANGE | Charge only (often **late crouch** after boost death) |
| RED | Never reached echoes=4 |

## Short charge (magic-frame boost counter)

Speed Booster tracks **velocity** (dash+forward every frame) separately from the
**boost counter** (increments only on run-animation magic frames while
dash+forward are held). Echoes appear at boost-counter 4; a stored shinespark
still travels at full spark speed even if Samus was near walking speed.

| Region | Dash-only frames (forward held from 0) | Store on last |
|--------|----------------------------------------|---------------|
| NTSC | 25, 50, 70, 85 | frame 85: dash+DOWN |
| PAL | 20, 40, 60, 70 | frame 70: dash+DOWN |

**Stutter-walk** before the first magic frame shortens runway further:

* NTSC `3-4-4-4-2+3B` → min ≈ **163.2 px** (164.2 to full stop)
* PAL `3-4-3-2-3-` → min ≈ **157.7 px**

Skill API (`routes/skills/shinespark.py`):

```python
spark.short_charge_plan("NTSC", stutter=True, store_on_last=True)
spark.charge_until_boost(session, "RIGHT", mode="stutter")  # or mode="short"
spark.charge_store_activate(session, charge_mode="short", store_on_last_magic=True)
```

Controllers accept `charge_mode="full"|"short"|"stutter"`:

* `play_moat_shinespark(..., charge_mode=...)` — product **full**
* `play_west_ocean_over_ocean_spark(..., charge_mode=...)` — product **stutter**
* `play_west_ocean_edge_spark(..., charge_mode=...)` — bowling practice

```bash
# Product: Moat handoff → over-ocean spark → Super WS 0xCA08
uv run python snes/super_metroid/scripts/probe/kpdr.py pure west-ocean-to-ws \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_moat_west_ocean_spark.state --headed

# Edge-spark bowling (practice; not Phantoon entry)
uv run python snes/super_metroid/scripts/probe/kpdr.py pure west-ocean-to-bowling \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_moat_west_ocean_spark.state \
  --place-x 350 --place-y 550 --headed
```

### Measured (emulator, 2026-08-10)

| Pin | Mode | frames | delta_x | notes |
|-----|------|-------:|--------:|-------|
| West Ocean spit ~(350,587) | full | 90 | **417** | continuous RIGHT+B |
| West Ocean spit | short | 86 | **195** | magic 25/50/70/85 |
| West Ocean spit | stutter | 86 | **141** | below wiki 163 px min |
| Landing Site practice | full | 90 | 460 | |
| Landing Site practice | short | 86 | 217 | |
| Landing Site practice | stutter | 86 | 156 | |
| West Ocean edge `--charge-mode stutter` | — | — | — | **GREEN** → bowling `0xC98E` |
| West Ocean edge `--charge-mode short` | — | — | — | **GREEN** → bowling `0xC98E` |
| West Ocean **over-ocean** `pure-ws` stutter | — | — | — | **GREEN** → WS `0xCA08` ~(57,139) gs=8 ×2 |
| West Ocean **over-ocean** `pure-ws` short | — | — | — | **GREEN** → WS `0xCA08` ×2 |
| Moat hop after short/stutter | — | — | — | **RED** (stalls ~x555); keep product `full` |

`--store-on-last` on stutter: `$0A68=179` armed on frame 85 with echoes=4.

Harness: **B**=dash on magic frames only; forward is RIGHT/LEFT.

## Moat pure (GREEN, pin-only)

```bash
uv run python snes/super_metroid/scripts/probe/kpdr.py pure kihunter-to-moat \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_kihunter_pre_moat_spark.state
# → scratch/post_moat_west_ocean_spark.state  West Ocean 0x93FE ~(49,1163)
```

Pure Moat spark residual purged (hop closed green; A/B loop above). Product default
charge remains **full** continuous `RIGHT+B`; pass `charge_mode="short"` /
`"stutter"` on `play_moat_shinespark` when the left runway is cramped.

## West Ocean over-ocean → WS (GREEN pure, product)

sm-json *Over Ocean Spark* (node 13 → green Super door 5). Natural Moat
handoff — **no free-place**, **no bottom swim as primary**.

```bash
uv run python snes/super_metroid/scripts/probe/kpdr.py pure west-ocean-to-ws \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_moat_west_ocean_spark.state
# GREEN room=0xCA08 ~(57,139) gs=8  charge=stutter ~627f  (dual ×2)
# saved scratch/post_west_ocean_ws_spark.state

# Human free-record from product WS pin (ship rooms / Phantoon approach)
uv run python snes/super_metroid/scripts/record/guided_human.py \
  --from ws-entrance --route ws-entrance --name ws_ship_human
uv run python snes/super_metroid/scripts/record/practice_takes.py \
  --segment ws-entrance --series ws_ship_v1
```

| Fact | Value |
|------|--------|
| Source | `scratch/post_moat_west_ocean_spark.state` ~(49,1163) |
| Charge | **stutter** (product); `short` also dual-green; `full` RED |
| Store / hop | crouch-store → hop 4f A → pre-stand UP 4f → RIGHT spark |
| Spark land | ~(2011,1163) green Super door lip |
| Super open | select weapon 2 + RIGHT+X pressure |
| Settle | **`0xCA08`** ~(57,139) gs=8 |
| Controller | `play_west_ocean_over_ocean_spark` / `play_west_ocean_to_ws` |

Do **not** STATUS-promote until continuous power-on compose (K5→Moat approach + this chain).

## Moat → West Ocean → WS compose (`chain-ws`)

Room-dispatch product path for continuous-style natural entry from standing
pins (not power-on continuous tip yet):

| Step | Controller | Charge |
|------|------------|--------|
| Kihunter/Moat → WO | `play_moat_to_west_ocean` / `play_moat_shinespark` | **full** |
| WO → WS `0xCA08` | `play_west_ocean_to_ws` | **stutter** |
| Compose | `play_moat_to_ws` | both |

```bash
# Product pin or human Moat end
uv run python snes/super_metroid/scripts/probe/kpdr.py compose moat-to-ws \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_kihunter_pre_moat_spark.state
uv run python snes/super_metroid/scripts/probe/kpdr.py compose moat-to-ws \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/alpha_pb_to_moat_human_end.state

# Ship free-record → Phantoon approach (refresh pin with compose first)
uv run python snes/super_metroid/scripts/record/guided_human.py \
  --from ws-entrance --name ws_ship_human
uv run python snes/super_metroid/scripts/record/practice_takes.py \
  --segment ws-entrance --series ws_ship_v1
```

| Source | Result (2026-08-10 dual) |
|--------|--------------------------|
| `scratch/post_kihunter_pre_moat_spark.state` | **GREEN** `0xCA08` ~(57,139) gs=8 ×2 (~1348f) |
| `scratch/alpha_pb_to_moat_human_end.state` | **GREEN** `0xCA08` ~(57,139) gs=8 ×2 (~4435f leave+clear) |
| `scratch/post_moat_west_ocean_spark.state` | over-ocean only (`kpdr.py pure west-ocean-to-ws`) |

Not continuous STATUS — pin compose only. Recording handoff: `--from ws-entrance`.

## West Ocean edge spark (GREEN pure, bowling practice only)

VOD recipe (screenshots under `debug/west_ocean_spark/`):  
run to water edge → **store at edge** → turn back a few steps → hop up → spark right.

```bash
uv run python snes/super_metroid/scripts/probe/kpdr.py pure west-ocean-to-bowling \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_moat_west_ocean_spark.state \
  --place-x 350 --place-y 550
# GREEN room=0xC98E Bowling Alley (mid-right blue) — not product WS
```

| Fact | Value |
|------|--------|
| Spit bootstrap | place ~(350,550) → settle ~(350,587) |
| Edge | ~(909,472) echoes=4 |
| Params | back=8, hop=4, aim RIGHT |
| Door hit | **`0xC98E` Bowling Alley** — practice only |

Notes: `debug/west_ocean_spark/USER_RECIPE.md`.

## Follow-up work (not done)

1. **Human store mastery** — `kpdr.py pure` + `shinespark.py` until `$0A68` arms consistently; then full spark on LS.
2. **Natural spit climb** — only needed if re-using edge-bowling path; product WO→WS uses ocean floor.
3. **Compose** Kihunter → Moat pure → over-ocean WS; natural-entry only for STATUS.
4. Keep Moat product on `full` unless runway pin moves.

## Related docs / plan

- Plan K6: `docs/plan.md` § Ship / Phantoon / Gravity
- Skill module: `routes/skills/shinespark.py`
- Probe: `scripts/probe/kpdr.py`
