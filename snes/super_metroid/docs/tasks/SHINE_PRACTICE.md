# Shinespark practice + K6 West Ocean follow-up

Human learning gym and pure probes for Speed-booster store/spark. **Not**
continuous STATUS evidence until natural-entry WS is pure.

## Scripts (findable entry points)

| Script | Role |
|--------|------|
| `scripts/probe/shine_practice.py` | **Landing Site human gym** — multi-take record + diagnose; **store drill** |
| `scripts/probe/moat_spark_watch.py` | Kihunter → Moat pure hop/spark (GREEN) |
| `scripts/probe/west_ocean_spark.py` | West Ocean edge→store→hop→spark pure (mid-right door) |
| `scripts/probe/landing_shine_practice.py` | Dual-track measure / bootstrap LS pin / diagonal proof |
| `routes/skills/shinespark.py` | Shared charge / store / activate skill surface |
| `routes/kpdr/moat.py` | `play_moat_shinespark` |
| `routes/kpdr/west_ocean.py` | `play_west_ocean_edge_spark` |
| `scripts/record/guided_human.py` | `--from west-ocean` free-record toward WS |
| `scripts/record/practice_takes.py` | `--segment west-ocean-to-ws` multi-take |

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
- Bootstrap from pre-Moat Kihunter (walk left a few rooms):

```bash
uv run python snes/super_metroid/scripts/probe/shine_practice.py bootstrap
# or
uv run python snes/super_metroid/scripts/probe/landing_shine_practice.py bootstrap
```

## Human store drill (preferred if store never arms)

**Measured trap (2026-08-06 takes `ls_edge_v1`):**  
Releasing **RIGHT** while keeping **B**, or going **idle**, dumps `echoes` **4→0 in one frame**.  
Crouching after that is ordinary crouch (`pose` 39/53, `$0A68=0`) — not a shine-store.

**Correct store:** while still holding **RIGHT** (+B ok) and echoes=4, **also press DOWN**.  
`DOWN+RIGHT+B` arms store. Do **not** release direction first.

```bash
# Bot charges + holds RIGHT+B after e=4; you only press arrow DOWN
uv run python snes/super_metroid/scripts/probe/shine_practice.py drill

# Free multi-take (F5=save+diagnose+reload)
uv run python snes/super_metroid/scripts/probe/shine_practice.py human --series ls_edge_v1

# Re-diagnose a take
uv run python snes/super_metroid/scripts/probe/shine_practice.py diagnose \
  snes/super_metroid/tasks/shine_practice/ls_edge_v1/take03.json

# Bot full horizontal spark demo
uv run python snes/super_metroid/scripts/probe/shine_practice.py demo
```

Takes land under `tasks/shine_practice/<series>/takeNN.json` with `diagnosis` + WRAM trace.

Units: `tests/test_shine_practice_diagnose.py`, `tests/test_shinespark_skill.py`.

### Diagnosis grades

| Grade | Meaning |
|-------|---------|
| GREEN | Spark pose + travel |
| YELLOW | Charge + store armed; activate missed |
| ORANGE | Charge only (often **late crouch** after boost death) |
| RED | Never reached echoes=4 |

## Moat pure (GREEN, pin-only)

```bash
uv run python snes/super_metroid/scripts/probe/moat_spark_watch.py pure \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_kihunter_pre_moat_spark.state
# → scratch/post_moat_west_ocean_spark.state  West Ocean 0x93FE ~(49,1163)
```

Pure Moat spark residual purged (hop closed green; probe above).

## West Ocean edge spark (GREEN pure, wrong door for Phantoon)

VOD recipe (screenshots under `debug/west_ocean_spark/`):  
run to water edge → **store at edge** → turn back a few steps → hop up → spark right.

```bash
uv run python snes/super_metroid/scripts/probe/west_ocean_spark.py pure
# GREEN room=0xC98E Bowling Alley (mid-right blue)
# saved scratch/post_west_ocean_door_spark.state
```

| Fact | Value |
|------|--------|
| Spit bootstrap | place ~(350,550) → settle ~(350,587) |
| Edge | ~(909,472) echoes=4 |
| Params | back=8, hop=4, aim RIGHT |
| Door hit | **`0xC98E` Bowling Alley** — **not** green WS `0xCA08` |
| Lower green WS | underwater — no Speed charge without Gravity |

Notes: `debug/west_ocean_spark/USER_RECIPE.md`.

Human free-record from post-Moat pin:

```bash
uv run python snes/super_metroid/scripts/record/guided_human.py \
  --from west-ocean --route west-ocean-to-ws --name west_ocean_to_ws_human
```

## Follow-up work (not done)

1. **Human store mastery** — use `drill` until `$0A68` arms consistently; then full spark on LS.
2. **Natural spit climb** — replace free-place bootstrap in West Ocean pure with climb from Moat handoff lower-left water.
3. **WS green Super door `0xCA08`** — underwater path + select Super + open; or Gravity later. Not the mid-right bowling spark.
4. **Compose** Kihunter → Moat pure → West climb → door path; natural-entry only for STATUS.
5. Optional: HUD flash / audio on echoes=4 in `shine_practice human`.

## Related docs / plan

- Plan K6: `docs/plan.md` § Ship / Phantoon / Gravity
- Skill module: `routes/skills/shinespark.py`
- Probe: `scripts/probe/moat_spark_watch.py`
