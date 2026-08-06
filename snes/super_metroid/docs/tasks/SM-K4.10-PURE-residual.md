# Residual — SM-K4.10-PURE / rr-re9 (Pure Wave Beam PLM collect)

## Result

RED

## Files changed

- `routes/kpdr/k4_wave.py` — `play_double_chamber_to_wave` + `WAVE_BEAM_MASK=0x0001`
- `routes/kpdr/k4_norfair.py` / `registry.py` / `__init__.py` — re-export + segment id
- `scripts/probe/kpdr.py` — `double-chamber-to-wave` pure CLI
- `tests/test_k4_norfair_scaffold.py` — registry unit
- `source_states.py` — successor `post_double_chamber_to_wave_pure` pin bounds

## Verify paste

```bash
uv run pytest snes/super_metroid/tests/test_k4_norfair_scaffold.py snes/super_metroid/tests/test_k4_speed_branches.py -q
# registry units green

uv run python snes/super_metroid/scripts/probe/kpdr.py pure double-chamber-to-wave \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_single_to_double_chamber_pure.state \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_double_chamber_to_wave_pure.state \
  --expect-room 0xADAD \
  --no-red-diag
# RED: Wave door missed / blue gate not cleared from pure pin
```

## Acceptance

- [x] Discover Wave room `0xADDE` + beam mask `0x0001` (place+collect verified)
- [x] Source: pure post Single→Double (`post_single_to_double_chamber_pure`) at `0xADAD` ~(61,139)
- [ ] Pure controller GREEN to Wave PLM collect
- [x] Units + registry wired
- [x] Residual PROCESS schema
- [ ] Dual pure / graph edge promote

## Geometry (partially resolved)

| Fact | Value |
|------|------:|
| Entry | Double Chamber `0xADAD` top-left ~(61,139) post Single→Double pure |
| Upper path | hop_run → spin16_run12; high landings ~(307,92), ~(333,99), ~(375,146) |
| Blue gate | ~x410 mid-room; switch is **top** mechanism (bars may be solid scenery) |
| Past gate | solid platform ~(520,140) (place-verified) |
| Right door | red; **Supers open it** from ~(940,139); settle into Wave ~(39,139) |
| Wave chozo | hop RIGHT to ~x160+, collect at ~(171,120) |
| Wave bit | `collected_beams & 0x0001` |

## Path recipe (intended)

1. Top: hop_run RIGHT to x≈210; spin16_run12 toward gate
2. Gate: high y≲140 UP+RIGHT beam at switch; walk through to x≳480
3. Right: hop/walljump to door sill y≲180 x≳920; Super red door → Wave
4. Wave: unmorph; hop RIGHT to chozo; X collect bit 0x01; fanfare stand

## Residual risks / RED cause

- **Blue gate PLM does not open** from pure top-left pin despite confirmed beam
  (`selected_item=0`, missiles stable) and visible impact flashes on the gate
  column. Possible causes: switch-only hitbox (need precise peak y), spin-shot
  collision, or geometry/slope preventing a stable standing shot line.
- Bottom path hard-stops at x≈475 (mid wall/pit).
- Human `speed_to_ice_moat_human` ends at Double entry flash ~(238,395); pure
  settle is top-left ~(39–61,139) — different natural seat than human mid door.
- Right-half climb from place 520 reaches near door but height control fragile.
- **Verified offline:** place 940 + Super door + Wave collect → beams `0x0001`
  (not pure evidence; place skips gate).

## Next action (required)

- **Next card ID:** rr-re9 retry / SM-K4.10-PURE-GATE (one-knob: blue gate open)
- **One change:** Stabilize standing or peak shot line that trips Double Chamber
  blue gate switch from pure pin, then re-run full pure dual
- **Source state:** `scratch/post_single_to_double_chamber_pure.state`

## Non-claims

- Did not STATUS-promote / continuous compose / forge progression RAM
- Did not mark graph edge `double_chamber_to_wave` continuous
- Place-based Wave collect is **not** pure-green evidence

## Probe pin (RED)

room=0xADAD pose=81 x=475 y=369 door_transition=0 frames=2248 last_pin=post_single_to_double_chamber_pure gate not cleared; beams=0x0000
