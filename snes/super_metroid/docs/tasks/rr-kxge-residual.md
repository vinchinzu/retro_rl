## Residual — rr-kxge Dual continuous `--to ice` stabilize

### Result
**COMPOSE LANDED** (prior) — Wave→Business return + Ice pure stack on ice tip.
**Single continuous GREEN** once (`ice_r3.json` 148192f room `0xA890` beams
`0x1007`). **Dual continuous not stable** — subsequent runs RED on Business
floor climb (HJ exit or 1339 miss). **No STATUS** promote continuous Ice.

### Climb harden (this session)
| Change | Detail |
|--------|--------|
| `pos_1339` | `business_climb` param: pure 84 / continuous retry 90 (1227 hop) |
| RIGHT-biased floor recover | Avoid stock LEFT recover → HJ `0xAA41` |
| HJ door recover | Soft return from shaft when setup kisses door |
| Attempt ladder | 8/84 → 14/84 → 8/90 → 14/90 (×2) with bound LEFT setup on pos≥90 |
| Floor dump | `business_floor_pre_ice_climb` / `_wave` for offline iteration |

### Verify paste
```bash
uv run pytest snes/super_metroid/tests/test_continuous_tips.py \
  snes/super_metroid/tests/test_k4_speed_branches.py \
  snes/super_metroid/tests/test_k4_ice_scaffold.py \
  snes/super_metroid/tests/test_k4_wave_return_scaffold.py \
  snes/super_metroid/tests/test_source_states_and_ram_cache.py -q
# → 55 passed

uv run python snes/super_metroid/scripts/probe/kpdr.py pure business-to-ice-gate \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_frog_save_to_business_pure.state \
  --expect-room 0xA7DE
# → GREEN room=0xA815 xy=(1752,651) frames=3255 (×2 exact dual)

uv run python snes/super_metroid/scripts/probe/kpdr.py pure business-to-ice-gate \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_business_continuous.state \
  --expect-room 0xA7DE
# → GREEN frames=891 (elev pin)

# Continuous (not dual-stable yet):
uv run python snes/super_metroid/scripts/record/continuous.py --to ice --no-video \
  --report snes/super_metroid/recordings/ice_rN.json
```

### Acceptance
- [x] Ice tip hops include Wave→Business return + Ice pure stack (11 hops)
- [x] Pure floor Business→Gate dual green (3255f×2 after climb harden)
- [x] Elev Business→Gate still green (891f)
- [x] Single continuous power-on → Ice once (`ice_r3` 148192f 0xA890 0x1007)
- [ ] **Dual continuous green** power-on → Ice (still flaky)
- [x] Default CLI tip remains `wave`
- [x] No STATUS promote continuous Ice
- [x] Tests + residual + NIGHT_WATCH_LOG

### Continuous evidence
| Report | Result | Notes |
|--------|--------|-------|
| `ice_r1` | RED | 1339/HJ — pre-harden compose residual |
| `ice_r2` | RED | 1339/HJ |
| **`ice_r3`** | **GREEN** | **148192f room 0xA890 beams 0x1007** — business_to_ice_gate @145255 |
| `ice_r4` | RED | left Business HJ during climb |
| `ice_r5a` | RED | left Business HJ |
| `ice_dual_a` | RED | 1339_ground (stayed Business) |

Return chain continuous splits (success path, ice_r3):
wave_to_double → … → frog_save_to_business f141870 → business_to_ice_gate f145255
→ ice_gate_to_acid → ice_acid_to_snake → ice_snake_to_ice f147824.

### Residual risks
1. **Continuous floor climb is flaky (not dual).** One-knob natural entry after
   141k frames still fails mid setup / 1339→1227 on ~half of runs; HJ door kiss
   on LEFT setup remains the dominant hard fail.
2. Offline continuous floor dump (`business_floor_pre_ice_climb_wave`) is useful
   but pin varies run-to-run; pure dual pin is stable.
3. Super-directed stop@y907 LEFT-to-lip **does not** connect (y907 right platform
   drops past Super lip) — elev climb + drop remains correct geometry.

### Next action (required)
- Stabilize dual continuous `--to ice` (2/2 integrity green); optional video.
- Offline: iterate from `scratch/business_floor_pre_ice_climb_wave.state` when present.
- **Do not** STATUS-promote continuous Ice without dual continuous green.

### Non-claims
- Did not STATUS-promote continuous `--to ice`
- Did not claim dual continuous green Ice (only single ice_r3)
- Did not change default CLI tip away from `wave`
