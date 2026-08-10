## Residual — rr-kxge Dual continuous `--to ice` stabilize

### Result
**COMPOSE LANDED** — Wave→Business return (rr-vqv3) wired into ice tip
spine before Ice pure stack. **Continuous dual still RED** at Business floor
→ Super climb (natural entry). Pure floor→Gate dual GREEN. **No STATUS**
promote continuous Ice.

### Compose wire (this session)

| Layer | Change |
|-------|--------|
| SpineHop ×7 return | `wave_to_double_chamber` → … → `frog_save_to_business` tip=`ice` |
| SpineHop ×4 ice | existing `business_to_ice_gate` → … → `ice_snake_to_ice` |
| TipSpec / catalog | ICE_ONLY_HOPS now 11 hops; parent `wave`; default tip still `wave` |
| Graph | Return edges spine-emitted continuous; hand-authored `frog_save_to_business` unverified removed |
| Floor re-pin | `play_frog_save_to_business` recenters floor ~(200–240,1419); `play_business_to_ice_gate` climbs floor→elev→Super drop |

### Verify paste
```bash
uv run pytest snes/super_metroid/tests/test_continuous_tips.py \
  snes/super_metroid/tests/test_k4_speed_branches.py \
  snes/super_metroid/tests/test_k4_ice_scaffold.py \
  snes/super_metroid/tests/test_k4_wave_return_scaffold.py \
  snes/super_metroid/tests/test_source_states_and_ram_cache.py -q
# → 55 passed

uv run python snes/super_metroid/scripts/probe/kpdr.py pure frog-save-to-business \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_speedway_to_frog_save_pure.state \
  --expect-room 0xB167
# → GREEN room=0xA7DE xy=(209,1419) frames=355 (×2 exact dual)

uv run python snes/super_metroid/scripts/probe/kpdr.py pure business-to-ice-gate \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_frog_save_to_business_pure.state \
  --expect-room 0xA7DE
# → GREEN room=0xA815 xy=(1752,651) frames=3219 (×2 exact dual)

uv run python snes/super_metroid/scripts/probe/kpdr.py pure business-to-ice-gate \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_business_continuous.state \
  --expect-room 0xA7DE
# → GREEN room=0xA815 frames=891 (elev pin regression)

# Continuous dual still RED (do not STATUS):
uv run python snes/super_metroid/scripts/record/continuous.py --to ice --no-video \
  --report snes/super_metroid/recordings/ice_r1.json
```

### Acceptance
- [x] Ice tip hops include Wave→Business return + Ice pure stack (11 hops)
- [x] Pure floor Business→Gate dual green (climb path)
- [x] Elev Business→Gate still green
- [x] Continuous reaches Business via return chain (all return splits fire)
- [ ] Dual continuous green power-on → Ice
- [x] Default CLI tip remains `wave`
- [x] No STATUS promote continuous Ice
- [x] Tests + residual + NIGHT_WATCH_LOG

### Continuous RED pin (ice_r1)
- **Last good split:** `frog_save_to_business` f141870 room `0xA7DE`
- **Fail hop:** `business_to_ice_gate` floor climb via `_business_high_jump_platforms`
- **Fail reason:** `business_1227_land` / retry `business_1339_ground` → left Business into HJ `0xAA41` or floor fall
- **Final residual pin:** room=`0xAA41` (HJ) pose=2 x=466 y=139 frames=143881
  (prior attempt stayed Business floor x=158 y=1419 after 1227 miss)
- **Report:** `snes/super_metroid/recordings/ice_r1.json`

Return chain continuous frame splits (success):
wave_to_double 136851 → double_to_single 138021 → single_to_bubble 138868 →
bubble_to_farm 140468 → farm_to_speedway 140857 → speedway_to_frog 141529 →
frog_save_to_business 141870.

### Residual risks
1. **Continuous floor climb ≠ pure dual pin.** Pure ~(209,1419) climbs dual
   GREEN 3219f×2. Continuous natural entry after 141k frames fails mid
   `business_climb` (1339→1227) — colder enemies/subpixels; LEFT setup can
   still kiss HJ door on retry.
2. Stock `_business_high_jump_platforms` is Warehouse-elev tuned; Ice only
   needs Super lip y∈[880,960]. A Super-directed climb (stop at y907, LEFT
   to lip) may be more stable than full elev climb + drop.
3. Capture continuous Business floor state after `frog_save_to_business` for
   offline climb iteration (avoid full 140k power-on each try).

### Next action (required)
- **Stabilize** continuous Business floor → Ice Super (one-knob climb or
  Super-directed hop); re-run dual continuous `--to ice`.
- **Do not** STATUS-promote continuous Ice without dual continuous green.

### Non-claims
- Did not STATUS-promote continuous `--to ice`
- Did not claim dual continuous green Ice
- Did not change default CLI tip away from `wave`
