## Residual — rr-cwu / SM-BUBBLE-BAT-SPAZER

### Result
GREEN

### Files changed
- `routes/skills/policies/bubble_to_bat.py` — `SAVE_HUMAN_SEAT_X=(25,30)` (was 32);
  `DOOR_CROUCH_FRAMES=18` Phase E crouch settle
- `routes/skills/door.py` — Phase E crouch uses `DOOR_CROUCH_FRAMES` policy
- `tests/test_k4_norfair_scaffold.py` — assert seat hi≤30 and crouch≥16
- `scratch/post_rising_tide_to_bubble_continuous_spazer.state` — continuous-like
  Bubble entry capture (business→…→rising Spazer chain) for pure-first pin

### Verify paste
```bash
# Pure baseline (no Spazer beams) — still GREEN
uv run python snes/super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state \
  --no-red-diag
# exit 0 · success true · frames=2071 · room=0xB07A

# Continuous-like Spazer Bubble entry — GREEN (was RED pin xy=(453,590))
uv run python snes/super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_continuous_spazer.state \
  --no-red-diag
# exit 0 · success true · frames=2070 · room=0xB07A

# Continuous power-on Spazer tip bat_cave
uv run python snes/super_metroid/scripts/record/continuous.py --to bat_cave --no-video \
  --report snes/super_metroid/recordings/bat_cave_spazer_cwu.json
# [GREEN] tip=bat_cave frames=127806 room=0xB07A beams=0x1004
# bubble_to_bat_cave @127684 · integrity loads=0 prog=0 deaths=0

uv run pytest snes/super_metroid/tests/test_k4_norfair_scaffold.py -q
# 32 passed
```

### Acceptance
- [x] Diagnose continuous Spazer Super door fail vs pure green
- [x] One-knob pure-first (fire seat hi + Phase E crouch settle group)
- [x] Continuous-like natural-entry source documented + greened
- [x] Continuous `--to bat_cave` Super door clears under Spazer (past bat cave)
- [x] Pure green regression kept (~2071f)
- [x] Residual written
- [x] bd close + sync on success

### Residual risks
- Phase E crouch length is Geruta-phase sensitive (crouch=20 was RED while
  18/24/28 greened in isolation). If a future continuous path desyncs
  Rising Tide timing, re-pin Phase E settle.
- Fire seat `human_hi=30` rejects x=32 continuous seats; if lower path stops
  short of max-left, seat walk may thrash (watch seat_walk reasons).
- Continuous tip dual integrity / STATUS promote is planner-owned (`rr-d20`).

### Next action (required)
- **Next card ID:** rr-d20 (Continuous tip `--to speed` dual integrity)
- **One change:** dual-record speed tip under always-Spazer after bat_cave green
- **Source state:** continuous bat_cave checkpoint / power-on

### Non-claims
- Did not STATUS-promote or edit STATUS.md
- Did not dual-integrity re-record for publish
- Did not claim `--to speed` continuous green
- Did not edit continuous.py / catalog.py / progression.py

### Probe pin (if pure/geometry)
**Pre-fix continuous RED:** room=0xACB3 pose=26 xy=(453,590) max_x=453
min_y=161 mid_reached=True top_reached=False door_reached=False
standing_mid_pinned=True launched=True phase_c_hit=True supers=5 selected=2
beams=0x1004

**Root cause:** continuous-like Bubble entry seats fire at x=32 (right edge of
old human band) → Phase D miss (min_y=161 mx~267). Pure seats x=31 with high
x_sub → tops. After seat fix, Phase E baseline crouch=8 desyncs continuous
SEEK; crouch=18 aligns pure + continuous-like.

**Post-fix continuous GREEN:** room=0xB07A frames=127806 beams=0x1004
bubble_to_bat_cave@127684 integrity loads/prog/deaths zero
