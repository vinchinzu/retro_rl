## Residual — SM-K4.4-PURE-R18

### Result
PARTIAL (large product progress; full pure Phase D still red)

Pure natural entry now:

1. Seats **max-left fire band** (x∈[25,32]) via L-angle stationary missile clear
   (no LEFT+X walk)
2. Fires **without arm-pump** (arm-pump desyncs pure seat jump)
3. Earns **pose 132** right-wall latch + **pose 84** left-wall bounce class
4. Free-air height **min_y≈156–159** (was 228 R16/R17)

Phase D (`x≥300 ∧ y≤200`) still fails on full pure: **mx200 hard-caps ~251**
(single-WJ ceiling). Isolation proof: same recipe tops when enemy slots 0/4/6
match a golden live dump — enemy **AI/state**, not HP (zeroing HP alone does
not unlock Phase D).

### Files changed
- `routes/kpdr/bubble_mountain_mid.py` — fire only from human seat band; seat
  via `bubble_seat_max_left_fire`; arm-pump off; lip fallback after seat tries
- `routes/kpdr/bubble_mountain_primitives.py` — `bubble_seat_max_left_fire`,
  L-angle stationary clear, `bubble_walljump_second_left_wall`
- `routes/kpdr/bubble_mountain_params.py` — `SAVE_ARM_PUMP=False`, R18 WJ2
  L14/R6/follow40, `WJ2_LEFT_*`, longer stationary X
- `scripts/probe/bubble_r18_velocity_dump.py` — velocity dump CLI
- `tests/test_k4_norfair_scaffold.py` — R18 params / skills
- Tip boards + this residual

### Load-bearing facts

| Fact | Detail |
|------|--------|
| Natural entry | `post_rising_tide_to_bubble_pure` ~(54,640) → lower → seat ~(31,395)p2 |
| Human pin isolation | still Phase D GREEN with **R15** arm-pump + WJ2 L24/R14 (not product) |
| Pure seat + product fire | p132 @ ~(267,297); pose 84 often; mx200≈251 |
| R18 WJ2 product | L20 a4 R8 + **L14 a2 R6** + follow40 (no arm-pump) |
| Dump seat (lucky) | `post_bubble_fire_seat_live_r18.state` → product fire **top=True** mx200=301 |
| Full pure fire start | any open-loop WJ2 grid → mx200=251 (enemy AI phase) |
| Enemy patch | copy live slots 0/4/6 full 0x40 → full pure path **top=True** |
| Enemy HP=0 only | does **not** unlock Phase D |
| Arm-pump on pure seat x~31 | **RED** — jump fails (min_y=395) |

### Acceptance

- [x] Pure velocity dumps (fire seat / post_run / wall-approach paths + probe)
- [x] Dump-isolated Phase D GREEN (`live_r18` seat + product recipe)
- [ ] Full pure `top_reached=True` from CATH-04 — **red** (enemy phase)
- [x] Unit green; Phase A–C envelope held (min_y≤260, launched, often phase_c)
- [x] Residual + next card

### Probe

```text
uv run pytest super_metroid/tests/test_k4_norfair_scaffold.py -q
# 28 passed

# Isolation Phase D (dev only — not hop GREEN):
# post_bubble_fire_seat_live_r18.state + bubble_save_runway_fire_recipe
# → top=True min_y=133 mx200=301

uv run python super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state \
  --pin-json super_metroid/debug/bubble_to_bat_pure_pin_r18.json --no-red-diag
# success=false top_reached=False
# min_y≈159 max_x≈435 launched=True phase_c often True
# p132>0 saw84 on fire arc; mx200≈251
```

### Rejected this session

| Attempt | Why |
|---------|-----|
| Arm-pump on pure max-left seat | Jump desync; no height / no p132 |
| Fire from x~48 wide window | Walks off runway; min_y=395 |
| LEFT+X while walking to seat | KB p138 / Save door |
| Open-loop WJ2 grid on full-pure fire start | Zero tops; mx200=251 |
| Zero enemy HP only | Still mx200=251 |
| Wait 1200f for “clear window” | Deseats; loses fire arc |
| Patch human R15 WJ2 as product | Pure fails; human isolation only |

### Shipped product defaults

```text
seat: stationary X+L clear → walk-brake to x∈[25,32] face-left p2
prepare(y_clear=True, crouch=False)
runway_dash(frames=21, arm_pump=False)
spin_glide(83)
coast + WJ1 L20 a4 R8 + WJ2 L14 a2 R6 + RIGHT+B+A×40
```

### Next action (required) — **continue spine**

- **Next card ID:** `SM-K4.4-PURE-R19`
- **One change class:** enemy-phase-aware fire (wait/clear that preserves seat
  velocity) **or** closed-loop left-wall (pose 84) bounce that clears mx200≥300
  without enemy-state patches — then full pure `top_reached`
- **Do not** re-open arm-pump / wide fire window / LEFT+X seat without new pin
- **Source:** `scratch/post_rising_tide_to_bubble_pure.state`
- **Golden isolation:** `scratch/post_bubble_fire_seat_live_r18.state` (dev only)

### Non-claims

- Did not STATUS-promote / continuous tip advance
- Did not close SM-K4.4-PURE hop GREEN to Bat
- Isolation live dump top ≠ hop GREEN
- Enemy RAM patch is diagnostic only — never product
