## Residual — SM-K4.4-PURE-R19

### Result
**GREEN** (full pure Bubble → ordinary Bat Cave)

Natural entry from CATH-04 source:

1. Seats max-left fire band (R18)
2. **Idle-waits** on seat until Geruta slots 4/6 hit a proven clear class
3. Fires product double-WJ (no arm-pump) → **Phase D** `top_reached` ~(305,141)
4. Sticky right-structure WJ + Super pressure → ordinary **`0xB07A`**

Two matching pure probes: **2012f**, room `0xB07A`, pose 11 ~(39,395).

### Files changed
- `routes/kpdr/bubble_mountain_params.py` — `FIRE_PHASE_*`, `DOOR_SUPER_*` / WJ period
- `routes/kpdr/bubble_mountain_primitives.py` — `bubble_fire_phase_geometry`,
  `bubble_wait_fire_phase`, `bubble_read_enemy_slot`; fire recipe `phase_wait=True`
- `routes/kpdr/bubble_mountain_mid.py` — product fire uses phase wait
- `routes/kpdr/bubble_mountain.py` — Phase E sticky-right Super door (no beam swap)
- `tests/test_k4_norfair_scaffold.py` — R19 geometry unit tests
- Tip boards + this residual

### Load-bearing facts

| Fact | Detail |
|------|--------|
| Natural entry | `post_rising_tide_to_bubble_pure` → lower → seat ~(31,395)p2 |
| Enemy root cause (R18) | open-loop fire mx200≈251 unless Geruta AI phase matches |
| Phase wait | pure **idle** on seat (preserves x∈[25,32]); no LEFT+X |
| Clear class A | e4∈[117–125]×[270–276], e6∈[190–198]×[158–172] (~89–93f on fullpure seat) |
| Clear class B | e4∈[158–165]×[272–276], e6∈[175–182]×[184–190] (~233–235f) |
| Rejected class C | live seat (179,113)/(146,155) tops, but pure near-miss (185,105)/(140,157) **false positive** |
| Enemy HP=0 | still does **not** unlock (AI phase, not damage clip) |
| Phase D end | ~(305,141) p79; min_y≈132 mx200≥300 |
| Phase E | sticky period-10 right WJ + Super when x≥420 y≤160; ~342f from D pin |
| Full pure | **2012f** ×2 → `0xB07A` ordinary |
| Successor | `scratch/post_bubble_to_bat_pure.state` |

### Acceptance

- [x] Enemy-phase wait preserves seat velocity (idle; no deseat)
- [x] Full pure `top_reached=True` from CATH-04
- [x] Full pure ordinary Bat Cave `0xB07A`
- [x] Two matching pure probes (2012f)
- [x] Unit green (30 scaffold + controller_common)
- [x] Residual + tip boards
- [x] Graph edge `bubble_to_bat_cave` → `controller_dev` (+ Cathedral stack)
- [x] Path board Bubble `0xACB3` + Bat `0xB07A` → **controller_dev**
- [ ] Continuous / STATUS tip advance — **planner compose next** (not this card)

### Probe

```text
uv run pytest super_metroid/tests/test_k4_norfair_scaffold.py -q
# 30 passed

uv run python super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bubble_to_bat_pure.state \
  --pin-json super_metroid/debug/bubble_to_bat_pure_pin_r19.json --no-red-diag
# success=true roomIdHex=0xB07A frames=2012
# samus ~(39,395) pose 11 door_transition=0
```

### Rejected this session

| Attempt | Why |
|---------|-----|
| Fixed wait 90f only | pure_seat phase offset differs; need geometry |
| Wide “class C” live box | pure_seat hits near-miss in 1f and fails top |
| Zero enemy HP | still mx200≈251 (R18 fact held) |
| Old Phase E walk-right + beam swap | falls to y~280; supers never open shell |
| period8 / spin-only door from D pin | no Bat |

### Shipped product defaults

```text
seat: max-left x∈[25,32] (R18)
phase_wait: idle ≤280f until Geruta class A or B
prepare(y_clear=True, crouch=False)
runway_dash(frames=21, arm_pump=False)
spin_glide(83) + WJ1 L20 a4 R8 + WJ2 L14 a2 R6 + follow40
door: select Supers; period-10 LEFT+A×3 / RIGHT+A×2 / RIGHT+B+A×5
      when x≥420 y≤160: RIGHT+X/B Super pressure
```

### Next action (required) — **continue spine**

- **Next card ID:** Bat → Speed Hall pure (`SM-K4.5-PURE` / registry
  `bat-to-speed-hall` if scaffolded)
- **One change class:** pure controller from `post_bubble_to_bat_pure` → Speed
  Hall / Speed path — **not** another Bubble Phase D isolation
- **Do not** re-open arm-pump / wide fire window / enemy RAM patch as product
- **Source for next hop:** `scratch/post_bubble_to_bat_pure.state` (room `0xB07A`)
- Continuous tip remains Frog Save until compose/stabilize after Speed pure

### Non-claims

- Did not STATUS-promote continuous tip past Frog Save
- Did not graph-compose Cathedral→Bubble→Bat into **continuous** yet
  (graph edges are `controller_dev` only)
- Enemy phase wait is **read-only** geometry; no enemy RAM writes
