## Residual — SM-K4.4-PURE-R1

### Result
RED (honest geometry gap — mid walljump not pure-green; lower mid OK)

### Files changed
- `super_metroid/routes/kpdr/k4_norfair.py` — real geometry
  `play_bubble_to_bat_cave` (lower HJ climb + delayed fresh-A mid walljump +
  Super door phase); import `ROOM_BAT_CAVE`.
- `super_metroid/routes/kpdr/registry.py` — segment `bubble_to_bat_cave`.
- `super_metroid/scripts/probe/kpdr.py` — pure choice `bubble-to-bat-cave`.
- `super_metroid/tests/test_k4_norfair_scaffold.py` — registration + room id.
- `super_metroid/docs/tasks/SM-K4.4-PURE-R1.md` — living R1 card.
- `super_metroid/docs/tasks/SM-K4.4-PURE-R1-residual.md` — this residual.
- `super_metroid/docs/tasks/SM-K4.4-PURE-residual.md` — tip pointer refresh.
- Ephemeral: `scratch/post_bubble_mid_climb_pure.state` (lower pin, dev).
- Probe/debug under `super_metroid/debug/bubble_*` / red_diag (not route-ready).

No committed pure-green claim; no STATUS / continuous promote.

### Verify paste

```text
uv run pytest super_metroid/tests/test_k4_norfair_scaffold.py -q
exit 0
12 passed in 0.18s

uv run python super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bubble_to_bat_pure.state \
  --pin-json super_metroid/debug/bubble_to_bat_pure_pin.json
exit 1
success=false
error: bubble_to_bat_cave: Bat Cave Super door missed before room 0xB07A;
  room=0xACB3 pose=26 xy=(320,459) door_transition=0
  max_x=332 min_y=388 mid_reached=True top_reached=False door_reached=False
  supers=5 selected=0
frames=68053 controllerOnly=true
```

### Acceptance

- [x] Source loads at `0xACB3` (CATH-04 pin band) — source OK
- [ ] Top band y≤200 / x≥300 still in `0xACB3` — **not achieved pure**
- [ ] Ordinary `0xB07A` without warp / item grants — **not achieved pure**
- [ ] Successor state only if pure GREEN — **no successor written**
- [x] Unit/registration green (12 passed)
- [x] Residual PROCESS fields; no continuous/STATUS claim

### Residual risks / recon facts (load-bearing)

1. **Lower climb is real.** Charged Hi-Jump zig-zag mid-left column from
   entry (~y634) reaches save-door height (y≤400). Mid pin captured offline:
   `scratch/post_bubble_mid_climb_pure.state` ≈ (112, 369). Live pure probe
   reports `mid_reached=True`.

2. **Mid walljump still the blocker.** Pure probe min_y≈388; never
   `top_reached`. Maprando strat 154: from **left save-door platform**, run
   jump to **cavity** right wall (not outer x≈475), walljump twice with
   HiJump. Fresh-A delayed walljumps register briefly (pose 26 height ticks
   observed offline) but do not chain to y≤200 under door-avoidance.

3. **Far-right outer wall is a trap.** Place/climb on x≥400 stalls ~y325–395
   at Single Chamber door height; bounce forever, never enters top screen.
   Stay cavity x≈120–380 for the intended path.

4. **Door still not the pure blocker.** Place `(420, 130)` + Super pulses
   still enters ordinary Bat Cave (dev diagnostic only).

5. **Wrong-door hard-avoid remains load-bearing** (left Rising Tide / Save /
   Missiles Super; right Single Chamber).

6. **`velocity_x` RAM reads 0** while x clearly moves (pose 9 run / spin).
   Do not gate wall-contact on `velocity_x`; use pose stall / position.

### Next action (required)

- **Next card ID:** `SM-K4.4-PURE-R2` (or re-open R1 with tighter knob)
- **One change:** Scripted open-loop walljump from a **standing** save-door
  platform pin (not free-air place): run-up frames + jump arc into cavity
  right wall (~x250–320) + 2–3 consecutive fresh-A walljumps with re-engage,
  hard-cap x&lt;400 to avoid SC height trap. Optionally split pure mid source
  re-capture after lower lands standing (pose 1/2, vy=0).
- **Source state:**
  `scratch/post_rising_tide_to_bubble_pure.state` or
  `scratch/post_bubble_mid_climb_pure.state` if lower re-pin is standing-stable.

### Non-claims

- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence.
- Place/door success is **development diagnostic only**, not pure-green.
- Did not close SM-K4.4-PURE or R1 as green.
- Pure tip remains first Bubble (`post_rising_tide_to_bubble_pure`); Bat is
  still the blocker for more of K4.

### Probe pin (if pure/geometry) — mandatory metrics

```text
room=0xACB3 pose=26 x=320 y=459 door_transition=0
frames=68053 max_x=332 min_y=388
mid_reached=True top_reached=False door_reached=False
supers=5 selected=0
# No Bat Cave ordinary settle. No successor state written.
```
