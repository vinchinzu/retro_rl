## Residual — SM-K4.4-PURE

### Result
RED (honest geometry gap — not scaffold-only; climb not yet pure-green)

### Files changed
- `super_metroid/docs/tasks/SM-K4.4-PURE.md` — living card.
- `super_metroid/docs/tasks/SM-K4.4-PURE-residual.md` — this residual.
- R1 implementation + residual: see `SM-K4.4-PURE-R1-residual.md`
  (`play_bubble_to_bat_cave` registered; pure still RED).

No committed pure-green claim: pure probe not green; do not promote.

### Verify paste

```text
# SM-K4.4-PURE-R1 pure probe (2026-08-02) — RED
uv run pytest super_metroid/tests/test_k4_norfair_scaffold.py -q
# 12 passed

uv run python super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state \
  --pin-json super_metroid/debug/bubble_to_bat_pure_pin.json
# success=false mid_reached=True top_reached=False min_y=388
# pin room=0xACB3 pose=26 x=320 y=459 frames=68053
```

### Acceptance

- [ ] Source loads at `0xACB3` (CATH-04 pin band) — source OK
- [ ] Ordinary `0xB07A` without warp / item grants — **not achieved pure**
- [ ] Successor state only if pure GREEN — **no successor written**
- [x] Residual PROCESS fields; no continuous/STATUS claim

### Residual risks / recon facts (load-bearing)

1. **Door is not the blocker.** Top-right green Super door (node 7, block
   `[31,7]` ≈ x496 y112) opens with continuous Super pulses from the door
   ledge. Place `(420,130)` enters Bat Cave reliably (~153f). Pure must reach
   that ledge naturally.

2. **Lower climb works.** Charged Hi-Jump platforming from entry `(~58,642)`
   reaches **mid_reached** (save-door height y≤400). Mid isolation state:
   `scratch/post_bubble_mid_climb_pure.state` ≈ (112, 369).

3. **Hard gap: y≈350 → y≤200.** Maprando path is junction (node 9) → top-right
   via **Walljump with HiJump** on the **cavity** right wall (x≈250–320), not
   the outer right wall (stalls at SC height y~360). Pure R1 still RED
   (`top_reached=False`, min_y≈388).

4. **Wrong-door traps (must hard-avoid):**
   - Left y≈624: Rising Tide (node 3, entry) — re-exit to `0xAFA3`
   - Left y≈368: Save (node 2) → `0xB0DD`
   - Left y≈112: Green Missiles Super (node 1) → `0xAC83`
   - Right y≈368: Single Chamber (node 6) → `0xAD5E`

5. Morph-roll / right-pipe-only paths did not clear the gap without walljumps.

### Next action (required)

- **Next card ID:** `SM-K4.4-PURE-R3` (standing mid re-pin; see
  `SM-K4.4-PURE-R2-residual.md`)
- **One change:** Force standing mid handoff after lower climb before R2
  open-loop launch so full pure matches mid-iso height class (min_y≤260).
- **Source state:**
  `scratch/post_rising_tide_to_bubble_pure.state` (acceptance);
  mid isolation for knob tests only.

### Non-claims

- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence.
- Place/door success is **development diagnostic only**, not pure-green.
- Did not close SM-K4.4-PURE as green.
- Pure tip remains first Bubble; Bat is the current K4 pure blocker.

### Probe pin (if pure/geometry) — mandatory metrics

```text
room=0xACB3 pose=26 x=320 y=459 door_transition=0
min_y=388 max_x=332 mid_reached=True top_reached=False
frames=68053
# No Bat Cave ordinary settle.
```
