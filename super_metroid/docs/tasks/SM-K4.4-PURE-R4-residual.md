## Residual — SM-K4.4-PURE-R4

### Result
RED (honest geometry gap — lower pin-band exit shipped; full pure still unpinned)

### Files changed
- `super_metroid/routes/kpdr/k4_norfair.py` — R4 lower climb: HJ dir bias +
  exit conditions target save-door pin band (`x∈[77,160]`, stand-pin poses)
  instead of broad mid `100≤x≤320`; shared `_on_mid_iso_pin` for lower /
  re-pin / launch.
- `super_metroid/docs/tasks/SM-K4.4-PURE-R4.md` — living R4 card.
- `super_metroid/docs/tasks/SM-K4.4-PURE-R4-residual.md` — this residual.

No committed pure-green claim; no STATUS / continuous promote.

### Verify paste

```text
uv run pytest super_metroid/tests/test_k4_norfair_scaffold.py -q
exit 0
12 passed in 0.16s

# Full CATH-04 source
uv run python super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bubble_to_bat_pure.state \
  --pin-json super_metroid/debug/bubble_to_bat_pure_pin.json --no-red-diag
exit 1
success=false
error: bubble_to_bat_cave: Bat Cave Super door missed before room 0xB07A;
  room=0xACB3 pose=47 xy=(324,474) door_transition=0
  max_x=332 min_y=364 mid_reached=True top_reached=False door_reached=False
  standing_mid_pinned=False launched=True supers=5 selected=0
frames=64642 controllerOnly=true
```

### Acceptance

- [x] Source loads at `0xACB3` (CATH-04 pin band) — source OK
- [ ] Full pure `standing_mid_pinned=True` — **False**
- [ ] Full pure min_y≤260 — **not achieved** (min_y=364)
- [ ] Ordinary `0xB07A` — **not achieved pure**
- [ ] Successor state only if pure GREEN — **no successor**
- [x] Unit/registration green (12 passed)
- [x] Residual PROCESS fields; no continuous/STATUS claim

### Residual risks / recon facts (load-bearing)

1. **R4 one-change shipped:** lower climb no longer *breaks early* on broad
   mid (`100≤x≤320`); exit prefers pin band. Live full pure still ends
   unpinned (`standing_mid_pinned=False`) with **min_y=364** — same class as
   best R3 full pure. Left-column HJ bias alone does not put Samus on the
   save-door platform from CATH-04 entry.

2. **Mid-iso still works.** From `post_bubble_mid_climb_pure.state` the stack
   pins and reaches min_y≈260. The isolation state is the working handoff;
   full pure never reconstructs it.

3. **Likely geometry fact:** lower platforms from node-3 entry favor cavity
   mid-right shelves (~x200–320). Save-door platform (~x90–120, y≈370) may
   need a dedicated left-wall / ledge path (not only dir bias on the same HJ
   zig-zag). Offline recon of lower→(98,374) is the next informative probe.

4. **Open-loop + door unchanged.** Peak-cross still not pure-green even from
   mid-iso (min_y≈260, top_reached=False). After pin lands on full pure,
   open-loop retune may still be needed for top band.

5. **Wrong-door hard-avoid + cavity x cap remain load-bearing.**

### Next action (required)

- **Next card ID:** `SM-K4.4-PURE-R5`
- **One change:** Recon + implement a **dedicated lower-left ledge path** to
  the save-door platform (or re-capture a continuous-like mid pin after a
  scripted left-column climb). Prefer: short place/grid probe from CATH-04
  source for (x,y) waypoints that reach x∈[77,160] y∈[350,400] standing,
  then encode one scripted lower sub-phase (not more dir bias). Goal:
  `standing_mid_pinned=True` on full pure. Do not change door phase.
- **Source state:**
  `scratch/post_rising_tide_to_bubble_pure.state` (acceptance).

### Non-claims

- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence.
- Did not close SM-K4.4-PURE or R4 as green.
- Pure tip remains first Bubble; Bat still blocks K4 advance.

### Probe pin (if pure/geometry) — mandatory metrics

```text
# Full CATH-04 source
room=0xACB3 pose=47 x=324 y=474 door_transition=0
frames=64642 max_x=332 min_y=364
mid_reached=True top_reached=False door_reached=False
standing_mid_pinned=False launched=True
supers=5 selected=0
```
