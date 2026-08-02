# TASK SM-K4-CATH-03: Cathedral → Rising Tide (pure)

## Recipe step

1. Pure controller. Geometry green before graph promote / continuous.

## Model

Luna

## Wave type

implement

## Own files only

- `routes/kpdr/k4_norfair.py` — replace only
  `play_cathedral_to_rising_tide` (leave CATH-01/02 geometry and other
  scaffolds).
- `scripts/probe/kpdr.py` — add pure choice `cathedral-to-rising-tide`.
- `tests/test_k4_norfair_scaffold.py` — registration if needed.
- `docs/tasks/SM-K4-CATH-03-residual.md` — required PROCESS residual.

Do not edit `continuous.py`, `STATUS.md`, CATH-01/02 geometry, progression
verification ranks, or any other controller.

## Source and contract

- Preferred source:
  `custom_integrations/SuperMetroid-Snes/scratch/post_cathedral_entrance_to_cathedral_pure.state`
- Expected room: `0xA788` Cathedral (CATH-02 pure GREEN successor pin:
  room=`0xA788` pose=81 x=39 y=124 door_transition=0, **909 frames**)
- Target: ordinary Rising Tide `0xAFA3` through Cathedral **right green Super
  door** (node 2, block `[47, 7]`, orientation right; graph
  `connection_220_a788_2_to_afa3_1`; progression edge
  `cathedral_to_rising_tide`, requires `super_missiles`).
- Caps: Morph, Bombs, Missiles, Supers (≥1), Hi-Jump, Varia — **no Speed**.
- One named controller only: `play_cathedral_to_rising_tide`.

## Context

- **Repath:** first Bubble = Cathedral climb (no Speed).
- Prior: `SM-K4-CATH-02` pure GREEN (~909f Cathedral Entrance → Cathedral).
  Not continuous evidence; continuous tip remains power-on → Frog Save.
- Chain: CATH-01 → CATH-02 → **CATH-03** → CATH-04 (Bubble).
- Entry is left door lip of Cathedral after CATH-02; cross room right, select
  Supers, open green Super door, settle ordinary in `0xAFA3`.
- Scaffold already exists; registry segment id is
  `cathedral_to_rising_tide` (probe pure choice still missing).

## Read first

- `routes/kpdr/k4_norfair.py` (`play_cathedral_entrance_to_cathedral` Super-door
  pattern; `play_cathedral_to_rising_tide` scaffold)
- `routes/controller_common.py` (`select_weapon` — supers = 2)
- `docs/tasks/SM-K4-CATH-02-residual.md` (source pin + successor state)
- Graph connection `connection_220_a788_2_to_afa3_1` in
  `maps/full_room_graph.json`
- Progression `DoorEdge` `cathedral_to_rising_tide` in `progression.py`

## Do

1. Replace scaffold in `play_cathedral_to_rising_tide` with real geometry from
   left-lip Cathedral spawn to right green Super door into `0xAFA3`.
2. Register pure `cathedral-to-rising-tide` in `kpdr.py` (choices + play map).
3. Pure-probe GREEN → write
   `scratch/post_cathedral_to_rising_tide_pure.state`.
4. Residual → `SM-K4-CATH-04` or R1. No continuous/STATUS claim.

## Acceptance

- [ ] Source loads at `0xA788` (pin band matches CATH-02 successor)
- [ ] Ordinary `0xAFA3` without warp / item grants
- [ ] Successor state only if pure GREEN
- [ ] Unit/registration green
- [ ] Residual PROCESS fields; no continuous/STATUS claim

## Verify

```bash
uv run python super_metroid/scripts/probe/kpdr.py pure cathedral-to-rising-tide \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_cathedral_entrance_to_cathedral_pure.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_cathedral_to_rising_tide_pure.state \
  --pin-json super_metroid/debug/cathedral_to_rising_tide_pure_pin.json

uv run pytest super_metroid/tests/test_k4_norfair_scaffold.py -q
```

## Residual routing

- GREEN → `SM-K4-CATH-04` (Rising Tide → Bubble) or SRC catalog
- RED → `SM-K4-CATH-03-R1` one named phase; same Cathedral pure source

### PROCESS residual (required on exit)

Executor final message **and** `docs/tasks/SM-K4-CATH-03-residual.md` must
include every field from `docs/tasks/PROCESS.md` residual schema:

Result · Files changed · Verify paste · Acceptance · Residual risks ·
Next action (Next card ID + one change + source state) · Non-claims ·
Probe pin (room/pose/x/y/door_transition/frames).

Non-claims must state: no STATUS promote; no progression/capacity forge;
**not continuous evidence**.
