# TASK SM-K4-CATH-04: Rising Tide → Bubble Mountain (pure)

## Recipe step

1. Pure controller. Geometry green before graph promote / continuous.

## Model

Luna

## Wave type

implement

## Own files only

- `routes/kpdr/k4_norfair.py` — replace only
  `play_rising_tide_to_bubble` (leave CATH-01/02/03 geometry and other
  scaffolds).
- `scripts/probe/kpdr.py` — add pure choice `rising-tide-to-bubble`.
- `tests/test_k4_norfair_scaffold.py` — registration if needed.
- `docs/tasks/SM-K4-CATH-04-residual.md` — required PROCESS residual.

Do not edit `continuous.py`, `STATUS.md`, CATH-01/02/03 geometry, progression
verification ranks, or any other controller.

## Source and contract

- Preferred source:
  `custom_integrations/SuperMetroid-Snes/scratch/post_cathedral_to_rising_tide_pure.state`
- Expected room: `0xAFA3` Rising Tide (CATH-03 pure GREEN successor pin:
  room=`0xAFA3` pose=9 x=39 y=139 door_transition=0, **1162 frames**)
- Target: ordinary Bubble Mountain `0xACB3` through Rising Tide **right blue
  door** (node 2, block `[63, 7]`, orientation right; graph
  `connection_221_afa3_2_to_acb3_3`; progression edge
  `rising_tide_to_bubble`, caps `_K4_CAPS` — no Super required).
- Caps: Morph, Bombs, Missiles, Supers, Hi-Jump, Varia — **no Speed**.
- One named controller only: `play_rising_tide_to_bubble`.

## Context

- **Repath:** first Bubble = Cathedral climb (no Speed).
- Prior: `SM-K4-CATH-03` pure GREEN (~1162f Cathedral → Rising Tide).
  Not continuous evidence; continuous tip remains power-on → Frog Save.
- Chain: CATH-01 → CATH-02 → CATH-03 → **CATH-04** (Bubble closeout pure).
- Rising Tide is 5×1 screens (80×16 blocks). Left lip spawn after CATH-03;
  cross right through lava pits / Sovas / Dragons; blue door near x≈1008 /
  y≈112 into Bubble Mountain mid-left node 3.
- Scaffold already exists; registry segment id is `rising_tide_to_bubble`
  (probe pure choice still missing).

## Read first

- `routes/kpdr/k4_norfair.py` (`play_cathedral_to_rising_tide` knockback
  pattern; `play_rising_tide_to_bubble` scaffold)
- `routes/controller_common.py` (`select_weapon`, morph helpers)
- `docs/tasks/SM-K4-CATH-03-residual.md` (source pin + successor state)
- Graph connection `connection_221_afa3_2_to_acb3_3` in
  `maps/full_room_graph.json`
- Progression `DoorEdge` `rising_tide_to_bubble` in `progression.py`

## Do

1. Replace scaffold in `play_rising_tide_to_bubble` with real geometry from
   left-lip Rising Tide spawn to right blue door into `0xACB3`.
2. Register pure `rising-tide-to-bubble` in `kpdr.py` (choices + play map).
3. Pure-probe GREEN → write
   `scratch/post_rising_tide_to_bubble_pure.state`.
4. Residual → next Speed/Wave hop card or SRC catalog. No continuous/STATUS claim.

## Acceptance

- [ ] Source loads at `0xAFA3` (pin band matches CATH-03 successor)
- [ ] Ordinary `0xACB3` without warp / item grants
- [ ] Successor state only if pure GREEN
- [ ] Unit/registration green
- [ ] Residual PROCESS fields; no continuous/STATUS claim

## Verify

```bash
uv run python super_metroid/scripts/probe/kpdr.py pure rising-tide-to-bubble \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_cathedral_to_rising_tide_pure.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state \
  --pin-json super_metroid/debug/rising_tide_to_bubble_pure_pin.json

uv run pytest super_metroid/tests/test_k4_norfair_scaffold.py -q
```

## Residual routing

- GREEN → Bubble→Bat Cave / Speed Hall pure cards or SRC catalog for Bubble
- RED → `SM-K4-CATH-04-R1` one named phase; same Rising Tide pure source

### PROCESS residual (required on exit)

Executor final message **and** `docs/tasks/SM-K4-CATH-04-residual.md` must
include every field from `docs/tasks/PROCESS.md` residual schema:

Result · Files changed · Verify paste · Acceptance · Residual risks ·
Next action (Next card ID + one change + source state) · Non-claims ·
Probe pin (room/pose/x/y/door_transition/frames).

Non-claims must state: no STATUS promote; no progression/capacity forge;
**not continuous evidence**.
