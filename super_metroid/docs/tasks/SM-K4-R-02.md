# TASK SM-K4-R-02: Pure `kihunter-to-zeela-return` (alcove → upper → down door)

## Recipe step
1 pure controller (one-knob geometry)

## Model
Luna

## Wave type
implement

## Own files only
- `routes/kpdr/kraid_return.py` (`play_kihunter_to_zeela_return` only)
- optional residual: `docs/tasks/SM-K4-R-02-residual.md`

## Context (minimal)
- Continuous tip still power-on → Varia only; this is reverse pure chain K3.6
- Source (required):
  `super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_baby_to_kihunter_return.state`
  → room **`0xA4DA`** Warehouse Kihunter, lower-right alcove (~x=459 y=385)
- Prior pure chain GREEN: Kraid→eye→baby→kihunter (`controller_dev`)
- Graph: left vertical door to Zeela is **blue down** at block `[7, 15]`
  (upper-left). Not a floor-walk left exit.
- MapRando (room 81 node 2→5 Base): from bottom-right door, break **floor shot
  blocks** (obstacle D) and climb with **Hi-Jump** to Kihunter Junction, then
  left to vertical door. “Spinjump up into the tunnel to shoot the shotblocks.”
- Planner pin (failed scaffold): hard wall **x≈357**, ceiling peak with crouch
  jump **y≈316**, never `door_transition!=0`, never left lower alcove.

## Read first (only these)
- `routes/kpdr/kraid_return.py` (`play_kihunter_to_zeela_return` scaffold)
- `routes/kpdr/kraid_approach.py` (`play_kihunter_to_baby_kraid` — reverse of
  the bomb-drop path; use only as geometry hint, do not edit)
- `docs/tasks/SM-K4-R-02.md` (this card)
- `docs/SOURCE_STATES.md` row `post_baby_to_kihunter`

## Do
1. Replace naive `DOWN` hold with a bounded climb-out of the baby-door alcove:
   - Position under the upper tunnel / shot-block band
   - Shoot **up** (and diagonals) to clear floor shot blocks
   - Hi-Jump / spinjump into upper level (y well below ~280)
   - Traverse left on upper floor toward vertical door (~block x=7)
   - Aim DOWN, open blue hatch, drop into `ROOM_ZEELA` (`0xA471`)
2. One primary primitive focus: **alcove exit via shot-block + Hi-Jump climb**.
   Do not also rewrite zeela→warehouse in this card.
3. `wait_ordinary_room` settle on Zeela with a reasonable band.
4. On pure green: save
   `scratch/post_kihunter_to_zeela_return.state` via `--output` if CLI supports it.
5. Do **not** change graph verification (planner promotes after review).
6. Do **not** touch `continuous.py` / `STATUS.md`.

## Do not
- Free multi-room compose past Zeela
- Invent progression/door RAM
- Claim continuous
- Parallel-edit `business_climb` / `varia_return` / STATUS

## Acceptance
- [ ] Pure probe green from named source → ordinary `0xA471`
- [ ] `uv run pytest super_metroid/tests/test_controller_common.py -q` green
- [ ] Residual with PROCESS schema + pin if still red
- [ ] Optional source capture on green

## Verify commands
```bash
uv run python super_metroid/scripts/probe/kpdr.py pure kihunter-to-zeela-return \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_baby_to_kihunter_return.state
uv run pytest super_metroid/tests/test_controller_common.py -q
```

## Done when
Pure exits 0 into Zeela, or residual after ≤2–3 bounded climb strategies with
last pin (room/pose/x/y/door_transition) and **one** next knob.
