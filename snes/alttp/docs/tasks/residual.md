# ALttP route residual — 2026-10-01

This is the living residual for the opening route. `docs/STATUS.md` still owns
the Clean maturity gate; this report records the verified Survival result and
the work needed before that result can become C1.

## Verified route

The power-on controller and an independent forward replay now cover the route
from reset through the castle B1 room `0x81`, Zelda rescue, and Sanctuary.
The route leaves `0x81` through the measured north path `0x71` → `0x70` →
`0x80`; the former west-edge hypothesis was false and has been removed from
the maps.

Evidence is in `recordings/sanctuary_20261001/`:

- `controller.json`: 36,675 committed frames, seven development search
  restores, seven disclosed HP refills, and no progression or capacity writes.
- `replay.json`: 36,675 frames, zero desyncs, zero replay state loads, and
  seven audited HP refill writes.
- Final story state: room `0x12` at `(1272,568)`, control ready, follower
  `$F3CC == 0`, progress `$F3C5 == 2`, flags `$F3C6 == 0x15`, and HP 10.

The rescue stage ends with Zelda following (`$F3CC == 1`) in the cell. The
escort stage opens the mantle, crosses the sewers, and reaches Sanctuary where
the follower is released. The route uses the real predecessor state and does
not grant a key, item, progression byte, or heart capacity.

This is Survival development evidence under `docs/ASSIST_CONTRACT.md`. The
controller's search restores and HP refills keep it out of the Clean ladder;
the Clean continuous tip remains room `0x50`.

## Audit fix

`rr-0by1` is fixed. `LiveRun` now keeps its audited boundary when helpers use
`env.unwrapped`, so direct emulator assignments are counted. The regression
and Kholdstare audit live in `spine/tape.py` and `tests/test_tape.py`. The
Kholdstare fight remains isolated and is not part of this route proof.

## Remaining work

1. Recompose the route without Survival refills or controller search restores,
   starting from the real Clean predecessor, then run the tape replay.
2. Use that result to close C1/AP-082 and move the continuous tip only after
   the Clean acceptance predicate passes.
3. Keep dungeon boss and item pins isolated until their natural predecessors
   are chained; the Sanctuary result does not imply any later dungeon rung.

## Reproduction

```bash
export QT_QPA_PLATFORM=offscreen SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
PYTHONPATH=snes uv run python snes/alttp/scripts/audit_power_on.py controller \
  --through sanctuary \
  --out snes/alttp/recordings/sanctuary_20261001/controller.json
PYTHONPATH=snes uv run python snes/alttp/scripts/audit_power_on.py replay \
  --tape snes/alttp/recordings/sanctuary_20261001/controller.tape.npz \
  --out snes/alttp/recordings/sanctuary_20261001/replay.json
```
