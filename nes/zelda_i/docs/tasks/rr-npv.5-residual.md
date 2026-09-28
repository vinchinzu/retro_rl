> Historical lane note. The live sitting is the gathering prefix in [PRE_L1.md](../PRE_L1.md). This file stays because the clean-tip ladder or a route doc still cites it. It is not the current plan.

# rr-npv.5 residual — Clean L9 dodge / leftover-relative (fixture-live)

Closed 2026-09-28: `clean_poweron_c12` reached the credits from power-on (STATUS.md). The notes below are the 2026-09-10 fixture lane.

## Landed this sitting

- Patra/Ganon: no `idle(n)` through cooldown. Dodge manhattan ≤14
  (horizontal, flip at x=56/200), Gohma face-then-fire. Brown Ganon
  **commits to the silver-arrow axis** (dodge-off-column was first-red).
- `natural_path` / `path.py` door holds use `door_band_goal(leftover, hold_dir)`.
- Unit tests: `tests/test_level9_dodge.py` (boundary coords + NORTH_41 leftover).
- ROM, no assist / no pokes / deaths 0 / `route_eligible=false`:
  - Patra north door from `Level9FinalPatraReconFixture` (3716f, health 255→247)
  - Ganon `$0672 != 0` from `Level9BeforeGanonReconFixture` (1136f fight, 5 arrows)
  - credits mode 0x13 from the same pin (3167f e2e enter→credits)

`Level9EntranceReconFixture` glanced: play `0x76` `(120,205)` mode 5 TF `0xFF`
sword 3 arrows 2. `runner.open_env` ImportErrors (`load_state` missing);
labs used `make_env` + `resync_custom_state`.

## Leftover / next

- **Entry→Patra from the enter pin** is a later hop (16 prefix hops + join
  under Clean / no health refill). Suffix pins used for boss labs.
- First Ganon red: 7000f timeout, brown seen, 352 silver-arrow pulses missed
  because cooldown dodge walked off the column. Axis-commit fix; trial 2 green.
- `natural_path.py` is 1243 LOC (was 1221); no sibling extract.

## Not this sitting

- Power-on / spine-green `--through level9-credits --clean`
- Growing `prefix.py`
