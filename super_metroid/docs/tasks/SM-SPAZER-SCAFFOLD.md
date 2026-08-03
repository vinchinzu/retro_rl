# TASK SM-SPAZER-SCAFFOLD: Spazer detour module + room constants

## Recipe step
1 pure controller scaffold (geometry green not required)

## Model
Luna

## Wave type
implement

## Own files only
- `routes/kpdr/rooms.py` — add `ROOM_SPAZER = 0xA447` (and door IDs if known)
- `routes/kpdr/spazer.py` (**create**) — stub collect/return helpers
- `tests/test_spazer_scaffold.py` (**create**)
- residual: `docs/tasks/SM-SPAZER-SCAFFOLD-residual.md` (optional)

No continuous / STATUS / progression promote. No registry race required.

## Context
- Epic: [`SPAZER_EARLY.md`](SPAZER_EARLY.md)
- Continuous already reaches Below Spazer `0xA408` (`--to below_spazer`).
- Spazer Room `0xA447` is KPDR K2.2 optional; parked one-liner `SM-OPT-SPAZER`
  superseded by this ladder.
- Practice policy exists: `policies/room_clears/room_a447_from_a408_to_a408.json`
  — dual-track only; do not claim continuous from it.
- Human ref: red room walljump context `docs/tasks/refs/early_spazer_red_room.png`.

## Read first
- `docs/tasks/SPAZER_EARLY.md`
- `routes/kpdr/red_tower.py` (`play_bat_to_below_spazer` style)
- `routes/kpdr/rooms.py`
- `docs/tasks/SM-ALPHA-PB-01.md` or `SM-CHARGE-01.md` (scaffold style)

## Do
1. Add `ROOM_SPAZER = 0xA447` next to `ROOM_BELOW_SPAZER`.
2. Create `routes/kpdr/spazer.py` with:
   - `play_below_spazer_to_spazer` stub (`require_room` Below Spazer → timeout)
   - `play_spazer_collect` stub (require Spazer room; bounded pedestal approach)
   - `play_spazer_return_to_below` stub
   - Docstrings naming walljump residual risk (red-room approach, not force-green)
3. Unit tests: importable, room constant, callables exist.
4. Residual: next card `SM-SPAZER-SRC` if no continuous-like source listed;
   else `SM-SPAZER-PURE`.

## Do not
- Touch `continuous.py`, `STATUS.md`, `progression.py`
- Claim pure/continuous green
- Progression / capacity / beam RAM writes
- Copy practice JSON as route proof

## Acceptance
- [ ] Scaffold + tests green
- [ ] Non-claims: not continuous Spazer evidence
- [ ] Residual names next card ID

## Verify commands
```bash
uv run pytest super_metroid/tests/test_spazer_scaffold.py -q
```

## Done when
Executor returns residual with next card ID. Planner reviews before pure green.
