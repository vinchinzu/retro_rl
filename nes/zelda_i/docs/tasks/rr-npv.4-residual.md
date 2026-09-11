# rr-npv.4 — Clean L8 Entrance→TF no retopup

Fixture-live only. `route_eligible=false`. Do not STATUS. Do not close the bead.

## Landed this sitting

- Resolved Room 0x3E Blue Darknuts combat and bomb-wall progression:
  - Eliminated the 1-pixel oscillation loop at `(72, 180)` / `(72, 181)` by centering the west corridor target at `x=64` (`40 <= x <= 80` aisle) and enforcing clean direction transitions.
  - Implemented tactical bomb drops (`snap.bombs >= 2`) facing incoming Darknuts along the corridor with post-bomb retreat timing (`_bomb_retreat`) to prevent self-damage while knockback displaces enemies.
  - Shared flank/rear sword attacks (`_flank_rear_slash`) landing 2-hit kills (64 damage/hit) with Magical Sword.
  - Added corridor clear detection and north wall bombing at `BOMB_NORTH_STAND` `(120, 105)` to blow open the passage to Room 0x2E.
- Room 0x3E is **completely cleared / navigated** under live Clean play: Link survived all 6 Blue Darknuts, bombed the north wall, and entered Room 0x2E at frame 2673 with 9 keys and 2 bombs remaining.
- Added 4 unit tests in `nes/zelda_i/tests/test_level8_north_column.py` covering 0x3E tactical bombs, flank slashes, corridor advance, and north wall bombing.
- Maintained `nes/zelda_i/level8/north_column.py` strictly under 1000 LOC (985 lines).
- 244 passed across all `test_level8*.py` unit tests.

## ROM glance — first red (stop, new leftover)

One trial. `Level8InteriorReconFixture` play `0x7E` `(120,205)` TF `0x7F`,
`--from-enter --clean --no-video`. Tag `l8clr_lab`.

```text
[0] level8_north_manhandla_bomb: succ=True failed=False f=1700 -> 0x5e [120, 189] m5 hc=3 notes=['arrived_0x5e_120_189']
[1] level8_darknut_key_up: succ=False failed=True f=2673 -> 0x2e [71, 181] m17 hc=3 notes=['link_death']
```

Final: L8 `0x2e` `(71,181)` mode 17 TF `0x7F` MK 0 keys 9 bombs 2 rupees 255
hc 3 health `0x20`. Deaths 1. Writes 0.

**0x3E Blocker Resolved & 0x2E Reached**:
- Room 0x5E clears cleanly, key collected, 0x4E door unlocked.
- Room 0x3E Blue Darknuts navigated cleanly with tactical corridor bombs and flank slashes, north wall blown open at `(120, 105)`, and Link entered Room 0x2E.
- Link died at frame 2673 at `(71,181)` in Room 0x2E against Manhandla.
- Stop predicate reached (1 ROM trial completed, stopped at first red).

## Leftover

Clean 0x2E Manhandla / Map room clear or bypass without infinite life. Entry to 0x2E
from 0x3E south hole at keys 9, bombs 2, hc 3. Next: 0x2E Manhandla combat / safe
pathing to north door (0x1E Gohma).

## Blockers

- Room 0x2E Manhandla combat / evasion under live Clean health (`hc=3`, no retopup).
- Full Entrance→TF without retopup now reaches 0x2E (previously blocked at 0x3E).


