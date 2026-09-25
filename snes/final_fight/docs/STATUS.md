# Status: Final Fight

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M3 |
| Best verified result | Stage 1 and Stage 2 segment clears; Stage 3 through wave 5 |
| Last verification | 2026-07-27 |
| Runtime class | Bronze |
| Intervention class | Clean |

Controller segments from development save states are the verified work.
Writes of `game_status` to clear an area, heal pokes, and
`--force-enemy-hp` are development bridges. They are not Clean clears and
they are not natural entry.

## Verified

- `Stage1.state` is fight-ready. Damnd reaches HP underflow at
  `Stage1_Clear`. Natural `0x0CD2` and clear-round were not observed. Idling
  on the underflow ghost does not advance the round.
- Subway `Stage2` is reached by a clear-area bridge, not by that missing
  round flag. Sodom is killed from `Boss2_Drawn` with spaced `UP+Y` then a
  grab (reproduced 3/3) into `Stage2_Clear`. `0x0CD2` stays 0.
- West Side `Stage3` is reached by a clear-area bridge through the Break Car
  bonus. Waves 1 through 5 clear from resume states. A continuous run from
  `Stage2_Clear` still dies in wave 2.
- Area 1 at camera 2560 has a thug near HP 250. Face-Y chips it. The best
  logged chip used heal pokes (about 250 to 101). A legitimate kill is open.
  `Boss3` exists only as a force-HP map. That map is not a clear.
- Player base is `0x0D00`. Up increases Y. Combat parsing and stage bytes are
  in `docs/ram_map.md`.

## Not done

- Area 1 HP 250 kill without a heal poke, then a real Boss 3 fight.
- Natural clear-round from Damnd and from Sodom.
- Title-to-credits with no mid-run save and no RAM writes.
