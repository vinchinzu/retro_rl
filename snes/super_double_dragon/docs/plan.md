# Plan — Super Double Dragon

Ladder #4 (tier 1). Development follows the TMNT IV pattern: prove short
segments, preserve named resume states, document any dev heal or transition
clone, and only then harden longer chains.

## Controls and policy

- Y punch, A kick, X jump, B block, L/R Dragon Power.
- Default combat aligns lanes, approaches, and alternates punch/kick.
- The first segment never uses B. Area `0x19` uses block/punch only as a
  low-HP Chin counter-loop breaker.
- Game-specific navigation currently covers the Mission 1 leftward top floor,
  Mission 2 spiral passage, and Mission 2 runway.

## RAM

See `docs/ram_map.md`. The important discovery is the indirect P1 page byte at
`0x1CF9`; actor kind `0x09` is not a safe player discriminator.

## Milestones

- Boot and first no-block combat segment: done
- Mission 1 -> natural `Stage2`: done
- Mission 2 -> natural `Stage3`: done
- Mission 3 areas `0x17`/`0x18`, `0x19` fight: done
- Natural `0x19 -> 0x1A` gym stairs and `0x1B` bosses: open
- Mission 4 from documented clone -> natural `Stage5`: done
- Mission 5 first combat group: done
- Mission 5 remainder, Missions 6–7, ending: open

## Next work

1. Capture the event/camera bytes around the archived run's Mission 3 stair
   entry and compare them with `Area19_Clear`.
2. Replace the `Stage4` transition clone with a natural Mission 3 boss clear.
3. Diagnose the post-wave trigger at `Stage5_FirstClear`, then reach `0x1E`.
4. Continue one area at a time with `scripts/run_area.py`; add policy branches
   only for observed collision or boss mechanics.
5. After the ending exists as segments, run a no-dev-heal continuous eval.
