# Status: The Great Waldo Search

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M8 |
| Best verified result | Continuous power-on to the five-scrolls ending |
| Last verification | 2026-07-25 |
| Runtime class | Bronze |
| Intervention class | Clean |

One session from `NONE`, `players=2`, no mid-run state load. Boot uses
`build_boot_script`. Scenes 1 through 5 use `SCENE_RECIPES`. The ending is
held. Do not mash A. Latest capture: score 18850, about 9868 emulator frames,
`recordings/great_waldo_search_full_credits.mp4`.

Continuous inter-scene `pre_idle` (a `.state` load mutates RNG, so Cleared
rebuild timings are not this path):

| Advance | pre_idle | pulses |
|---------|----------|--------|
| Scene 2 | 0 | 8 |
| Scene 3 | 1 | 7 |
| Scene 4 | 2 | 7 |
| Scene 5 | 0 | 7 |

Scene 3 Waldo click on this path is `(196, 100)`.

## Not done

- A scene-id byte. `0x00C3` moves with the camera.
- Score RAM is noisy during the bonus animation. Scene 5 needs a settle of at
  least 200 frames.
