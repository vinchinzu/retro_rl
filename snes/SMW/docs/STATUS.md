# Status: Super Mario World

| Field | Value |
|-------|-------|
| Last verification | 2026-08-10 |
| Runtime class | Bronze |
| Intervention class | Clean |

This file does not name an M-number. The open gate is below.

## Gate

Two independent BizHawk 2.11 replays through the first three normal route
levels, with the same RAM boundaries, entry and exit states, and one clean
single-attempt skill per level. Not met. The trusted prefix stops after two
exits.

## Verified on 2026-08-10

- USA ROM SHA1 `6b47bb75d16514b6a476aa0c73a683a2a4c18765`, SHA256
  `0838e531fe22c077528febe14cb3ff7c492f1f5fa8de354192bdff7137c27f5b`.
- BizHawk 2.11 loads native BK2 headlessly under Xvfb/Mono and exposes SMW
  WRAM to the oracle Lua script on Snes9x and BSNESv115+.
- TASVideos user file `637823197083827931` (2022, native BizHawk 2.3.2
  Snes9x) replays unchanged on BizHawk 2.11. Yoshi's Island 2 is green from
  power-on: translevel `0x2A`, entry frame 1649, exit frame 6138. The next
  stage repeats deaths and a game-over, so it is not a clean multi-level
  skill source.
- TASVideos user file `34596324054209273` (native BizHawk port of the
  published warps run) keeps its input rows. With only old core metadata
  retargeted to BSNESv115+, two BizHawk 2.11 runs match through two exits:
  - Yoshi's Island 2 / translevel `0x2A`: frames 1634-3943, maximum X 4858,
    normal end-timer completion.
  - Yoshi's Island 3 / translevel `0x27`: frames 4331-6314, maximum X 2721,
    one same-life wings sublevel, then the exits-completed transition.
  Both segments have entry and exit states and `clean_single_attempt`
  skills. Translevel `0x26` loses lives under v115, so the trusted movement
  prefix ends after the second exit.
- The original Snes9x 1.43 SMV and the bsnes v085 LSMV load after conversion.
  Current-core power-on playback dies or stalls in Yoshi's Island 2. Those
  lanes are red and must not provide skills.
- Submission `10095S` (2025) matches the ROM and was sync-verified on
  BizHawk 2.11. It uses ACE to force exits. Compatibility and level
  enumeration only, not a movement-skill source.

Raw movies, ROMs, states, and replay evidence stay ignored under `tas/ref/`
and `recordings/tas_oracle/`.

## Not this gate

Stable-retro chained clears and optimizer seeds are development recordings.
They are not the three-level BizHawk gate above.
