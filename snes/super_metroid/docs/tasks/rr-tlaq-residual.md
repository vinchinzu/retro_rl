## Residual — rr-tlaq Phantoon fight pure (0xCD13)

**Status:** window 1 charge 300 GREEN. Skip-right + left beam-snipe rain:
**no drops**, rain still cycling at +6k, **no left fig-8**. Full fight RED.
**Pin:** `scratch/post_ws_basement_to_phantoon.state`
**Probe:** `window --windows 2 --weapon beam --wait 6000`
**Report:** `scratch/phantoon_window_beam_w2_farm.json`

Do **not** STATUS-promote. Default CLI stays `ice`. Super-spray is not a hit.
Do **not** start another 16k until a measured W2 chip. Do not fire x=219.
Do not retry x≥230. Do not morph-tank rain.

### Public policy (wiki)

https://wiki.supermetroid.run/Phantoon
https://wiki.supermetroid.run/Phantoon#Phantoon_First

KPDR beginner 4-round: charge when the eye opens, two more, repeat.
Flame droplets: shoot for drops when farming. Super **enrages**.

Eye is **slot 1**. Live flags: `$0FB2` / eye IL `$0FD2`.

### What works (verified this pass)

- **Window 1** charge **300** at `(104, 149)` p43 vs `(120, 108)` dy=41.
- Skip-right: `charge_window_ok` is left fig-8 only (park x at func change).
- Left standing snipe (pose 3, x=37–53, UP+tap X): **alive at 99** after
  ~6k post-W1 wait. Morph-tank died ~f4238 at 0. Assist off.

### What fails

1. **Flame snipe produced zero drops — halt.** `farm.health_up=0`,
   `farm.missile_up=0`. `list_pickups` (Spore `$F337` table) stayed empty
   the whole skip/rain. Missiles stayed 20. Health only went down
   (239→219→199→179→159→139→119→99).
2. **Rain did not end** in 6000f after W1. Still `$D7F7`/`$D788` at f7613
   vs (128, 96) / (168, 128). No left fig-8. Timeout, health 99 pose 3
   `(37, 187)`.
3. Right fig-8 at x=219 is the body. Left morph cannot tank at 239.

### Rain dump (every ~30f)

`D72D` (208, 96) → `D5E7` skip → open (203, 83) skipped standing → `D82A`
then `D767`/`D788`/`D7D5`/`D7F7` cycling (186,123) (88,128) (48,96)
(128,96) (168,128) (168,64) (88,64) (208,96)… still cycling at timeout.

### Next actions (do not start another 16k first)

1. No-assist cannot farm post-W1 rain from the left corner with the Spore
   pickup table / uncharged UP taps. Next is a **different farm seat or
   shot** (not under (128, 96), not x=219) **or** accept rain is not a
   farm from this pin at 239. Halt at first miss.
2. Dual-green `scratch/post_phantoon_poweron.state` only after a kill
   (do **not** clobber `post_phantoon_defeated.state`).

```bash
uv run python snes/super_metroid/scripts/probe/phantoon_combat.py window --windows 1 \
  --weapon beam --report snes/super_metroid/scratch/phantoon_window.json
```

### Non-claims

- Did not STATUS-promote past Ice
- Did not change `DEFAULT_CONTINUOUS_TIP`
- Did not write `recordings/ws.json`
- Did not append to `POST_ICE_SPINE` / `WS_ONLY_HOPS` / `--to ws`
- Did not close `rr-g3nj`
- Did not rewrite `play_ws_entrance_to_main` / `play_ws_main_to_basement` /
  `play_ws_basement_to_phantoon`
- Did not close `rr-tlaq` (full kill still RED)
