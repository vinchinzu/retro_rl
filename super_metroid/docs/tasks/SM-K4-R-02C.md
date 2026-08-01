# TASK SM-K4-R-02C: Kihunter→Zeela — post-climb door-window retune

## Recipe step
1 pure controller (one-knob geometry residual)

## Model
Luna

## Wave type
implement

## Own files only
- `routes/kpdr/kraid_return.py` (`play_kihunter_to_zeela_return` only)
- optional residual: `docs/tasks/SM-K4-R-02C-residual.md`
- optional recon dump under `debug/` (gitignored) if useful

## Context (minimal)
- SM-K4-R-02 / 02B: shot-block + Hi-Jump **climb works** (`samus_y < 280`).
- Door setup still enters **Baby Kraid `0xA521`** instead of Zeela `0xA471`.
- R-02B pin: `room=0xA521 pose=105 x=39 y=116 door_transition=1`
  error `blue down-door entered Baby Kraid`.
- Current code: empty upper-traverse loop (`range(0)`), then RIGHT backoff /
  LEFT face / DOWN+X shots — still wrong hatch.
- Source:
  `custom_integrations/SuperMetroid-Snes/scratch/post_baby_to_kihunter_return.state`
  → `0xA4DA` lower-right alcove.
- Topology: reverse path Baby is the **east** hatch; Zeela is the **west
  upper blue down door** (block band ~`[7,15]`). MapRando room 81 node 2→5.
- Continuous still Varia-only; pure K3.6 only.

## Read first
- `routes/kpdr/kraid_return.py` (`play_kihunter_to_zeela_return`)
- `docs/tasks/SM-K4-R-02B-residual.md`
- `docs/SOURCE_STATES.md` (`post_baby_to_kihunter`)
- Forward hint only (no edit): `play_kihunter_to_baby_kraid` in
  `kraid_approach.py` — where the baby door sits in this room

## Do
1. **One knob:** retune **post-climb door setup only** (do not rewrite climb
   cycles unless climb regresses to timeout).
   - After climb, sample/log `samus_x` / `samus_y` (reason strings ok).
   - Drive Samus into the **Zeela down-door x-band** (left/upper) and **away**
     from the Baby east hatch before any DOWN+shot/drop.
   - Prefer explicit x windows (e.g. stay left of baby door, right of left wall)
     over blind LEFT holds.
   - Keep Baby-room early-fail guards.
2. ≤3 bounded door-window strategies; residual with pin if still red.
3. On pure green: `--output` → `scratch/post_kihunter_to_zeela_return.state`.
4. No graph promote, continuous, STATUS.

## Do not
- Implement zeela→warehouse (SM-K4-R-03)
- Progression/door RAM forges
- Multi-room compose

## Acceptance
- [ ] Pure green → ordinary `0xA471` **or** residual with post-climb x/y pin
- [ ] Fail loud on `0xA521` (no silent green)
- [ ] `uv run pytest super_metroid/tests/test_controller_common.py -q` green

## Verify
```bash
uv run python super_metroid/scripts/probe/kpdr.py pure kihunter-to-zeela-return \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_baby_to_kihunter_return.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_kihunter_to_zeela_return.state
uv run pytest super_metroid/tests/test_controller_common.py -q
```

## Done when
Pure exit 0 into Zeela, or residual after ≤3 door-window knobs with
**post-climb** pose/x/y and one next change.
