# SM-K4-06E Residual

## Result
**GREEN** (pure) · graph `controller_dev` · **not** continuous

## Files changed
- `routes/kpdr/varia_return.py` — jump-enter left-door geometry after standing beam open
- `progression.py` — `kraid_to_eye_return` verification → `controller_dev`
- `tests/test_progression.py` — lock controller_dev; still not continuous
- `docs/SOURCE_STATES.md` — index `post_kraid_to_eye_return` capture
- `docs/tasks/QUEUE.md` — Wave 6 residual / next reverse hop
- `docs/routes/KPDR_TRACKER.csv` + `.md` — K3.3 → controller_dev

## Verify paste

```bash
uv run python super_metroid/scripts/probe/kpdr.py pure kraid-to-eye-return \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_varia_to_kraid_pure.state
# success=true roomIdHex=0xA56B samusX=472 samusY=397 pose=82 frames=610

uv run python super_metroid/scripts/probe/kpdr.py pure kraid-to-eye-return \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_varia_to_kraid_pure.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_kraid_to_eye_return.state
# captured room=0xA56B pose=2 x=472 y=395

uv run pytest super_metroid/tests/test_controller_common.py super_metroid/tests/test_progression.py -q
# 38 passed
```

## Acceptance
- [x] Pure green from continuous-like source
- [x] No progression/capacity/door/event/boss RAM forged
- [x] Graph `controller_dev` only
- [x] Eye source captured

## Residual risks
- **Not continuous:** `varia_to_kraid` + this hop not composed on power-on tip
- Downstream reverse hops (`eye_to_baby_return` …) still `unverified` / pure-red
  until each has natural source + pure green
- Jump-enter cadence is one-knob success; multi-run pure stability not measured
- Hi-Jump assumed (post-K3 loadout); band may differ without it

## Next action (required)
- **Next card ID:** SM-K4-R-01B (or SM-K4-07)
- **One change:** pure-tune `play_eye_to_baby_return` from
  `scratch/post_kraid_to_eye_return.state` only (door open + left exit)
- **Source state:** `scratch/post_kraid_to_eye_return.state` (0xA56B)

## Non-claims
- Did not STATUS-promote continuous frame tables
- Did not forge progression/capacity/door/event/boss RAM
- Not continuous evidence
- Did not open post-Varia continuous tip compose

## Probe pin (success)
room=0xA56B pose=82 x=472 y=397 door_transition=0 (settled ordinary eye room)

## Mechanism (for future reverse doors)
Floor band y≥400 on this left hatch does not transition; elevated jump-enter
after standing X-only shots does. Do not re-try free floor spin as the next knob.
