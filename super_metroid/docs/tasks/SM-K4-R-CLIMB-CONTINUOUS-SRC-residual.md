## Residual — SM-K4-R-CLIMB-CONTINUOUS-SRC

### Result
RED

### Files changed

- `routes/kpdr/kraid_return.py` — no retained geometry change; the one-knob
  early-right-bias attempt was reverted after it left the exact same pin.
- `docs/SOURCE_STATES.md` — registered the accepted Varia checkpoint and its
  natural Kihunter predecessor.
- `progression.py` / `docs/routes/KPDR_TRACKER.csv` — demoted fixture-only
  K3.6/K3.7 controller claims to `unverified`.

### Verify paste

```bash
uv run python super_metroid/scripts/record/continuous.py --to varia --no-video \
  --report super_metroid/recordings/start_to_varia_continuous_checkpoint.json \
  --state-output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_varia_continuous.state
# exit 0: varia_collected, 104382f, all Varia integrity fields true

uv run python super_metroid/scripts/probe/kpdr.py pure kihunter-to-zeela-return \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_varia_continuous_to_kihunter.state
# exit 1: best_min_y=291; final room=0xA4DA x=475 y=395 pose=137 door_transition=0
```

### Acceptance

- [x] Exact accepted continuous predecessor captured.
- [x] Natural-input reverse chain reaches Kihunter without state loads or
  progression/capacity writes.
- [ ] Kihunter→Zeela pure green from that predecessor.
- [x] No continuous or STATUS promotion.

### Residual risks

- The old fixture starts in a materially different room-state/inventory
  lineage and cannot certify the spine.
- The current mid-ledge maneuver reaches y=291 at x=359, then falls; right
  motion begins too late to land the x>=365 band.

### Next action (required)

- **Next card ID:** SM-K4-R-CLIMB-CONTINUOUS-SRC-01
- **One change:** Redesign only the Kihunter wall-plant→mid-ledge launch from
  `post_varia_continuous_to_kihunter.state`, using an exact x/y trace before
  changing any timing constant.
- **Source state:** `scratch/post_varia_continuous_to_kihunter.state`

### Non-claims

- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence.

### Probe pin

room=0xA4DA pose=137 x=475 y=395 door_transition=0
frames=226
last_pin=room=0xA4DA pose=77 x=359 y=291 door_transition=0
