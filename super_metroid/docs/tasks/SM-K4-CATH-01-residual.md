## Residual — SM-K4-CATH-01

### Result
GREEN

### Files changed
- `super_metroid/routes/kpdr/k4_norfair.py` — replaced scaffold with elevator settle → shallow RIGHT-first drop to top-right door band (y 840–900) → beam-shot blue door into `0xA7B3`.
- `super_metroid/scripts/probe/kpdr.py` — pure choice `business-to-cathedral-entrance` wired to `play_business_to_cathedral_entrance`.
- `super_metroid/tests/test_k4_norfair_scaffold.py` — pure-segment registration lock for cathedral entrance hop.
- `super_metroid/docs/tasks/SM-K4-CATH-01-residual.md` — PROCESS residual.
- `super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_business_to_cathedral_entrance_pure.state` — pure GREEN successor (generated).

### Verify paste

```text
uv run pytest super_metroid/tests/test_k4_norfair_scaffold.py -q
exit 0
8 passed in 0.17s

uv run python super_metroid/scripts/probe/kpdr.py pure business-to-cathedral-entrance \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_business_continuous.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_business_to_cathedral_entrance_pure.state \
  --pin-json super_metroid/debug/business_to_cathedral_entrance_pure_pin.json
exit 0
success=true roomIdHex=0xA7B3 samusX=39 samusY=139 pose=11 doorTransition=0 frames=959 controllerOnly=true developmentOnly=false sourceId=post_business_continuous
statePath=super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_business_to_cathedral_entrance_pure.state
```

### Acceptance

- [x] Source loads at `0xA7DE`
- [x] Ordinary `0xA7B3` without warp / item grants
- [x] Successor state only if pure GREEN
- [x] Unit/registration green
- [x] Residual PROCESS fields; no continuous/STATUS claim

### Residual risks

- Pure-green only; no continuous re-record, graph promote, or STATUS claim.
- Successor may need SOURCE_STATES fingerprint registration before CATH-02 pure isolation.
- LEFT-first shallow drop is a known failure mode (lower shelf y≈923 → fall past door lip); controller locks RIGHT-first + y≤900.

### Next action (required)

- **Next card ID:** `SM-K4-CATH-02`
- **One change:** Pure controller Cathedral Entrance `0xA7B3` → Cathedral `0xA788` (red door) from the captured pure successor.
- **Source state:** `custom_integrations/SuperMetroid-Snes/scratch/post_business_to_cathedral_entrance_pure.state`

### Non-claims

- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence.

### Probe pin (if pure/geometry) — mandatory metrics

```text
room=0xA7B3 pose=11 x=39 y=139 door_transition=0
frames=959 dwell=not reported last_pin=room=0xA7B3 pose=11 x=39 y=139 door_transition=0
```
