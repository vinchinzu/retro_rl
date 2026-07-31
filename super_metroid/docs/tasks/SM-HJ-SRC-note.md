# SM-HJ-SRC Capture Note

## Result

The requested source file now exists at:

`custom_integrations/SuperMetroid-Snes/scratch/hj_shaft_to_business_source.state`

It boot-validates in ordinary gameplay as:

```text
room=0xAA41 pose=1 x=400 y=187
```

The state is **developmentOnly**, not a natural-entry source. It was created
from the existing `dev_kpdr_business.state` with the development `hj_shaft`
door warp (`0x92D6`) and the existing shaft settle. No progression or capacity
RAM was forged by the capture helper. The state is useful for mechanical
bomb-tunnel work, but it cannot establish continuous evidence.

## Natural Capture Attempt

The preferred controller-only attempt booted
`custom_integrations/SuperMetroid-Snes/dev_hijump_collected_dev.state` and ran
`play_hj_room_to_shaft`. It did not reach the shaft: it timed out in ordinary
HJ room `0xA9E5` at `pose=1 x=80 y=187` after controller frame 733. Therefore
there was no natural source available in the existing anchors.

## Pure Result

Required command was attempted against the validated source:

```text
success: false
error: ensure_morph failed, pose=1
roomIdHex: 0xAA41
samusX: 400
samusY: 187
frame: 1142
controllerOnly: true
developmentOnly: false
```

The pure segment did not reach Business (`0xA7DE`); it failed before the
shaft traversal began while trying to confirm morph pose. The last reported
room/pose/coordinates are `0xAA41`, pose `1`, `(400,187)`.

## Non-Claims

- This does not claim a natural-entry source or continuous evidence.
- This does not claim `hj-shaft-to-business` pure-green.
- This does not STATUS-promote anything.
- No progression RAM was forged.
