# ALttP route residual — 2026-10-01

The Clean continuous tip remains room `0x50`. This sitting did not move it.

## rr-eog9.1 Sanctuary heart chest — collected

The previous unreachable conclusion was false. A chest is selected by its
attribute byte, not by requiring the interaction probe to hit the top-left
`$06E0` location. `Link_PerformOpenChest` also requires facing UP; the
previous RIGHT-only target poses could never open it.

From the exact escort terminal `(1272, 568)`, the measured button route is:

1. DOWN to `heart_aisle_south` `(1272, 640)`.
2. LEFT to `heart_west_south` `(1240, 640)`.
3. UP to `heart_south_lip` `(1240, 600)`.
4. RIGHT one pixel to `heart_open_front` `(1241, 600)`.
5. UP one frame to face north, then A; advance the receipt text with A/B.

Use tolerance 0 at all four approach points. Geometry is in
`maps/room_12.json`. At the old south stop, the UP probe `(1247, 606)`
misses at tile `(27, 11)`. One pixel RIGHT changes it to `(1248, 606)`,
tile `(28, 11)`, attr `0x58`, and opens the chest. The priests need not move.

`recordings/sanctuary_20261001/heart_controller.json` records the source
replay: all 36,675 frames matched the original tape CRCs, then 256 appended
button frames collected SanctuaryHeart. Independent replay of the combined
36,931-frame tape (`heart_replay.json`) also passed with zero desyncs and
max/current HP `32/32`. Max HP increased `24 -> 32` and
current HP `10 -> 32`; live chest bit `$0402 & 0x0100` and room save bit
`0x0010` are set. `Play_eog9_1.state` was saved in the integration directory
with provenance. The combined tape is `heart_controller.tape.npz`.

Leave glance: room `0x12`, module/submodule `0x07/0x00`, `(1241, 600)`,
facing UP, control ready, sword `1`, lamp `1`, keys `0`, follower `$F3CC=0`,
progress `$F3C5=2`, flags `$F3C6=0x15`, max/current HP `32/32`.
Full run: zero state loads, seven disclosed Survival HP refills inherited
from the escort prefix. Heart tail: zero writes, including zero capacity
or progression writes. This is assisted development evidence.

`SanctuaryExit` is a separate resumed pin from IcePalaceEntry, outdoors
on screen `0x13` at `(1880, 1083)`, with max HP `64` and later equipment.
It is not the three-heart escort tape terminal used by this bead.

Next: `rr-eog9.2`, measure the south exit from `Play_eog9_1`. Later beads
were not started. Did not STATUS-promote or treat a loaded pin as power-on.
Did not write follower, sword, keys, capacity, or progression. Did not copy
approach geometry into Python or overwrite `recordings/verified_tip_run.json`.
