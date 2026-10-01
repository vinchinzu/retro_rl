# ALttP route residual — 2026-10-01

The Clean continuous tip remains room `0x50`. No STATUS promotion.

## rr-eog9.3 — Eastern grounds reached; next rr-eog9.4

The original `0x1D` north-lip belief stays dead: none of the 23 east exits
is reachable from `(2664, 1544)` on ordinary walkable tiles. Changing the
south-border entry within the reachable western band does not connect it
to the eastern bank. Do not repeat that direct screen-graph route.

The current walker baseline from `Play_eog9_2` continued farther south,
then failed at `(3936, 2460)`, standing quadrant `0x27` of loaded area
`0x1E`, with HP 32/32. Two missing behaviors caused that later failure:
outdoor stairs (`attr 0x22`) were walls, and the walker searched only for
border exits when its target was an internal quadrant of the loaded 2×2
area. Both are fixed in the existing tile planner/walker. The unused
module-level probe CLI was removed; the library and pin/play interfaces stay.

Verified path: `0x13 → 0x14 → 0x15 → 0x1D → 0x25 → 0x2D → 0x2E → 0x26
→ 0x1E`. Segment from the actual Sanctuary exit: 3848f, PolicyGuard 1442
checks/45 detours, 1626 disclosed restores, zero non-rollout loads, zero
writes, no damage or deaths. Inventory is unchanged. Same-pin forward tape
replay: zero desync/loads/writes.

Composed forward **power-on Survival replay**: 41305f, zero desync, zero
state loads, zero deaths. Seven disclosed last-heart refills restored 14
hearts in the opening prefix; the new overworld segment has none.
Progression and capacity writes remain zero. This is development evidence,
not a Clean or continuous-tier promotion.

Leave RAM: outdoors, loaded/standing screen `0x1E`, module/submodule
`0x09/0x00`, `(3328, 2047)`, control ready, sword/lamp `1`, keys `0xFF`,
follower `$F3CC=0`, progress `$F3C5=2`, flags `$F3C6=0x15`, HP `32/32`,
rupees `0`, bombs `0`. `Play_eog9_3.state` is saved with provenance pointing
to the 41305f power-on replay. Evidence:
`recordings/eog9_crossing_20261001/{controller,replay,power_on_replay}.json`
and `{crossing,power_on}.tape.npz`.

Tests: real-ROM crossing regression passes from `Play_eog9_2`; an offline
corridor test checks that stairs connect two internal quadrants. Forward
replay acceptance rejects the old southeast `0x27` pose even though `$8A`
is already `0x1E`. Whole hut/hedge/lobby route tests remain open.

Next: `rr-eog9.4`, hut entry/dialogue/resources from `Play_eog9_3`, using
`maps/screen_1e_sahasrahla.json`. Do not load `EasternPalaceEntry` or
`SanctuaryExit`. Roadside cuts previously produced no drops; `rr-yx26`
tracks actual gathering without inventory writes. Epic remains in progress,
3 of 8 children closed. No hut, hedge, or lobby action has run from this
predecessor; `.7` still needs the production CLI composition.

Did not write follower, sword, keys, capacity, or progression. Did not copy
approach geometry into Python or overwrite `recordings/verified_tip_run.json`.
