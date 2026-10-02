# ALttP route residual — 2026-10-01

The Clean continuous tip remains room `0x50`. No STATUS promotion.

## rr-eog9.5 — hedge/stair approach proved; next rr-eog9.6

The actual hut-exit predecessor `Play_eog9_4` now reaches `eastern_door`
in `maps/screen_1e_eastern.json`. The mapped target is `(3920,1579)`;
accepted leave is `(3920,1582)` within the 4px band. Loaded screen `$8A`
stays `0x1E`; the entrance plaza is standing quadrant `0x1F`. The old
unmeasured `(1984,1144)` coordinate is unused.

The tactic is `opening_route.segment:approach_eastern_palace`, registered
as `eastern_approach`. It requires the completed hut's mapped outdoor exit
and story marker, uses the existing guarded outdoor collision walker,
and preserves inventory and story. Geometry remains in the JSON map;
no new runner, dispatcher, or production spine row. `rr-eog9.7` still owns
production composition in `spine/hops.py`.

Hedge segment: **2702f**, 1681 disclosed rollout restores, zero non-rollout
loads/writes, one half-heart hit (`4HP`) at segment frame2416, no death.
The live path passes around hedge/boulder obstacles and through stone
stairs. Same-checkpoint forward replay: zero desync/loads/writes; **889
committed frames** touch stair attribute `0x22`. The damage event occurs
at `(3920,1797)`, leaving HP `32→28`; no HP refill in this tail.

Composed **power-on Survival replay: 46173f**, synchronization and route
goal accepted, zero desync/loads/deaths. Seven disclosed opening refills
restore 14 hearts; progression/capacity writes remain zero. No new refill
after the hut. This is assisted development evidence, not Clean or
continuous evidence. Evidence:
`recordings/eog9_hedge_20261001/{controller,replay,hedge_power_on_replay}.json`
and `{hedge,power_on}.tape.npz`. The replay goal independently checks the
mapped approach, loaded screen, standing quadrant, inventory prerequisites,
and completed story; matching a failed controller tape cannot pass.

Leave RAM: outdoors loaded screen `0x1E`, standing `0x1F`, module/submodule
`0x09/0x00`, `(3920,1582)`, control ready, HP `28/32`, sword/lamp `1`, keys
`0xFF`, follower `0`, progress `2`, flags `0x15`, map indicator `3`,
rupees/bombs `0`. `alttp.screen_glance` passes the mapped leave band with
the outdoor screen as the location. `Play_eog9_5` is saved from the 46173f
forward replay with provenance. The stale indoor room register is not an
outdoor location proof.

ROM gates: all three outdoor predecessor regressions (crossing, full hut
visit, new approach) pass; all seven existing room-engine entry/stair
regressions pass. Targeted offline route/map/replay tests: 65 passed.
Docs gate: 8 passed. Broader ALttP offline suite: 329 passed, one baseline
failure in `test_secret_entrance_clear.py::test_run_from_sword_exit_ok`,
reproduced from a temporary archive of HEAD `02478f1d` (1 failed/6 passed
in that file). Filed separately as `rr-mott`; no weakened acceptance.

`rr-eog9.5` is complete. Epic `rr-eog9`: **5/8 children closed**.
Next: `rr-eog9.6`, mapped UP entrance `eastern_door_to_0xc9` and natural
lobby-to-`north_to_0xb9` seam from `Play_eog9_5`; never load
`EasternPalaceEntry`. Capacity is four hearts; current HP is 3.5 hearts.
The outdoor room engine currently passes the loaded screen to its point
walker; that caller must derive the entrance's standing quadrant to avoid
rejecting a point in `0x1F` inside loaded `0x1E`. Door transition evidence
in this map remains isolated; this sitting did not enter the lobby.
Then `.7` production spine composition and `.8` whole entrance coverage.

The hut rear cache remains closed behind type `0x28` bombable wall, and
natural bombs/rupees gathering stays `rr-yx26`. Preserve the proved crossing
detour `0x13 → 0x14 → 0x15 → 0x1D → 0x25 → 0x2D → 0x2E → 0x26 → 0x1E`;
the direct `0x1D` north-lip east exit remains unreachable. Crossing/hut
predecessor evidence stays in `recordings/eog9_{crossing,hut}_20261001/`.

Did not STATUS-promote, treat a save-state pin as power-on, write follower,
sword, keys, capacity, or progression, copy approach geometry into Python,
or overwrite `recordings/verified_tip_run.json`.
