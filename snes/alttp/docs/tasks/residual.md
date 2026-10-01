# ALttP route residual — 2026-10-01

The Clean continuous tip remains room `0x50`. No STATUS promotion.

## rr-eog9.4 — hut visit proved; next rr-eog9.5

The actual Eastern grounds predecessor now enters Sahasrahla's hut (full
room `0x0105`), talks to the Elder, inspects both accessible pots, and exits
through the mapped south door. Geometry lives in `screen_1e_sahasrahla.json`
and `room_05.json`. The existing outdoor walker supports point targets;
the existing mapped room engine handles outdoor origins and guarded paths.
The composed tactic is `opening_route.segment:visit_sahasrahla_hut`, exposed
as segment `sahasrahla_hut`; no separate hut runner or second dispatcher.

Messages `0x32/0x33` mark the pendant locations: `$F3C7` changes `1→3`.
Local game source `Sprite_Sahasrahla`/`Sasha_Idle` confirms this no-pendant
branch. Progress `$F3C5=2`, flags `$F3C6=0x15`, follower `$F3CC=0`, boots and
pendants stay unchanged. No boots reward before the Eastern pendant.

The left pot restores a heart when HP is low; a damage-bearing probe
confirmed HP `24→32`. The accepted guarded segment stays at full HP,
so neither pot changes inventory. Rupees/bombs remain zero. All three rear
chests (50 rupees,3 bombs,50 rupees) remain closed behind live north door
type `0x28` (bombable); no bombs or boots are available. `rr-yx26` still owns
natural resource gathering and a later cache return.

Hut segment: 2166f, 491 disclosed rollout restores, zero non-rollout loads
or writes, no damage/refills. Same-checkpoint forward replay: zero
desync/loads/writes. Composed **power-on Survival replay**: 43471f, zero
desync/loads/deaths. Seven disclosed opening refills restored 14 hearts;
none occur in the hut tail. Progression/capacity writes remain zero.
Evidence: `recordings/eog9_hut_20261001/{controller,replay,power_on_replay}.json`
and `{hut,power_on}.tape.npz`. This is assisted development evidence.

Leave RAM: outdoors screen `0x1E`, module/submodule `0x09/0x00`,
`(3312,1851)`, control ready, HP `32/32`, sword/lamp `1`, keys `0xFF`,
follower `0`, progress `2`, flags `0x15`, map indicator `3`, rupees/bombs `0`.
`Play_eog9_4` is saved from the 43471f forward replay with provenance.

ROM gates: full hut predecessor regression, outdoor entry, existing
indoor room/stair edges, and Sanctuary-to-Eastern crossing. Offline route,
map and replay acceptance tests pass; docs gate passes. `rr-eog9.4` is
complete. Epic `rr-eog9`: 4/8 children closed. Next: `rr-eog9.5`, hedge/Armos
approach from `Play_eog9_4`, then lobby `.6`, production spine composition
`.7`, whole entrance coverage `.8`. Do not load `EasternPalaceEntry`.

Preserve the proved crossing detour `0x13 → 0x14 → 0x15 → 0x1D → 0x25 →
0x2D → 0x2E → 0x26 → 0x1E`; the direct `0x1D` north-lip east exit remains
unreachable. Crossing predecessor evidence is in
`recordings/eog9_crossing_20261001/` (41305f forward Survival replay).

Did not STATUS-promote, treat a save-state pin as power-on, write follower,
sword, keys, capacity, or progression, copy approach geometry into Python,
or overwrite `recordings/verified_tip_run.json`.
