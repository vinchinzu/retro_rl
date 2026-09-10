# rr-8t4.4 — natural L6 → bait shop `0x34`

Living Survival residual. Do not STATUS. Do not add Food/bomb/key pokes.

## This sitting (2026-09-10) — bead `rr-ps7.3`

Claimed `rr-ps7.3`. Leftover-relative L2 OW walk on `0x4C` east mouth.

Policy (`overworld/path.py`): UP/DOWN hop with `align_x`, leftover on the
east/west edge: walk toward `align_x` on a walkable y. Occupancy miss
(true no-move, not a 2px slide) → block cell → y-peel; no path → stand.
Never RIGHT at `x≥232` (scrolls to `0x4D`). Inland hops stay
`align_and_push`.

### Power-on `--through level2-entry` (3/3 this sitting)

Stop at first red: `clear23_key` L1 play `0x23` `(144,149)` mode 5 TF `0x00`
keys 0 bombs 0 rupees 10 health `0x22` lo==hi, `occupancy_patrol` 4627
misses / 6000f. Not the `0x4C` hop. `set_state=0`. Tag `l2_entry_rrps73`.

### Isolated ROM (Level1ExitOverworld, `door_path=True`)

- Natural door hops: L2 play `0x7d` `(120,205)` mode 5 TF `0x01` keys 0
  bombs 0 health `0x33` lo==hi, deaths 0, `food_writes=0` / progression
  writes 0, hop_10_3c then `level2_path_stop`.
- East-mouth knock `y=157` at `(240,133)` arrival: first action LEFT, then
  UP peel, enter L2 `0x7d` `(120,205)` same glance. ~700f after knock.

Unit: leftover `(240,157)` hop UP `align_x=112` first action LEFT, never
RIGHT, never `unstick_wait`. On-column still pushes UP.

Clean campaign: `docs/tasks/rr-npv-clean-parallel.md`. Do not STATUS.

## Shop hop (wired, untested live)

Dedicated `--through level7-bait-shop`: Recorder warp join peels north at
`0x54` → `0x44` → shop `0x34`. No `ADDR_FOOD` write on that hop.

Hypothesis (untested):

```text
0x22 ↓0x32 →0x33 ↑0x23 →0x24
0x24 blow DOWN → 0x45
0x45 ↓0x55 ↓0x65 ←0x64 ↑0x54 ↑0x44 ↑0x34
```

Gaps (OVERWORLD_DOORS): `0x54→0x44` x≈116, `0x44→0x34` x≈132.

## Dead

- `0x22/0x32/0x33/0x23/0x24/0x25` south to row-4
- `0x33` RIGHT at y=141 into `0x34`
- New Food/bomb/key writes to skip `0x4C`
- Poke-pin `Level6ExitOverworld` as leave proof
- Hold UP at `0x4C` east mouth `x≥232` (trees)
- Occupancy 1px-grade on OW 2px UP slide (oscillated 157↔155)

## Leftover

Power-on: L1 play `0x23` `(144,149)` mode 5, TF `0x00`, Food 0, bombs 0,
keys 0, rupees 10, `food_writes=0`, `set_state=0`. Glance: water-maze
Goriya room, not L2 door. Next: L1 `clear23_key` (other lane) then
recompose `--through level2-entry` / `--through level7-bait-shop`.
