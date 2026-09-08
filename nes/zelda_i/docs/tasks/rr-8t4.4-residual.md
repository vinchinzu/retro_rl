# rr-8t4.4 — natural L6 → bait shop `0x34`

Living Survival residual. Do not STATUS. Do not add Food/bomb/key pokes.

## This sitting (2026-09-08) — OW farm fold, bead `rr-wabn`

Folded `zelda_i.overworld.locations` enemy-drop farms into hop policy so
shops can kill+restock instead of poking rupees. Assist hearts off for
L1 overworld combat practice. Do not close `rr-wabn` (arrow splice onto
the default spine is still open). L2 `0x4C` leftover below is unchanged.

No-assist `run_to_level1.py --natural-entry --screen-only` **1/1**: OW
`0x37` `(240,141)` mode 5, sword 1, rupees 0, hearts 2/3 (`0x22`),
nav 1431f, `farm_attempts=0`. Chasing octoroks on 0x78 at
`farm_below_hearts=3` died; L1 farms only below 2 hearts. Path hops
still default `farm_below_hearts=3` (inert with Survival assist).

`--through level1-arrows` still the dedicated 80R buy. Spine default
assist stays on; `--no-infinite-life` is legal.

## Prior sitting

Dedicated `--through level7-bait-shop` is wired: Recorder warp join peels
north at `0x54` → `0x44` → shop `0x34`. No `ADDR_FOOD` write on that hop.
Default `level7-entry` still uses `SurvivalBaitPurchaseController`.

Power-on never reached the shop. `enter_level2` is red **3/3**
(byte-identical, seed 0):

| `--through` | leftover | hop |
|-------------|----------|-----|
| `level7-bait-shop` t1 | OW `0x4C` `(240,157)` mode 5 TF `0x01` bombs 0 keys 0 rupees 12 | index 10 `0x3C` UP, timeout 25000f, end 48820 |
| `level7-bait-shop` t2 | same | same |
| `level2-entry` control | same | same |

PNG: `recordings/l7_bait_shop_rr8t4_final.png`. East mouth of `0x4C` after
`0x4D` LEFT; UP at x=240 is trees. Next hop wants `align_x=112`.
Bead **`rr-ps7.3`**. No inventory poke.

Poke-pin `Level6ExitOverworld` is not leave proof: warp from `0x24` landed
on L6 door `0x22` and stuck (entrance suppresses Recorder). Do not iterate
the shop walk from that fixture.

## Hypothesis (shop, untested live)

```text
0x22 ↓0x32 →0x33 ↑0x23 →0x24
0x24 blow DOWN → 0x45
0x45 ↓0x55 ↓0x65 ←0x64 ↑0x54 ↑0x44 ↑0x34
```

Gaps (OVERWORLD_DOORS recon): `0x54→0x44` x≈116, `0x44→0x34` x≈132.

## Dead

- `0x22/0x32/0x33/0x23/0x24/0x25` south to row-4
- `0x33` RIGHT at y=141 into `0x34`
- New Food/bomb/key writes to skip `0x4C`

## Leftover

OW play `0x4C` `(240,157)` mode 5, TF `0x01`, Food 0, bombs 0, keys 0,
rupees 12, `food_writes=0`, `set_state=0`. Glance: east mouth, not shop
`0x34`. Next: `rr-ps7.3` (L2 walk), then recompose `--through level7-bait-shop`.
