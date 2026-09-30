# F2 layout draft

Companion to `F2_development.md`. Base map: persistent farm on `Y1_D2_PowerOn_FarmClear` (`$7EA4E6`). End cover is grass. Uncovered tiles: **0**. Counted tiles: **2,968** (≥ 2,950).

## End state

Every FarmClear cell below `0xA0` is grass `0x70` (or a later stage `0x71`–`0x79` / winter `0x7A`). Those ids count and are walkable.

| Source on the clear pin | Cells | Prep, then plant |
| --- | ---: | --- |
| `0x01` untilled | 1809 | Hoe → `0x07` → grass seed → `0x70` |
| `0x02` dried | 1135 | Same hoe, then seed |
| Stump quads `0x09`–`0x0C` | 16 | Axe effect `$0D` writes 2×2 `0x02`, then hoe and seed |
| `0x55` | 8 | No tool bit. Winter D1 writes `0x07`. Next spring, seed writes `0x70` |

1809 + 1135 + 16 + 8 = 2968.

Stump top-lefts: `(34,42)`, `(42,44)`, `(36,51)`, `(42,58)`.

`0x55` cells: `(12,27) (13,27) (14,27) (12,28) (14,28) (12,29) (13,29) (14,29)`. They count today. They become `0x07` on Winter D1 and drop out until a spring/summer plant. Fall/winter seed writes `0x1D`, and the next Winter D1 clears `0x1D` as well. Plant them in spring or summer.

No new fences. The clear pin has no `0x05`. Grass is the walkable counted cover.

## Six cells on the later mask

FarmClear stores `0xA0` at `(36,45) (37,45) (38,45) (36,46) (37,46) (38,46)`. Fall D1 and Winter D1 store `0x02` there. Those six are how the later pins reach 2,974. They are not in the 2,968. `0xA0` cannot take a seed. If a later natural map matches the fall pin, hoeing those six and planting `0x70` raises the count to 2,974. The 2,950 proof does not depend on them.

## Paths

BFS on the grass map from the house stand `(8,26)`, 4-directional. Walk ids are `FARM_WALKABLE` minus `TRAVEL_SOLID_TILES`. `FARM_NO_GO_TILES` stays blocked, except the bin lip `(9,28)`, which F8 stood on (id `0xA8`).

| Place | Cell | Id on the clear pin | Role | Dist |
| --- | --- | --- | --- | ---: |
| House | `(8,26)` | `0xA8` | door stand, face up | 0 |
| Bin lip | `(9,28)` | `0xA8` | north lip of `$F2`; water is `(8,29) (9,29) (8,30) (9,30)` | 3 |
| Exit | `(0,24)` | `0xA1` | inside `MapExit` region `(0,24)–(0,28)` → path `0x0C` | 10 |
| Barn | `(19,21)` | `0xA1` | door stand, face up | 22 |
| Coop | `(28,21)` | `0xC5` | door stand, face up | 25 |
| Shed approach | `(26,31)` | `0x02` → `0x70` | south of the door | 27 |
| Shed door cell | `(26,30)` | `0xFF` | not a stand | — |
| Shed flanks | `(25,30)`, `(27,30)` | `0xA8` | reached, adjacent to the door cell | 1 from the door cell |

`(0,25)` is `0xA6` and `(0,26)`–`(0,28)` are `0xC0`. Those four are outside `FARM_WALKABLE`. The exit the farmer can occupy in the region is `(0,24)`.

`(26,28)` is `0xA1` and walkable in the catalog, and the shed no-go boxes it in. BFS does not reach it. It is not one of the 2,968.

On the original clear map, before any plant, every bare `< 0xA0` cell (soil and the four stumps) has a reached orthogonal neighbor from `(8,26)`. Nothing in the 2,968 is stranded.

## Uncovered

None. The 24-tile budget is unused on this geometry.

Slack: 2,968 − 2,950 = 18 on this pin, 24 if the fall mask's extra six are planted. A hurricane on the sleep into Summer D30 (grass 1/16 → `0x02`) is larger than either slack. See `F2_development.md`.
