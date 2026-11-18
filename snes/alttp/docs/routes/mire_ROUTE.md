# Dungeon route — Misery Mire

Development only. Isolated poke and scan pins. A save-state pin is not a power-on run and is not a STATUS claim.

## Entry

| Field | Value |
|-------|--------|
| Pin | `MiseryMireEntry` (kind poke). Room `0x0098`, module `0x07`, dungeon `0x0E`, indoors, layer 0, link `(4216,5080)`, HP `0x18/0x18`. Audit: 0 incoherent. Ether `$F348=1`, boots, pearl, lamp, magic 128, sword 1, shield 1, ability 252, keys `$F36F=0`, big key `$F366=0`. Follower `$F3CC=5` was already on the load. |
| Room | `0x98`. US dungeon entrance table ROM `$02C813` (file `0x14813`). Index `0x09` reads room `0x0084`. The palace-`0x0E` index whose room word is `0x0098` is `0x27`. |
| First hop | Unisolated. The live door is slot 0 south type `0x0A` (pos `$1D1C`), and DOWN leaves indoors. The forward gap is the pit. `hop.json` stayed in `0x98`. The hop is an isolated poke. |

`recordings/probe_AP-171/entry.py` saved the pin after `trigger_entrance` wrote `$010E` and module `0x06`. `make_entry_pin.py` `ENTRIES` was left unchanged. Ether is in the kit. Casting it at the desert door is AP-177.

`recordings/probe_AP-171/hop.json` is one trial from `MiseryMireEntry`. Sword notes left zol HP at 4. Bait south (60 frames, link `(4216,5089)`) left slots 0–3 asleep at spawn on the dash rows. Hookshot reason `no hookshot ancilla` after 118 frames, item byte 1, Link HP 24 to 8 at `(4201,5056)`. `crossed` is false. End module `0x07`, room `0x98`, dungeon `0x0E`, follower 5. The gap recorded there is runway local `(192,312)`, lip x 272, land local x 336, heading RIGHT.

## Requirements

z3 tokens on the entrance gates and later nodes. The pin kit is a poke.

| Item | Why | z3 source |
|------|-----|-----------|
| hasProperMedallion (vanilla Ether) | entrance gap and exit; kit item, not the cast | `mire.yaml` gates `Misery Mire Entrance Gap`, `Misery Mire Exit` |
| MoonPearl | same gates | those gates |
| RescueZelda | same gates | those gates |
| canDash or canGrapple | entrance gap (or) | those gates |
| canLightTorches | compass chest and big-key chest | locations `Misery Mire - Compass Chest`, `Misery Mire - Big Key Chest` |
| BigKeyD6 | big chest, map, lobby, Vitreous | `requirements.keys` |
| KeyD6 | map chest and main lobby | `requirements.keys` |
| CaneOfSomaria | Vitreous | location `Misery Mire - Vitreous` |
| hasLightCone | Vitreous | location `Misery Mire - Vitreous` |

## Room table

Rooms reached with the keys on the scan pin (small keys 0, big key 0). Geometry stays in `alttp/maps/room_98.json`. Dumps: `Scan_mire_0x98`, `recordings/probe_AP-171/pin_probe_0x98.txt`.

| Room | Doors + types | Chests | Sprites | Notes |
|------|---------------|--------|---------|-------|
| 0x98 | slot 0 south entrance `0x0A` pos `$1D1C` tile (14,58) | none | five zols type 143 HP 4: (4352,4912), (4240,4928), (4288,4928), (4336,4928), (4224,4976) | Pin `Scan_mire_0x98` kind scan, module `0x07`, dungeon `0x0E`, link (4216,5080), layer 0, keys 0, big key 0, follower 5. |

Stopping lock, with small keys 0. The far side was not entered.

- Pit on rows 39–42, columns 36–41, in the pin_probe window (columns 7–60). Floor west of it holds the zols. Stairs sit at columns 49–50, rows 35–38, east of the pit. `hop.json` did not reach land local x 336 and did not change rooms. South slot 0 is the overworld backtrack: DOWN from `MiseryMireEntry` ended module `0x09`, indoors clear, dungeon `0xFF`, xy `(296,3368)`.

Chest-table room words from `recordings/probe_AP-171/entry.json`. None of these rooms were loaded.

| File | Room word | Item in `mire.yaml` |
|------|-----------|---------------------|
| `0xEA61` | `0x00A2` | KeyD6 Bridge Chest |
| `0xE9DA` | `0x00B3` | KeyD6 Spike Chest |
| `0xEA64` | `0x00C1` | CompassD6 |
| `0xEA5E` | `0x00C2` | KeyD6 Main Lobby |
| `0xEA67` | `0x00C3` | CaneOfSomaria Big Chest |
| `0xEA6A` | `0x00C3` | MapD6 |
| `0xEA6D` | `0x00D1` | BigKeyD6 |

## Keys and big key

No small key and no big key was taken in play. `Scan_mire_0x98` shows keys `$F36F=0` and big key `$F366=0`. The chest rooms above are the z3 locations.

| Location | Item | Address | Requires |
|----------|------|---------|----------|
| Misery Mire - Bridge Chest | KeyD6 | `0xEA61` | none |
| Misery Mire - Spike Chest | KeyD6 | `0xE9DA` | not ohko, or canInvul |
| Misery Mire - Big Key Chest | BigKeyD6 | `0xEA6D` | canLightTorches |
| Misery Mire - Compass Chest | CompassD6 | `0xEA64` | canLightTorches |
| Misery Mire - Map Chest | MapD6 | `0xEA6A` | KeyD6 and BigKeyD6 |
| Misery Mire - Main Lobby | KeyD6 | `0xEA5E` | KeyD6 and BigKeyD6 |

A poke does not count as a key that rose in play.

## Big chest

`Misery Mire - Big Chest`, item `CaneOfSomaria`, address `0xEA67`, chest-table room `0x00C3`, requires `BigKeyD6`. The room was not entered. The cane byte did not rise in play.

## Mechanics

The entrance gap in this room is the pit in `maps/room_98.json`. Dash and hookshot are the yaml `or`. The saved trial cleared neither the zols nor the pit.

`somaria_block` selects the cane and pushes ancilla `0x2C` until `$0646` rises or the block's tile is `0x23`, `0x24`, `0x25`, or `0x3B`. It does not write `$F350`. On `Scan_mire_0x98` that byte is 0. Both layers of the room have zero of those switch attributes. The closed pit was not walked.

`Whatif_Scan_mire_0x98_somaria` is a disclosed poke of `$F350=1`, not a chest rise. Follower stayed 5. Two pushes, magic `$F36E` `128→120` each time, `$0646` stayed 0:

- RIGHT ended ancilla `0x2C` at `(4272,5064)`, tile `(22,57)` attr `0x00`. `recordings/probe_AP-174/somaria_block.json`.
- UP from tile `(12,55)` attr `0x00` ended ancilla `0x2C` at `(4207,5031)`, tile `(13,52)` attr `0x00`. `recordings/probe_AP-174/somaria_up.json`.

The cane chest is still room `0x00C3`, which this scan did not enter. Boss-door edges were not isolated.

## Boss

`Scan_mire_boss` is kind poke. `mire.yaml` location `Misery Mire - Vitreous` has no room word. z3 enemy id 189 is sprite `0xBD` (hp 128). The dungeon sprite list at `$09:D62E` lists that type in one palace `0x0E` room, `0x90`. Warp submodule `0x15` from `Scan_mire_0x98` saved the pin. It is not a walked arena door and not an entry pin. `$F3CC` was not written. Follower stayed 5. Sword on the pin is 1.

Landing `(120, 5080)` is tile `(15, 59)` attr `0xF0` (closed door). Module `0x07`, submodule 0, dungeon `0x0E`, doorway `$006C=0`. Slot 0 south is type `0x18` shutter. No chests. Cardinal holds of 80 frames do not move. A disclosed stand on a live floor `0x00` cell, then UP for 20 frames, moves and keeps HP 24. That stand is not a walk off the shutter.

z3 `dmgFromLink` for Vitreous and the small eye (id 190, hp 48) is 0 at the fighter-slash index and 4 at the next index. Green-mail `dmgToLink` is 32.

| Field | Value |
|-------|--------|
| Sprite ids | one `0xBD` vitreous, slot 0. Thirteen `0xBE` vitreous eye. AP-175 idle had no `0xBF`. `recordings/probe_AP-176/recharge.json` logged one `0xBF` at frame 418; Link HP was still 24 at frame 520. |
| HP byte | Body `$0E50` reads 128. Each eye `$0E50` reads 48. |
| Phase byte | Body `$0D80` stayed 0, and `$0DB0` stayed 0. An eye `$0D80` goes 0 to 1 when it leaves the cluster: slot 6 at idle frame 37, slot 11 at frame 101. |
| What hurts it | Fighter B on an eye leaves 48. A 72-frame B charge released 16px or 24px south of a cluster eye sets `$0EA0=15` on frame 1 and drops that eye 48 to 44 on frame 7. A charge of 64 at dy 24 drops the eye sum 624 to 620; charges 40 and 56 leave 624 (`recordings/probe_AP-176/bait.json`). The same release on the body, at offsets 0 through 32px, left body HP 128 and flash 0. Writing every eye `$0E50` and `$0DD0` to 0, a disclosed poke, also left the body at 128 and ai 0. In-play spins left the body at 128. The body was not shown to take damage. |
| When | The eye drop is the spin release, not the slash. A moving eye at chebyshev 36 still had 48 after 16 release frames; it was 44 by the time it reached chebyshev 8, and that second charge killed Link. Idle on the shutter: slot 6 reaches chebyshev 7 at frame 164 and Link HP goes 24 to 0. One eye was not taken to 0. Standing 16px south of the body drops Link 24 to 16 with body flash still 0. |

Audit of `Scan_mire_boss`: hp `0x18/0x18`, kind poke, room `0x0090`, dungeon `0x0E`, module `0x07`, 0 incoherent. `boss_probe.py` 1800 idle frames: body HP stayed 128 and `$0D80` stayed 0; Link 24 to 0 at frame 164; no new sprite type. The frames after that death are not a live phase cycle. Evidence: `recordings/probe_AP-175/`.

`bosses/vitreous:fight` was scored twice from `Scan_mire_boss` idle 0. One disclosed stand at `(120, 5060)`, then the same south spin. Try 1 died at frame 817 with 3 swings at `(80, 5056)`, HP 24 to 0, hit vitreous eye. Try 2 at frame 400 had 2 swings, eye sum 612, body HP 128, then died at frame 750 with 3 swings at `(62, 5056)`, HP 24 to 0, hit vitreous eye. The heart container did not rise. Not 10 of 12. `$F3CC` was not written. Logs: `recordings/probe_AP-176/try1_stage.log` and `try2_stage.log`. `recharge.json` took the eye sum 624 to 600 in two swings and died at frame 536 at `(86, 5056)` with body HP 128.

Isolated poke. Not natural, not continuous, and not a STATUS claim.

z3 location `Misery Mire - Vitreous` requires `hasLightCone`, `CaneOfSomaria`, and `BigKeyD6`. Prize item on `Misery Mire - Prize` is `Crystal6`.

## Prize and exit

Prize in the yaml is `Crystal6` (addresses `0x120A2`, `0x53F48`, `0x53F49`, `0x180057`, `0x180075`, `0xC703`). The only leave measured from `MiseryMireEntry` is the south entrance backtrack: indoors cleared, module `0x09`, dungeon `0xFF`, xy `(296,3368)`, HP `0x18`. That glance is the overworld side of the entry door. The prize room was not entered.

## Save points

| Save point | Predecessor | What it proves |
|------------|-------------|----------------|
| `MiseryMireEntry` | HyruleCastleGrounds plus entrance index `0x27` (kind poke) | Indoors, module `0x07`, room `0x98`, dungeon `0x0E`. Development pin. Isolated. |
| `Scan_mire_0x98` | `MiseryMireEntry` (kind scan, frame 0) | Same room, south door, no chests, five zols. Keys 0. Big key 0. |
| `Scan_mire_boss` | `Scan_mire_0x98` via submodule `0x15` (kind poke) | Room `0x90` warp landing. Not a walked arena door. Not an entry pin. |

## Open reds

| Red | Evidence | Blocker |
|-----|----------|---------|
| AP-171 | `hop.json` `crossed` false, end room `0x98` module `0x07`. Sword HP stayed 4. Hookshot made no ancilla. | First forward edge is not green 2/2. |
| AP-173 | Scan reaches only `0x98` with keys 0. Big key `$F366` stayed 0. | Pit rows 39–42 columns 36–41. The chest rooms were not entered. |
| AP-174 | Two `somaria_block` pushes made ancilla `0x2C` and spent 8 magic. `$0646` stayed 0. `$F350` on the scan pin stayed 0. | No switch tile `0x23`/`0x24`/`0x25`/`0x3B` on either layer of room `0x98`. Cane did not rise in play. Boss door not isolated. |
| Vitreous body | Eye spin drops `$0E50` 48 to 44. Body HP stayed 128 after the disclosed eye clear and after the in-play spins (`probe_AP-176` sum 624 to 600; try 2 frame 400 sum 612). | What hurts the body was not measured. Eyes were not cleared in play. The `0x90` landing is closed door `0xF0` and does not walk. |
| AP-176 | Two idle-0 fights, frames 817 and 750, both vitreous eye, HP 24 to 0. Body HP 128 at the last live sample. | Heart did not rise. Not 10/12. |
