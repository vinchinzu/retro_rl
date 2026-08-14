# Level 9 — Death Mountain (route notes)

**Status:** backward endgame recon is live; the natural Level 9 route is still
unbuilt. Spectacle Rock is overworld `0x05`, the settled entrance is room
`0x76`, the final Patra room is `0x52`, Ganon is `0x42`, and Zelda is `0x32`.
The preserved endgame states are explicitly composed, route-ineligible
fixtures—not Clean or Survival route evidence.

**Beads:** `rr-sz8` (Level 9 epic), `rr-sz8.1` (pre-Ganon → credits recon).

Planning sources:

- [Zelda Dungeon — Level 9: Death Mountain](https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/level-9-death-mountain/)
- Local archive: [research/DUNGEON_WALKTHROUGHS.md](research/DUNGEON_WALKTHROUGHS.md)
- RAM: `ADDR_TRIFORCE`, `ADDR_RING`, `ADDR_ARROWS`, `ADDR_MAGIC_KEY`, bombs

All room IDs in the backward-recon section are **live**. Unvisited interior
route claims remain source planning until reached from their real predecessor.

## Backward endgame recon (live 2026-08-14)

```text
fixture-cleared final Patra 0x52
  ─UP─► Ganon 0x42 (object 0x3E)
  ─four registered Magical Sword hits─► brown ObjState nonzero
  ─Silver Arrow─► LastBossDefeated $0672 = 1
  ─collect Power Triforce / north door─► Zelda 0x32 (object 0x37)
  ─clear two guard fires / center trigger─► ending → rolling credits → final page
```

Repeatable proof:

```bash
uv run python nes/zelda_i/scripts/run_level9_ganon.py \
  --build-fixture --from-state Level9BeforeGanonReconFixture \
  --infinite-life --save-state --trials 1 \
  --tag l9_ganon_credits_recon
```

The build begins from live `Level9EntranceReconFixture`, uses the game room
loader for `0x52`, then explicitly writes the full inventory, removes the
final Patra, and opens the north door. Its report and every state provenance
sidecar set `track=recon_fixture`, `fixture_only=true`, and
`route_eligible=false`. These inventory/room/object/door writes are forbidden
by `ASSIST_CONTRACT.md`; they exist solely to develop the route backward.

Evidence: `recordings/l9_ganon_credits_recon.json`; screenshots
`l9_ganon_credits_recon_t0_{before_ganon,ganon_start,ganon_arrow_kill,ganon_defeated,zelda_room,ending_start,credits,final_screen}.png`.

---

## Gates / required capabilities

| Cap | RAM | Source role |
|-----|-----|-------------|
| **All 8 TF shards** | `ADDR_TRIFORCE == 0xFF` | Old Man allows passage; L9 content locked without |
| Bombs | `ADDR_BOMBS` | OW rock entrance + interior walls |
| Sword | preferably Magical | Combat density |
| Bow + arrows | `ADDR_BOW`, `ADDR_ARROWS` | Silver Arrow is arrow-type upgrade |
| **Red Ring** (dungeon) | `ADDR_RING` value 2 (source) | Damage quartered vs base |
| **Silver Arrows** (dungeon) | `ADDR_ARROWS` value 2 (source) | Only way to kill Ganon after stun |
| Magical Key (optional) | `ADDR_MAGIC_KEY` | Route splits: Magical Key path vs key-farm path |
| Red Potion | `ADDR_POTION` | Source strongly recommends full red before entry |

**Predecessor:** all of L1–L8 Triforce bits. OW bomb rock can be mapped
earlier; interior Old Man blocks without full TF.

**Do not** poke TF bits / Silver Arrows for Clean STATUS.

---

## Overworld

### Spectacle Rock / bomb entrance (source)

From start (ZD): **right, up×5, left, up×2, left×2**. Two large rocks; bomb
**just below the left rock** → cave / Level 9.

| Landmark | Source hops from start `0x77` | Hypothesized id | Live? |
|----------|-------------------------------|-----------------|-------|
| Bomb-rock screen (Spectacle Rock) | R U×5 L U×2 L×2 | **`0x05`** | **yes** |
| Nearby potion shop | one screen left of rock | **`0x04`** | no |

Hop arithmetic:

```text
0x77 →R→ 0x78 →U×5→ 0x28 →L→ 0x27 →U×2→ 0x07 →L×2→ 0x05
```

Live recon reached `0x05` through the authentic overworld scroll loader and
bombed the left rock to settle in Level 9 room `0x76`. The full natural walk
from the earned L8 predecessor remains unverified.

**Scaffold:** `level9_overworld.py` — `LEVEL9_ROCK_HOPS`, `has_full_triforce()`,
bomb-entry notes (controller TBD).

### Remaining natural-entry goals

1. Walk to rock screen from the real post-L8 predecessor.
2. Bomb the left rock with naturally held bombs and full Triforce.
3. Settle `level==9`, room `0x76` without inventory/progression writes.
4. Continue through the Old Man gate from that natural entry.

---

## Interior (source summary)

Two routes: **with Magical Key** (ZD §10.2) vs **without** (§10.3). Prefer
Magical Key path for automation (fewer key bottlenecks). Room IDs **unknown**.

### Magical Key path (condensed source)

| Phase | Action | Notes |
|-------|--------|-------|
| Entry | UP | 12 Keese optional |
| Old Man | full TF check | pass only if `triforce == 0xFF` |
| LEFT / bomb N | Lanmola | head hits; push left block → stairs |
| Underground | tunnel | |
| Like-Likes | protect Magical Shield | key RIGHT |
| Patra #1 | **skippable** | orbiting eyes; leave DOWN |
| Patra #2 | kill for **Map** | bomb walls continue |
| Wizzrobe / blocks | clear, push left block | stairs → **Red Ring** |
| Backtrack | Magical Key doors | Old Man bomb hint LEFT |
| Stairs chain | more Wizzrobes / Patra | |
| Item | stairs → **Silver Arrows** | required for Ganon |
| Final Patra | clear → door UP | |
| **Ganon** | stun then Silver Arrow | see below |
| Zelda | princess room | ending sequence |

### Ganon (live)

| Signal | Live value |
|--------|------------|
| Room / object | `0x42` / type `0x3E` |
| Scene phase | `$0445 == 2` during the fight |
| Initial HP | `$0485 + slot == 0xF0` |
| Sword sequence | `F0 → B0 → 70 → 30`; the next registered hit resets `F0` and enters brown |
| Brown | `ObjState[$00AC + slot] != 0`; engine seeds `0xFF`, first external post-step value is commonly `0xFE` |
| B item | `$0656 == 2` selects arrows (`1` is bombs) |
| Dying | `$042C + slot != 0` after Silver Arrow collision |
| Persistent kill | `$0672 != 0` (`LastBossDefeated`) |

Pulse A; holding it does not start the next sword swing. The controller chases
Ganon's live coordinates, waits 12 frames between sword pulses, then axis-aligns
and pulses the Silver Arrow on B. Collect the Power Triforce after the kill;
the north-door bit (`0x08`) then opens the path to Zelda.

### Red Ring / Silver Arrows RAM (source + Data Crystal style)

| Item | Address | Planned nonzero value |
|------|---------|------------------------|
| Ring | `0x0662` (`ADDR_RING`) | 1 = blue, **2 = red** |
| Arrows | `0x0659` (`ADDR_ARROWS`) | 1 = wooden, **2 = silver** |

Confirm values live before stop predicates rely on them.

---

## Zelda / ending stops (live)

Zelda room `0x32` contains Zelda object `0x37` and two guard-fire objects
`0x3F`. Clear the flames while walking to Link x=`0x70..0x80`, y=`0x95`;
the rescue switches to ending mode `0x13`.

Mode initialization reuses submode numbers, so mode + submode alone yields a
false early match. Require `$0011` (`IsUpdatingMode`) to be nonzero:

| Stop | Predicate |
|------|-----------|
| Rolling staff credits | `mode == 0x13 && is_updating_mode != 0 && submode == 3` |
| Final “Press Start” page | `mode == 0x13 && is_updating_mode != 0 && submode == 4` |

The verified replay first entered rolling credits at frame 3,395 and the final
page at frame 4,595 from `Level9BeforeGanonReconFixture` (1/1, Survival health
refill plus inherited fixture composition). `level9_ending_stop` accepts either
update-loop endpoint.

The proof preselected Silver Arrows in the fixture and reported
`selected_item_writes=0` during combat. Survival telemetry restored four
filled-heart units and reported zero progression/capacity writes; those
telemetry counters do not legalize the inherited fixture composition.

---

## Boss / item stop predicates

```text
level9_red_ring      — ADDR_RING == 2 (planned)
level9_silver_arrows — ADDR_ARROWS == 2 (planned)
level9_ganon_dead    — ADDR_LAST_BOSS_DEFEATED ($0672) != 0
level9_ending        — update mode 0x13 submode 3 (credits) or 4 (final)
```

Full-clear program stop is **not** `triforce & 0x80` alone (that is L8);
Death Mountain end is Zelda/credits after Ganon.

---

## Checkpoints

| State | When |
|-------|------|
| `Level9EntranceReconFixture` | live `level==9`, room `0x76`; composed full inventory |
| `Level9BeforeGanonReconFixture` | live final-Patra room `0x52`, Patra fixture-cleared, north open; requested start |
| `Level9GanonReconFixture` | room `0x42`, scene phase 2, Ganon type `0x3E` |
| `Level9GanonDefeatedReconFixture` | `$0672=1`, Power Triforce collected, north open |
| `Level9ZeldaRoomReconFixture` | room `0x32`, Zelda type `0x37` |
| `Level9EndingStartReconFixture` | ending mode `0x13` entered |
| `Level9CreditsReconFixture` | update-loop submode 3, visible staff credits |
| `Level9FinalScreenReconFixture` | update-loop submode 4, final Press Start page |
| `Level9RedRing` | after Red Ring |
| `Level9SilverArrows` | after Silver Arrows |

Every `*ReconFixture` state has a `.provenance.json` sidecar that warns it is
development-only and not a natural-entry checkpoint.

---

## Runners / probes

```bash
uv run python zelda_i/scripts/probe_level9_entry.py --plan-only
uv run python zelda_i/scripts/run_level9_ganon.py --build-fixture \
  --infinite-life --save-state --trials 1 --tag l9_ganon_credits_recon
```

Modules: `zelda_i/level9_overworld.py`, `zelda_i/level9_ganon.py`.

---

## Evidence boundary

- Live: Spectacle Rock `0x05`; entrance `0x76`; final Patra `0x52`; Ganon
  `0x42`; Zelda `0x32`; combat states; credits and final-screen stops.
- Fixture-only: full inventory, final-Patra removal, north door opening, and
  Link/loader composition. These are enumerated in the JSON report.
- Not live from predecessor: natural Level 9 interior, Red Ring, Silver Arrow,
  and final Patra acquisition/clear path.
- TF bit map: shards 1–8 = bits `0x01`…`0x80`; full = `0xFF`.
