# Level 9 — Death Mountain (route notes)

**Status:** planning (gated). Full Triforce of Wisdom required for Old Man
gate; Silver Arrows required to finish Ganon. No live bomb-rock screen or
entry room in this repo yet.

**Beads:** `rr-c8v` (plan + TF gate).

Planning sources:

- [Zelda Dungeon — Level 9: Death Mountain](https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/level-9-death-mountain/)
- Local archive: [research/DUNGEON_WALKTHROUGHS.md](research/DUNGEON_WALKTHROUGHS.md)
- RAM: `ADDR_TRIFORCE`, `ADDR_RING`, `ADDR_ARROWS`, `ADDR_MAGIC_KEY`, bombs

All screen/room ids are **source-hypothesized** unless marked **(live)**.

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
| Bomb-rock screen (Spectacle Rock) | R U×5 L U×2 L×2 | **`0x05`** | no |
| Nearby potion shop | one screen left of rock | **`0x04`** | no |

Hop arithmetic:

```text
0x77 →R→ 0x78 →U×5→ 0x28 →L→ 0x27 →U×2→ 0x07 →L×2→ 0x05
```

Some secondary guides insert a “cross river / stairs” nuance mid-path; treat
`0x05` as **ZD-hypothesized** until live trail.

**Scaffold:** `level9_overworld.py` — `LEVEL9_ROCK_HOPS`, `has_full_triforce()`,
bomb-entry notes (controller TBD).

### Live recon goals

1. Walk to rock screen with bombs (infinite-life OK for geometry).
2. Save `OW_L9Rock` without needing full TF.
3. Bomb left rock → confirm cave / `level == 9` only if full TF path desired;
   otherwise map rock hole only.
4. Save `Level9Entrance` when `level==9` settle confirmed.

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

### Ganon (source)

1. Sword (or other damage) until Ganon turns brown / becomes hittable state
   (classic: hold still after enough hits → brown).
2. Fire **Silver Arrow** into brown Ganon to finish.
3. Heart / open path to Zelda.

Exact object type ids and brown-state RAM **TBD live**. Policy must equip
Silver Arrows on B (or menu select arrow type) — not A-only mash.

### Red Ring / Silver Arrows RAM (source + Data Crystal style)

| Item | Address | Planned nonzero value |
|------|---------|------------------------|
| Ring | `0x0662` (`ADDR_RING`) | 1 = blue, **2 = red** |
| Arrows | `0x0659` (`ADDR_ARROWS`) | 1 = wooden, **2 = silver** |

Confirm values live before stop predicates rely on them.

---

## Ending / credits stop ideas (planning)

No live ending probe yet. Candidate stop signals to investigate later:

| Idea | Signal | Notes |
|------|--------|-------|
| Zelda room entered | `level==9` + room id TBD | after Ganon |
| Triforce complete already | `triforce == 0xFF` | pre-entry, not ending |
| Mode change | mode raw ≠ 5 for long hold | credits / fanfare modes TBD |
| Dialog / timer | `ADDR_DIALOG_TIMER` | Zelda speech |
| Manual segment end | controller success after Zelda touch | safest first pure |

Do **not** invent a Clean “credits” stop until mode/room evidence exists.
Scaffold exposes `level9_ganon_planning_notes()` / placeholder
`level9_ending_stop` that always returns False with reason `"unverified"`.

---

## Boss / item stop predicates (stubs)

```text
level9_red_ring      — ADDR_RING == 2 (planned)
level9_silver_arrows — ADDR_ARROWS == 2 (planned)
level9_ganon_dead    — TBD object / room flags
level9_ending        — TBD; stub False
```

Full-clear program stop is **not** `triforce & 0x80` alone (that is L8);
Death Mountain end is Zelda/credits after Ganon.

---

## Checkpoints (planned names)

| State | When |
|-------|------|
| `OW_L9Rock` | Spectacle Rock screen mapped |
| `Level9Entrance` | `level==9` after bomb cave settle |
| `Level9RedRing` | after Red Ring |
| `Level9SilverArrows` | after Silver Arrows |
| `Level9GanonCleared` | after Ganon |
| `Level9Zelda` / complete | ending stop once defined |

---

## Scaffold / probe

```bash
uv run python zelda_i/scripts/probe_level9_entry.py --plan-only
# Dock/rock only (no TF required for OW map):
uv run python zelda_i/scripts/probe_level9_entry.py --rock-only --infinite-life --save-state
# Full entry attempt refuses without triforce == 0xFF unless --plan-only
uv run python zelda_i/scripts/probe_level9_entry.py --infinite-life --save-state
```

Module: `zelda_i/level9_overworld.py`.

---

## Evidence

- Source walkthrough + hop arithmetic only.
- No live L9 recordings yet.
- TF bit map: shards 1–8 = bits `0x01`…`0x80`; full = `0xFF`.
