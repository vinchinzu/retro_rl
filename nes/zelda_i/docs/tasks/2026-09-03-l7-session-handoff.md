# 2026-09-03 L7 orchestration session — handoff

Orchestrator (Sonnet) ran L7-only subagents on branch
`zelda-beads-parallel-20260903`. A **separate parallel orchestration** (the
user's automation) was working the L8/L9/bait beads in the same working tree
throughout — this session stayed strictly in L7 after that was discovered
(user decision: "Coordinate — I take L7 only").

**Not pushed. Not on main. `bd` not written by this session. STATUS not
promoted.**

## Commits this session (L7 + repo hygiene)

| SHA | What | Verified |
|-----|------|----------|
| `b6fb1a86` | L7-B `Level7InteriorReconFixture` (walks `0x79→0x6B`, discloses Food/bombs/keys) + `0x6B→0x6C` DIGDOGGER_1 | 2/2 |
| `28dcb8cd` | L7-B recon scratch: `0x6B` UP = dead-end `0x5B`; `0x6A` has no bombable north wall | 2/2 |
| `d1564e0e` | L7-B east mainline `0x6C→0x6D` STALFOS_KEY, `0x6B→0x5B`, `0x69` west-bomb→`0x68` | 2/2 |
| `8ed59b95` | L7-B branch chain `0x69→0x68`(KEESE_TRAPS)`→0x58`(DODONGOS_UPGRADE) + recon fixtures | 2/2 |
| `1a7cb3ab` | L7-B `0x58 EAST → 0x59` GORIYA_COMPASS | 2/2 |
| `827e1c1d` | L7-B `0x59` UP nav solved, `0x59→0x49` GORIYA_BUBBLE; **`0x49` needs the Stepladder** | 2/2 walk; onward gated |
| `cb11be6d` | **Snapshot** of the concurrent L8/L9 lane's in-tree work (not this session's) | untested here |
| `330d08cb` | Merge agent C: full test suite **2 collection errors + 2 fails → 470–479 pass, 0 fail** | ✓ |

Repo-hygiene detail in `330d08cb`: restored public L3/L6 spine stop symbols
(`level3_dest_6b_stages`, `level6_east_key_success`) dropped in the package
split; restored the L2 diamond-east occupancy walk; audited the `847bce83`
checkpoint fragments (L8 topology + natural-entry rows kept + new
`test_natural_entry_segments.py`; L9 whitespace reverted).

## L7-B interior map — state after this session

```
row-6 corridor (all OPEN doors, all 2/2 fixture-live):
  0x79 ─N→ 0x69 ─E→ 0x6A ─E→ 0x6B ─E→ 0x6C(DIGDOGGER_1) ─E→ 0x6D(STALFOS_KEY, small key)
                     │                                          └─ dead-end (only LEFT)
  0x6B ─UP(x≈118)→ 0x5B(OLD_MAN_NOSE) ── dead-end (all walled)

candle mainline branch:
  0x69 ─WEST BOMB (stand (44,141) face LEFT)→ 0x68(KEESE_TRAPS: 4 blade traps 0x49 + 4 keese)
  0x68 ─UP(x=120)→ 0x58(DODONGOS_UPGRADE: 3× invuln 0x31 hp240, room_item_id 0x0f uncollected)
  0x58 ─EAST→ 0x59(GORIYA_COMPASS: goriya 0x05/0x06)
    0x59 ─RIGHT (KILL_CLEAR)→ COMPASS (dead-end pickup)
    0x59 ─UP (perimeter waypoint micro, boxes at (48,125) naively)→ 0x49(GORIYA_BUBBLE)
      0x49 ─DOWN→ 0x59 (backtrack, 1/1)
      0x49 ─UP→ DIGDOGGER_2 ── **GATED: full-width water moat ~y120 tile 0xF4, needs STEPLADDER**; L/R walled
```

Controllers wired for every 2/2 transition (`Room6CEastController`,
`Room68NorthController`, `Room58EastController`, `Room59*`,
`Room69WestBombController` on `dungeon.bomb_wall.BombWallController`, …), all
`route_eligible=false`, recon-only, NOT on the executable chapter chain.
Recon fixtures: `Level7Interior{6B,68,58,59,49}ReconFixture` — disclosed
writes are Food 0→1 + bombs/keys *count* top-ups only; every provenance is
`development_only:true` / `fixture_only:true` / `route_eligible:false` /
`natural_entry:false`.

## What is still NOT done for "L7 finished → get to L8"

1. **The Stepladder pickup room** — the candle mainline past `0x49` is
   ladder-gated. Find where the Stepladder is picked up (source hypothesis:
   `MOLDORM_KEY_OPT` or a side room off the `0x68`/`0x58` branch —
   `0x68` DOWN → ROPES_KEY and `0x58` → `0x48` BOMB_UPGRADE are seen but not
   2/2), or build a recon fixture that carries it, then walk
   `0x49 → DIGDOGGER_2 → GORIYA_PRE_HUNGRY → HUNGRY_GORIYA → MAP → … →
   CANDLE_PUSH → RED_CANDLE_CELLAR`.
2. **Hungry Goriya** consumes Food (the fixture carries Food 1). Still
   unobserved.
3. **Red Candle pickup** — room unobserved; `ADDR_CANDLE` 1→2 must be
   NATURAL, never poked.
4. **L7-C entirely hypothesis** — forced Digdogger, Aquamentus boss, shard,
   settled OW leave.
5. **L7→L8 leave packet UNMEASURED** — `PostLevel7Handoff.verified` must stay
   `false`. This is the real "get to L8" gate; the concurrent L8 lane needs
   `MEASURED_POST_L7_EXIT`.
6. L7-A pond-drain gate is still fail-closed (`--through level7-entry` greens
   through `level7_bait_purchase` on the Survival Food fixture, then
   fail-closes at `level7_pond_drain_entry` on OW `0x25`). The natural bait
   route is the concurrent lane's `rr-8t4.5` / `rr-8t4.4`.

## Exact next resume point (L7-B)

From `Level7Interior49ReconFixture` (settled `0x49` `(120,205)` S mouth,
keys 4 / bombs 7 / Food 1): the UP path is Stepladder-gated. First find the
Stepladder — recon `0x68` DOWN (→ ROPES_KEY) and `0x58` → `0x48`
(BOMB_UPGRADE, key-gated) cleanly 2/2, and check `MOLDORM_KEY_OPT`. Once the
ladder is in hand (real pickup or a disclosed fixture), cross the `0x49`
moat UP and continue the candle chain. Do **not** poke `ADDR_CANDLE`.

## Reference-only (not merged)

`worktree-agent-afbe9499e5cc76c3b` @ `bd9ac79f` — L7-A bait recon:
`PostSwordStart → forest band → 0x54 → 0x44 → 0x34` walkable (single run,
pre-L6 leg); `0x34` cave interior — Bait pedestal right x≈168, price 60,
Key 80 / Blue Ring 250, merchant `0x7a` at x=120. Left for the concurrent
bait lane to use.

## Test state

`uv run pytest nes/zelda_i/tests -q --ignore=test_level3_spine.py --ignore=test_level6_overworld.py`
→ **470 passed, 0 failed**. (The two ignored modules' collection errors were
fixed in `330d08cb`; full unignored run is 478–479 pass depending on the
concurrent lane's uncommitted L9 test file.)
