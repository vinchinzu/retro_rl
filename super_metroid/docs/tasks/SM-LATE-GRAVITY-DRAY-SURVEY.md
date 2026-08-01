# TASK SM-LATE-GRAVITY-DRAY-SURVEY: diagnostic survey from Gravity through Space Jump

## Recipe step
docs / diagnostic-only topology survey; no controller, policy, progression, or
continuous wiring

## Scope and result

**Result: BLOCKED.** This survey touched the 18-room downstream route from
Gravity Suit through Botwoon, Draygon, and Space Jump.  The 17 ordinary-room
arrivals below are **development-only observations** from existing dev anchors;
they are not clears, natural entries, item collects, or route progress.

The route boundary is K6/K7 in `docs/routes/ROUTE_KPDR.md`:

```text
Gravity (0xCE40) → West Ocean → Moat → Crateria Kihunter → Elevator
→ Caterpillar → Red Fish → Mt. Everest → Crab Shaft → Aqueduct
→ Botwoon Hallway → Botwoon → E-Tank → Halfie → Colosseum → Precious
→ Draygon → Space Jump (0xD9AA)
```

`0xC98E` Bowling Alley is the immediate pre-Gravity boundary, not one of the
18 downstream rooms.  It is included only where SEG-08's source requirement is
needed to explain why Gravity remains partial.

## Read first

- `AGENTS.md`
- `docs/SOURCE_STATES.md`
- `docs/routes/ROUTE_KPDR.md` (K6–K7)
- `docs/research/PATH_ROOM_BOARD.md`
- `docs/BOSS_PIPELINE.md`
- `docs/tasks/SM-ROOM-SEG-08-residual.md`
- `docs/tasks/SM-ROOM-SEG-20-residual.md`
- `docs/tasks/SM-ROOM-SEG-21-residual.md`
- `docs/tasks/SM-DRAY-01.md`, `docs/tasks/SM-DRAY-02.md`,
  `docs/tasks/SM-WRAP-DRAY.md`

## Evidence rules

- **Observed** = the named dev-only probe entered the room and settled in
  `gameState=8`.  It does not establish movement, natural entry, items, bosses,
  or continuous readiness.
- **Partial** = an isolated room attempt reached a meaningful boundary but
  could not test the real item/event under a capability-valid source.
- **Blocked** = the remaining question is a natural boss/event or item-closeout
  gate, so another dev warp would not produce useful route evidence.

All three available late anchors were inspected.  They carry development
loadout `items=0xF32F`, `beams=0x100B`, and boss bits
`[4,3,1,1,3,0,1,0]`; this includes Gravity, Space Jump, and the relevant boss
bits.  They cannot prove a PLM delta, a legitimate boss defeat, or a natural
capability set.

## Actual probe evidence

### A — Gravity → Botwoon (11 diagnostic door hops)

```bash
uv run python super_metroid/scripts/probe/route.py leg gravity_suit botwoon \
  --source-state super_metroid/custom_integrations/SuperMetroid-Snes/dev_route_anchor_gravity_suit.state
# exit 0
# success=true hopsCompleted=11 finalRoomIdHex=0xD95E developmentOnly=true
```

The standard probe reported these ordinary-gameplay target pins (pose was not
emitted by this CLI):

```text
0x93FE x=1720 y=853; 0x95FF x=640 y=85; 0x948C x=840 y=85;
0x962A x=120 y=180; 0xA322 x=128 y=44; 0xD104 x=120 y=180;
0xD0B9 x=632 y=40; 0xD1A3 x=120 y=180; 0xD5A7 x=120 y=203;
0xD617 x=120 y=356; 0xD95E x=100 y=187 (all gameState=8).
```

### B — Botwoon → Draygon (5 diagnostic door hops)

```bash
uv run python super_metroid/scripts/probe/route.py leg botwoon draygon \
  --source-state super_metroid/custom_integrations/SuperMetroid-Snes/dev_route_anchor_botwoon.state
# exit 0
# success=true hopsCompleted=5 finalRoomIdHex=0xDA60 developmentOnly=true
```

```text
0xD7E4 x=120 y=180; 0xD913 x=120 y=210; 0xD72A x=120 y=210;
0xD78F x=120 y=187; 0xDA60 x=576 y=187 (all gameState=8).
```

The command's `botwoon` source setup applies the dev loadout and Botwoon boss
bit inside its disposable emulator session.  That setup is specifically why
the post-Botwoon arrivals are observations rather than evidence of a defeated
Botwoon.

### C — Draygon doorway → Space Jump (one bounded diagnostic)

```bash
uv run python - <<'PY'
import json
from pathlib import Path
from super_metroid.dev.common import boot_from_state, door_warp, make_dev_env
from super_metroid.dev.route_dev import run_leg
from super_metroid.rooms.entry_bootstrap import build_entry_door_map

env = make_dev_env()
try:
    initial = boot_from_state(
        env,
        Path("super_metroid/custom_integrations/SuperMetroid-Snes/dev_route_anchor_botwoon.state"),
    )
    leg = run_leg(env, "botwoon", "draygon")
    door = build_entry_door_map()[(0xDA60, 0xD9AA)]
    final = door_warp(env, door, expected_room=0xD9AA)
    print(json.dumps({
        "developmentOnly": True,
        "initialRoom": f"0x{initial.room_id:04X}",
        "initialItems": f"0x{initial.collected_items:04X}",
        "botwoonToDraygon": leg["success"],
        "spaceJumpDoor": f"0x{door:04X}",
        "final": {
            "room": f"0x{final.room_id:04X}", "gameState": final.game_state,
            "phase": final.phase.name, "x": final.samus_x, "y": final.samus_y,
            "pose": final.pose, "doorTransition": final.door_transition,
            "items": f"0x{final.collected_items:04X}",
        },
    }))
finally:
    env.close()
PY
# exit 0
# initialRoom=0xD95E initialItems=0xF32F botwoonToDraygon=true
# Space Jump door=0xA978
# final room=0xD9AA gameState=8 phase=ORDINARY_GAMEPLAY
#       x=264 y=187 pose=155 doorTransition=0 items=0xF32F
```

The catalog edge identifies `0xDA60 → 0xD9AA` as a gray/local lock requiring
`draygon_defeated`; the probe forced that known door definition against an
already boss-complete dev state.  It did **not** fight Draygon or collect Space
Jump.

### Anchor inspection

```bash
uv run python -c '<boot each dev_route_anchor_{gravity_suit,botwoon,draygon}.state and print parsed RAM>'
# exit 0
# gravity: 0xCE40 after 5f, but still gameState=11 / ROOM_TRANSITION;
#          after 180f it was 0xC98E gameState=11, doorTransition=1
# botwoon: 0xD95E gameState=8 x=100 y=187 pose=155
# draygon: 0xDA60 gameState=8 x=576 y=367 pose=155
```

The Gravity anchor is therefore usable only for the forced topology probe; it
is not an ordinary, controllable post-Phantoon/pre-Gravity source.

## Per-room status

`D-G` means probe A's `dev_route_anchor_gravity_suit`; `D-B` means probe B's
`dev_route_anchor_botwoon`.  Each is full-loadout/boss-complete diagnostic
input, not a real source.  “Natural requirement” combines K6/K7, the physical
edge graph, and boss catalog requirements; absent graph requirements are not a
claim that the room has no movement geometry.

| # | Room | Status | Source / natural capability requirement | Actual evidence / final pin |
|--:|------|--------|-----------------------------------------|-----------------------------|
| 1 | `0xCE40` Gravity Suit (WS) | **partial** | Need a controllable natural post-Phantoon source at `0xC98E`, WS powered (`boss_bits[3] & 0x01`), with Gravity still uncollected. | `D-G` loads only a transition. SEG-08 reached `0x93FE` but item delta stayed `0x1004`; residual pin `0x93FE pose=12 x=1496 y=907 door_transition=0`. |
| 2 | `0x93FE` West Ocean | **observed** | Real predecessor is Gravity exit; K6 loadout must already include the movement/items earned through Alpha PB plus Gravity. | A: `0xA300` → `0x93FE`, `gameState=8 x=1720 y=853` (pose unavailable). |
| 3 | `0x95FF` Moat | **observed** | Gravity-lineage source; real K6 uses Speed + Hi-Jump for its intended crossing. | A: `0x89CA` → `0x95FF`, `gameState=8 x=640 y=85`. |
| 4 | `0x948C` Crateria Kihunter | **observed** | Real predecessor Moat.  Its elevator exit requires Power Bombs in the physical graph. | A: `0x8ADE` → `0x948C`, `gameState=8 x=840 y=85`. |
| 5 | `0x962A` Elevator to Caterpillar | **observed** | Post-Kihunter; retain Power Bomb capability for the preceding `0x948C → 0x962A` edge. | A: `0x8A42` → `0x962A`, `gameState=8 x=120 y=180`. |
| 6 | `0xA322` Caterpillar | **observed** | Natural elevator predecessor; no additional physical-edge capability listed. | A: `0x8B02` → `0xA322`, `gameState=8 x=128 y=44`. |
| 7 | `0xD104` Red Fish | **observed** | Natural Caterpillar predecessor. | A: `0x90C6` → `0xD104`, `gameState=8 x=120 y=180`. |
| 8 | `0xD0B9` Mt. Everest | **observed** | Natural Red Fish predecessor; underwater traversal needs the post-Gravity movement state. | A: `0xA474` → `0xD0B9`, `gameState=8 x=632 y=40`. |
| 9 | `0xD1A3` Crab Shaft | **observed** | Natural Everest predecessor; its Aqueduct exit requires Super Missiles in the physical graph. | A: `0xA468` → `0xD1A3`, `gameState=8 x=120 y=180`. |
| 10 | `0xD5A7` Aqueduct | **observed** | Super Missiles for `0xD1A3 → 0xD5A7`; real route also needs the post-Gravity water movement state. | A: `0xA4C8` → `0xD5A7`, `gameState=8 x=120 y=203`. |
| 11 | `0xD617` Botwoon Hallway | **observed** | Natural Aqueduct predecessor; carries the post-Gravity loadout into Botwoon. | A: `0xA72C` → `0xD617`, `gameState=8 x=120 y=356`. |
| 12 | `0xD95E` Botwoon | **blocked** | Must arrive naturally from `0xD617`; catalog recommends Supers/Missiles (Gravity preferred).  Fight/exit requires a real Botwoon defeat. | A: `0xA774` → `0xD95E`, `gameState=8 x=100 y=187`; anchor pin `pose=155`.  No fight was run. |
| 13 | `0xD7E4` Botwoon E-Tank | **observed** | Requires Botwoon defeat to make this post-boss path meaningful; no natural post-fight source exists. | B: `0xA918` → `0xD7E4`, `gameState=8 x=120 y=180`. |
| 14 | `0xD913` Halfie Climb | **observed** | Real predecessor is Botwoon E-Tank after Botwoon defeat; retain Gravity movement. | B: `0xA870` → `0xD913`, `gameState=8 x=120 y=210`. |
| 15 | `0xD72A` Colosseum | **observed** | Natural Halfie predecessor; the outgoing Precious edge requires Super Missiles. | B: `0xA8E8` → `0xD72A`, `gameState=8 x=120 y=210`. |
| 16 | `0xD78F` Precious | **observed** | Super Missiles from Colosseum; Draygon Eye Door carries `clear_local_lock`. | B: `0xA7F8` → `0xD78F`, `gameState=8 x=120 y=187`. |
| 17 | `0xDA60` Draygon | **blocked** | Catalog/graph: Gravity + `botwoon_defeated`; fight needs Supers/Missiles and a natural active entry. | B: `0xA840` → `0xDA60`, `gameState=8 x=576 y=187`; independent anchor pin `pose=155 x=576 y=367`.  No boss action or bit write is evidence. |
| 18 | `0xD9AA` Space Jump | **blocked** | Need `draygon_defeated`, gray-door closeout, and an uncollected Space Jump PLM. | C: catalog door `0xA978` → `0xD9AA`, exact pin `pose=155 x=264 y=187 door_transition=0`.  `items=0xF32F` already contains Space Jump, so no item delta was testable. |

## Related Maridia side-branch residuals (not counted as main-route rooms)

These were read because they constrain a future Maridia survey, but neither
side branch is substituted for the K7 main path above.

| Existing residual | Honest result | Capability/source blocker | Required next atomic card |
|-------------------|---------------|---------------------------|---------------------------|
| `SM-ROOM-SEG-20-residual` — Crab Tunnel `0xD08A` | RED; returned inside room, pin `pose=138 x=245 y=171` | Need controllable natural Gravity + at least one Super; the only Gravity anchors froze input or are full-loadout dev fixtures. | `SM-ROOM-SEG-20-R1`: capture/identify exactly that source before policy changes. |
| `SM-ROOM-SEG-21-residual` — Spring Ball `0xD6D0` | RED; returned through `0xD8C5`, pin `pose=10 x=984 y=139` | Need natural Gravity + Bombs while Spring Ball remains uncollected; suitless alternate requires X-Ray and a specialized clip. | `SM-ROOM-SEG-21-R1`: capture that doorway-natural capability set before policy changes. |

## Blocker register and one atomic next card per blocker

| Blocker | Why it blocks this survey from becoming played-route evidence | One next atomic card |
|---------|---------------------------------------------------------------|----------------------|
| Natural post-Phantoon/pre-Gravity source | `D-G` is full-loadout and transition-unstable; SEG-08 cannot evaluate the powered Gravity PLM. | **`SM-ROOM-SEG-08-R1`** — capture one ordinary `0xC98E → 0xCE40` source with Phantoon flag set and Gravity clear; rerun the unchanged policy. |
| Crab Tunnel side branch | Existing fixture has no Gravity/Super and cannot legally test the green-gate branch. | **`SM-ROOM-SEG-20-R1`** — capture one natural Gravity + Super source for the existing doorway, with no policy edit. |
| Spring Ball side branch | Existing fixture lacks Gravity; it reaches the return door without the item delta. | **`SM-ROOM-SEG-21-R1`** — capture one natural Gravity + Bombs/Spring-clear doorway source, with no policy edit. |
| Botwoon natural activation | A strategy scaffold exists, but `BOSS_PIPELINE.md` forbids continuous use before a natural K7 entry exists. | **`SM-BOTW-NATURAL-ENTRY`** (proposed) — capture only an active `0xD95E` state reached by play from `0xD617`; record room, inventory, and boss-bit provenance. |
| Draygon natural activation | SM-DRAY-01/02/WRAP-DRAY are dev-only strategy/wrap work; the required natural Maridia predecessor and fight proof are absent. | **`SM-DRAY-NATURAL-ENTRY`** (proposed) — capture only an active `0xDA60` state reached from a real Botwoon exit; do not tune or wire a controller. |
| Space Jump closeout | The dev state already owns Space Jump and has Draygon's bit, so it cannot test gray door, PLM, or fanfare. | **`SM-DRAY-CLOSEOUT-01`** (proposed) — from a natural Draygon completion, cross `0xA978`, collect the live PLM, and assert the item delta plus ordinary settle. |

## Acceptance

- [x] Touched all 18 scoped downstream route rooms diagnostically.
- [x] Recorded the actual commands, door IDs, ordinary-gameplay targets, and
  available final pins.
- [x] Distinguished dev warps/loadout/boss-bit setup from route evidence.
- [x] Preserved SEG-08/20/21 residual conclusions and their next atomic cards.
- [x] Did not change `STATUS.md`, `continuous.py`, `progression.py`,
  `routes/catalog.py`, `routes/kpdr/__init__.py`, or `registry.py`.
- [ ] Natural entry, item collection, boss defeat, continuous evidence, and
  route-ready clearance — blocked by the register above.

## Files changed

- `docs/tasks/SM-LATE-GRAVITY-DRAY-SURVEY.md` — this survey card only.

## Non-claims

- No continuous or STATUS GREEN claim.
- No door warp, forced placement, dev loadout grant, or boss-bit setup is
  presented as route progress.
- No progression, capacity, item, event, or boss RAM write was used as a
  route-success claim.
- No controller, policy, shared path code, or existing residual/card was
  edited.
