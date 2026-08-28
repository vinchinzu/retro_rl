# Seven-hop contractor research

Date: 2026-08-27. Practice-ROM **contractor** only. Success is **Join**
(`is_join` / `LeaveSpec` / `hop_glance`), not room-id. Living tip stays
Phantoon. Do not edit `STATUS.md` or `DEFAULT_CONTINUOUS_TIP`. Do not dual a
spine hop from this note. ADR:
[`docs/adr/0009-generalist-contractor.md`](../adr/0009-generalist-contractor.md).

## Conclusion

The overnight mix plateau (9/16 Join, the seven at 0/8) is not one learning
problem, and it is not “nearest-door forever.” The live tree already replaced
overnight’s nearest-clip-9 + `+256` potential with Goal-door routing and a
monotone room-stage distance (`generalist/steering.py`). A no-training
heuristic probe then **opened the Climb door, entered Pit, and still 0/8
Join** by falling into the pit and jumping at the wall for 1,800 frames.

So the remaining defect is **in-room geometry the 13×13 occupancy teacher
cannot sequence** (platform gap, shot-block morph, bomb wall, long climb) —
plus one editor-graph hole that makes Climb→Morph a 11-door scenic route.
More mix PPO on occupancy + nearest-door, or on the new Goal-door reward
without a room option, optimizes the wrong task.

None of the seven pins is loadout-gated.

## Overnight vs the live tree

Treat these as two different contractors. Observation **layout** is still
`OBS_DIM=226` (digest `0717ecb7…` matches overnight). Reward **is not**.

| | Overnight `best.zip` / `train_crateria_s1.json` | Live tree |
|---|---|---|
| Cross-room target | `potential_xy` nearest clip-9 | `steering_target`: bounded Goal-door, else nearest-door fallback |
| Distance | Euclidean + 256 off-room | local Euclidean + `remaining_doors * 4096` |
| Reward digest | `76d15e03…` (contract v1) | `d1b47de0…` (contract v2 in `obs.reward_contract`) |
| Teacher | occupancy-row walk/jump | same, plus cross-room horizontal shoot and transition idle |
| Probe | none per fail | `models/generalist/diagnostics/probe_pit_room_heuristic.json` |

`best.zip` may still *predict* (obs dim unchanged) but must not be *resumed*
onto the new reward, and must not be scored in the new env as if the steer
fill were the same. The overnight 9/16 table is not a published result.

DoorPlan as a *design* is already behind the `steering.py` seam. Do not
re-implement it inside `env.py`.

## Classification

Catalog pins: `maps/practice_repertoire.json`. Items: `0` none, `4` Morph,
`4100` (`0x1004`) Morph+Bombs (`ram.py` `MORPH_BALL_MASK` / `BOMBS_MASK`).
Goal is always the **next** repertoire pin (`goal_from_session`).

Steering at each **start** pin (ROM-free, live `editor_door_edges` +
`steering_target`, 2026-08-27):

| Hop | Pin → Goal | Start steering | Classification | Proved vs inferred |
|---|---|---|---|---|
| `pit_room` | Climb `0x96BA` (475,2187) items **0** → Morph **pedestal** `0x9E9F` (1408,680) pose 0 (`LeaveSpec` pose class **any**, not a door) | **`nearest_door` (504,2184)** — graph Climb→Morph is **11** doors the long way (`0x96BA→0x92FD→0x91F8→…→0x9E9F`) because Climb’s `doors[]` has **no** dest to Pit `0x975C`. `MAX_ROUTE_DOORS=3` then falls back. Collision still has clip-9 at (504,2184) (the Pit door); that is **not** the Tourian grey at (8,2168). | **walkable floor / pit-gap**; door-cap solved; editor-graph hole at start | **Proved:** Pit left→right Base requires nothing (`Pit Room.json` link `[1,2]`). First-visit right door is unlocked without Morph+missiles (`Pit Room.json` right-lock note). Heuristic N=8: enter Pit f68 (`target_kind=goal_door` to (760,120) = Pit-right → elevator `0x97B5` → Morph pedestal), then stall-walk at **(171,187)** pose 137 (stand) until timeout 1800. RGB: `diagnostics/rgb/pit_room_e00_f1800_final.png` is Samus **in the pit**, platforms above. Occupancy treats clip 1 ledges as solid and clip 8 pit walls as solid, so “blocked → jump-right” never climbs out. Wiki: ledgegrabs optional; jumps are the first-visit policy. **Not** loadout. **Not** post-entry wrong-door (live tree). |
| `construction_zone` | Morph `0x9E9F` (1964,651) items **4** → First Missile `0xA107` (85,139) | **`goal_door` (2040,632) → `0x9F11`**, 2 doors | **morph + shot-block fall**; door-opening | **Proved:** top→bottom Base requires Morph (`Construction Zone.json` link `[1,3]`). Pin has Morph. Human KPDR: crouch, shoot the floor, morph, downback while falling. Editor collision marks those tiles **clip 12**, which `is_solid` treats as occupied forever (`solid.py`). Steering already picks Construction’s **bottom-left** `(8,376)` over the entry (`test_generalist_steering.py`). Overnight 1800f timeout: inferred wander on the top floor (heuristic prefers leftover `rel_x` run+shoot over Down when the Goal door is below). |
| `construction_zone_revisit` | First Missile `0xA107` (85,139) items **4** → blue elevator `0x97B5` (128,136) | **`goal_door` → Construction**, 3 doors (`0xA107→0x9F11→0x9E9F→0x97B5`) | **morph + shot-block climb**; door-opening | **Proved:** bottom→top Base also requires Morph (link `[3,1]`). Same clip-12 floor, now going **up**. Not loadout-gated. |
| `climb_up` | Pit `0x975C` (131,139) items **4** → Climb **top** `0x96BA` (416,91) | **`goal_door` Pit-left → Climb**, then **`kind=join`** in Climb | **climb** (9-screen, no item gate) | **Proved:** Climb bottom-right → Main Junction → top Base requires `[]` (`Climb.json` links `[5,6]` and `[6,1]`). After the first door the potential is exact Join xy, not a door. Overnight 1800f + 8 Survival refills: inferred pirate contact during a failed ascent. 13×13 grid is ~208 px; Climb is 144 tiles tall. |
| `parlor_revisit` | Climb top `0x96BA` (416,91) items **4** → Flyway pin in Parlor `0x92FD` (873,619) | **`goal_door` Climb-up → Parlor**, then **`kind=join`** | **climb / parlor traverse** | **Proved:** Parlor is the Goal room after one door. `/tmp/parlor-revisit-f040.png` (and f060) show Samus in the **lower Parlor shaft** with Morph, not at a wrong door. Overnight 1800f + 16 refills. Wiki: walljump ≈ ledgegrab up Climb/Parlor. Parlor `doors[]` omits Flyway; irrelevant here because Join xy is *in* Parlor. |
| `alcatraz` | Flyway `0x9879` (64,139) items **4100** → Terminator pin in Parlor `0x92FD` (277,153) | **`goal_door` Flyway-left → Parlor** | **bomb/crumble + morph escape** | **Proved:** pin is Morph+Bombs. Base Alcatraz escape is `h_bombThings` (Morph and Bombs or one PB) (`Parlor and Alcatraz.json` link `[5,8]`; `helpers.json`). Alt: precise walljump + instant midair Morph (noob path does not need this). Occupancy cannot mark opened bomb tiles. Obs has a Morph **item** bit, **no Bombs bit** (`obs.samus_vector`). Teacher never lays a bomb. |
| `green_pirate_shaft` | Terminator `0x990D` (99,667) items **4100** → green elevator `0x9938` (126,139) | **`goal_door` (8,632) → GPS `0x99BD`**, 3 doors (`0x990D→0x99BD→0x9969→0x9938`) | **vertical descent + tank**; overnight wrong-door is **fixed in the tree**, unprobed live | **Proved:** KPDR is run-through / tank; logical `[2,4]` “Tank the Damage” is 5 pirate hits, then `[4,3]` / `[3,4]` Base `[]`. Survival refills make tank legal. Morph is only for the **upper** crumble, which this route never visits. `test_generalist_steering.py` asserts GPS `(224,1163)` steers to **bottom-left** `(8,1656)→0x9969`, not the entry. Overnight 531f stall was nearest-door to the middle-right; that mechanism is gone, Join is **not** thereby proved. |

Stall-vs-timeout is not a taxonomy. Pit overnight stalled at the Climb **cap**
(~259f); the new teacher shoots, enters, and **times out in the pit**. Same
hop, two failure modes, one geometry.

Curriculum poisoning (9 easy hops drowning the seven) can dilute a gradient
but cannot repair a pit wall, a clip-12 floor, or a bomb tile the grid never
opens.

## Ranked bets

Keep new behaviour behind a **small interface**, not more branches in
`env.py` (already the occupancy + Survival + Join host; steering already
extracted). `OBS_DIM` stays 226 unless a versioned schema is explicit, and
then `best.zip` cannot resume.

1. **Practice-only Chip / option per failure family** (top). One hop, RAM
   policy, Join via `LeaveSpec`. Families: Pit top-ledge jumps; Construction
   shoot-then-morph through clip-12; Alcatraz `h_bombThings`; Climb/Parlor
   ascent. Interface idea: `option(state, waypoint) -> action | done` in a
   new module (for example `generalist/options.py`), called from the env
   when the waypoint says so. The net keeps the open-floor spans. This is
   a **contractor Chip**, not a spine Skill and not a tip. Static occupancy
   can stay frozen; opened-block tracking, if added later, is a new obs
   version.

2. **In-room waypoints on the editor grid**, not only room-graph doors.
   Pit: stay on the clip-1 ledge row until x→right door. Construction: the
   shot-block column, then the bottom door. Seam: extend `SteeringTarget`
   (already `kind ∈ {join, goal_door, nearest_door}`) rather than a second
   parallel potential. Still needs an action that can actually take the
   waypoint (bet 1).

3. **Repair Climb’s missing Pit edge in the editor export** (hygiene).
   Today Climb `doors[]` is Parlor + Climb Super grey/yellow + dummy
   `0xDEDE`. Shortest path then walks Climb→Parlor→Landing→… (11 doors).
   **Do not raise `MAX_ROUTE_DOORS`** to make that path legal — it points
   *away* from Pit. Filling the dest does not Join pit_room; the probe
   already entered Pit via nearest-door luck.

4. **Hard-seven sampling** only after one family has a green N=8 Join.
   Freeze the nine solved hops as a regression eval. This is gradient
   hygiene, not a substitute for (1).

## Cheap probe for bet 1

Pit-room door-cap is already measured. Do **not** re-run overnight PPO.
Do **not** score `best.zip` in the new steer fill and call it a regression.

Throwaway, one pin, N=8, practice ROM, no `learn`:

Control (already on disk): heuristic `kpdr25/crateria/pit_room` → 0/8
timeout, final `(171,187)` in `0x975C`, RGB in
`models/generalist/diagnostics/rgb/`.

Treatment: a short **top-ledge jumper** (stay near collision row y=9 clip-1
platforms; jump the air gaps; do not drop to y≥11). Reuse
`generalist.diagnose` so traces still log `is_join` separately from
room-id. Same pin, `--episodes 8`, `--capture-rgb`.

```bash
# control already captured; treatment is the option under test, not PPO
uv run python -m super_metroid.generalist diagnose \
  --session kpdr25/crateria/pit_room --policy heuristic --episodes 8 --capture-rgb
```

Readout: Join rate against the Morph `LeaveSpec`; also whether the body
stays above the pit (y≲160) and whether Pit-right `0x97B5` is entered.
If the jumper Joins, the contractor needed an option, not more mix steps.
If it crosses the pit but dies at the elevator, the next waypoint is
elevator-down, still not PPO. If it cannot Join even scripted, stop and
glance the Morph pin (`x=1408,y=680`, pose 0) — that is a LeaveSpec
problem, not a net problem.

Second cheap characterization (no new policy): the same diagnose command
on `kpdr25/crateria/construction_zone` with `--capture-rgb`. Expect
Goal-door to Construction, then a sit on clip-12. That confirms family 2
before writing a shoot+morph option.

`green_pirate_shaft` diagnose N=8 is the check that overnight’s wrong-door
stall is gone; it is **not** the top bet.

## Explicit rejections

- **Ship-only PPO.** Same-room occupancy is the solved floor
  (`train_same_room_s1.json` Join 1.0). It never sees a pit gap, shot
  block, bomb wall, or 9-screen climb.
- **More mix PPO on occupancy + nearest-door.** That is the overnight
  plateau. Four of the seven also had anti-progress nearest-door; Pit
  still fails after that was replaced.
- **More mix PPO on the new Goal-door reward without an option.** The
  live teacher already has Goal-door + door-cap shoot and is 0/8 on
  pit_room.
- **Raise `MAX_ROUTE_DOORS` so Climb→Morph routes.** The 11-door path is
  the wrong way around Zebes.
- **In-place obs mutation / Bombs bit on `best.zip`.** Version the
  schema; do not resume.
- **STATUS or `DEFAULT_CONTINUOUS_TIP` from a contractor Join.**

## Sources

- Practice pins / items / xy: `snes/super_metroid/maps/practice_repertoire.json`
  (kpdr25 Crateria block starting ~line 14348). Official preset mirror:
  [sm_practice_hack `kpdr25_data.asm` @ `181c76b`](https://github.com/tewtal/sm_practice_hack/blob/181c76b1a5e6e86eef6e1b1e9ba82c8a6c38e1f6/src/presets/kpdr25_data.asm#L159-L355).
- Room logic: local `snes/super_metroid/refs/sm-json-data/` —
  `region/crateria/central/{Pit Room,Climb,Parlor and Alcatraz,Flyway}.json`,
  `region/brinstar/blue/Construction Zone.json`,
  `region/crateria/west/Green Pirates Shaft.json`,
  `connection/crateria/{central,west}.json`, `connection/brinstar/blue.json`,
  `helpers.json` (`h_bombThings`).
- Human technique (not topology): [KPDR Room Strategies](https://wiki.supermetroid.run/KPDR_Room_Strategies)
  (Pit jumps/ledgegrabs; Construction crouch+shoot+morph; Climb/Parlor
  walljump≈ledgegrab; Alcatraz bomb timing; GPS run-through/tank).
- Editor collision / doors: sibling
  `snes_editor/.../export/sm_nav/rooms/room_{96BA,975C,9E9F,9F11,99BD,92FD,9879,990D}.json`
  via `SUPER_METROID_EDITOR_NAV`. Clip 9 = door (walkable); clip 12 = shot
  block (occupied forever). Bank `$7F` clipdata is not on this practice core.
- Live contractor: `snes/super_metroid/generalist/{steering,solid,env,evaluate,obs,goals,diagnose}.py`.
- Overnight snapshot (do not copy into STATUS):
  `models/generalist/overnight/{status,train_crateria_s1,PLAN}.json`.
- Pit probe: `models/generalist/diagnostics/probe_pit_room_heuristic.json`.
