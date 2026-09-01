# Plan — Super Metroid

Verified facts: [STATUS.md](STATUS.md). Language:
[`CONTEXT.md`](../CONTEXT.md). Assist:
[ASSIST_CONTRACT.md](ASSIST_CONTRACT.md). Tracker:
`bd ready -l super_metroid -l spine`.

**Doc home:** STATUS is verified facts; this file is future work; beads
are ready/in-flight. Session loop lives in `.grok/skills/sm-session/`
(not a QUEUE, not PROCESS.md). One living residual per open tip. Do not
rewrite the route or claim a new tip from a pin bench.

**Program role:** Beat vanilla Super Metroid with a **skill API**, equal token
weight with Harvest. Solver/SMZ3 is downstream. See
[`CONTEXT.md`](../CONTEXT.md) and
[`docs/SOLVER_ARCHITECTURE.md`](../../../docs/SOLVER_ARCHITECTURE.md).

## Strategy

**Survival** (energy + unlocked ammo) makes combat attrition secondary. The
hard problem is long-horizon navigation. The campaign is a skillset. TAS and
human **tapes** are guidelines; trash them once the skill exists.

One **living tip**. Pin is for building; rung green is power-on to that tip.
Scratch, Clean, and practice are parallel, not a second product.

**Chip** is one room: tape → skill. **Sync** is a clean tie into the next room
(doorway pause / a few frames allowed). If the seam will not join, both rooms
are one change. Full power-on dual at milestones (Gravity, new living tip,
credits) and before **Publish** — not every slop hop.

**Clear rooms by play.** Door-warps are topology diagnostics only. Recipe
([ARCHITECTURE.md](ARCHITECTURE.md)):

```text
tape/TAS guideline → hop dual-green → SpineHop → power-on compose → Sync next
```

**Boss fights stay deferred** until natural *entry* to that boss room exists on
the played chain. Pipeline: [BOSS_PIPELINE.md](BOSS_PIPELINE.md).

**Agent discipline:** `.grok/skills/sm-session/` (one bead, one knob, halt-3,
no STATUS from a pin). Do not relax for scale.

**Ticket size:** one hop, or both rooms of a failed seam; prefer 30–90 min
sessions. STATUS/docs updates are planner-owned.

---

## Current focus

| Priority | Work | Beads |
|----------|------|-------|
| **★ Product next** | Gravity on the Phantoon tip (power-on) | `rr-kw8t` |
| Living tip | `--to phantoon` **195,336f** ×2 | STATUS `rr-b926` |
| Parallel | Chip prefix slop under Sync | not a second tip |
| Parked | TAS/oracle, 100%, Clean spore+ | not `spine` |

**Rungs:** Phantoon (living tip) → Gravity → Maridia → LN+Ridley → Tourian+credits
→ rewrite toward sub-hour.

**Critical path:** Phantoon is the living tip. Next compose is Gravity from
`scratch/post_phantoon_leave.state`. Living residual:
[tasks/rr-kw8t-residual.md](tasks/rr-kw8t-residual.md).

Live work: `bd ready -l super_metroid -l spine`.
Source states: [SOURCE_STATES.md](SOURCE_STATES.md).

Parallel room work, human-tape bootstrap, permissive scaffold assists, and
safe ten-agent speed waves are specified in
[PARALLEL_SPINE_PLAN.md](PARALLEL_SPINE_PLAN.md). Read it before dispatching
work that starts from archived room pins, changes a room already on the spine,
or attempts a tape-backed continuous chain. Its scaffold track is development
evidence, not the Survival living tip.
---

## Ceres opener (TAS boot) + arm-pump — 2026-08-07

**Policy:** when a faster prefix desyncs a later leg, **re-solve with WRAM** —
do not blind-restore product open-loop. Speed every section; re-pin tails.

**Code:** `routes/kpdr/ceres/` + `early_spine.play_boot_to_ceres`. TAS movies under
`tas/ref/` (Sniq any% #3653M). Unit: `tests/test_ceres_arm_pump.py`.

### Improvement table — boot / first Ceres control

| Milestone | Frames | Δ frames | Δ seconds (@ 60.0988) | Notes |
|-----------|-------:|---------:|----------------------:|-------|
| Legacy open-loop boot (`_boot_spans` sum) | 10,860 | — | — | Fixed title idle 2,100f + intro mash |
| Legacy first `gs=8` elev (probe) | 10,642 | — | — | Control during last boot spans |
| **TAS-style mash first `gs=8`** | **8,479** | **−2,163** | **−36.0 s** | START/A period-1 → A-every-other (Sniq pattern) |
| TAS boot hop end (+elev settle +plant) | ~8,572 | **−2,288** vs 10,860 | **−38.1 s** | y 0→72 pad settle required for outbound |
| Sniq any% first B+RIGHT (ref movie) | 8,639 | — | — | lsnes movie; not same core |

### Improvement table — morph spine (published / probe)

| Milestone | Frames | Δ frames | Δ seconds | Notes |
|-----------|-------:|---------:|----------:|-------|
| Product morph (pre arm-pump) | 27,074 | — | — | `morph.json` |
| Arm-pump morph dual GREEN | **26,824** | **−250** | **−4.2 s** | ridley 16,181; landing 21,548 |
| **Reactive Ceres main dual GREEN** | **24,475** | **−2,349** vs 26,824 | **−39.1 s** | ridley 14,399; landing 19,199 |
| **TAS-close Ceres morph dual GREEN** | **24,187** | **−288** vs 24,475 | **−4.8 s** | ridley 14,113; landing 18,912; Ceres 1 **−2f vs TAS** |
| TAS boot → ridley (probe) | 13,671 | **−2,510** vs 16,181 | **−41.8 s** | Outbound holds after settle |
| TAS boot full morph dual | **27,494** ×2 | **+670** vs product | **+11.1 s** | Elev GREEN; BB elev + morph reseed cost |

### Improvement table — Ceres Ridley fight (same enter pin)

Public RTA: [wiki Ridley § Ceres Station](https://wiki.supermetroid.run/Ridley#Ceres_Station) — escape starts at energy **< 30**; five tail hits at the right wall. Skill: `combat/ceres_ridley.py`. Seconds @ 60.0988.

| Policy | Frames | Seconds | Clock | Hits | Notes |
|--------|-------:|--------:|------:|-----:|-------|
| wait (left-door idle) | 3,212 | 53.445 | 00:53.53 | 8 | Previous product (`ceres_ridley_natural_countdown`) |
| tail_tank (nudge-left 5th) | 1,936 | 32.214 | 00:32.27 | 5 | Right wall; 5th hit f1302 |
| tail_tank (hold-A 5th) | 1,666 | 27.721 | 00:27.77 | 5 | Hits 610/706/802/905/**1066**; elev fast WJ joined |
| **fresh_fifth_jump** | **1,611** | **26.806** | **00:26.85** | 5 | Product; hits …/**1018**; A-release spin. Elev 571-chain joins |
| Δ vs hold-A | **−55** | **−0.915** | −00:00.92 | | TAS 5th hit is x=235 y=77 at 994f |

**fresh_fifth_jump is product.** Same-pin fight bench is still
`scratch/ceres_ridley_bench.json`. The y=628 WJ still misses: the body
at (224,639)/(232,631) is Ceres-door overlay ``$E23F`` (touch = RTL, no
knockback), not steam. Steam is ``$E1FF``; hide/show is ``$0F88`` bit 2
and x/y stay put (64/64 idle grades). 571-chain no longer idles at 475/363
— land-pose 166 hops immediately and coasts the land. Takeoff windows live
in ``takeoff.PlatformHop``. KPDR Ceres Station goal is **1:35** from first
elev control. Do not STATUS-promote from the pin bench. Skill:
`routes/kpdr/early_spine.py` elev hop. Residual:
`docs/plan.md` § Ceres arm-pump.

### Elev re-pin findings (`rr-14u`, 2026-08-07)

| Fact | Detail |
|------|--------|
| Ledge pin | TAS vs legacy: **identical** WRAM at left seat (`x45 y571 pose 138 x_sub=0`) |
| Desync cause | **Absolute-frame debris phase** in shaft (not subpixel at pin) |
| Product shaft | Open-loop s2–s10 works when phase matches; thrash hops burn timer |
| Phase search | Idle **0** = legacy green; idle **14** = TAS elev clear (probe scan) |
| Falling | Keep product **walk** into door (arm-pump mid desyncs elev entry) |
| Human (Kentroid) | Shaft ≈432f mostly `LEFT+B` + short A (spin); not used as restore |
| Sniq TAS | Short `LEFT+A` / `LEFT+B` pulses near Ceres end — reference only |
| Happy medium | Product wall-spin spans + phase idle list + top residual; no hop thrash |

### Verified facts (still true)

| Fact | Detail |
|------|--------|
| TAS-close Ceres morph dual | GREEN **24,187f** ×2 (open-loop boot + sole Ceres policy; Ceres 1 **309 vs TAS 311**) |
| TAS morph dual | GREEN **27,494f** ×2 (probe; slower — not a fallback) |
| Falling→elev | Mid-trans y≈139 fake; **gs=8 → bottom y≈651** — do not LEFT-walk on ghost y |
| Elev top | x211 y171 pose 137 → LEFT+A → ship pad |
| Product boot | open-loop `_boot_spans`; `play_boot_to_ceres_tas` is probe-only |

### STATUS / next

- **Shipped:** tail-tank Ceres Ridley on the spine; elev platform-hop body
  (center 475 land; ship leave still open). BB elev parity
  retry + reactive board, morph seed pad-return reseed; falling timeout 700f.
- **Product is 24,187.** TAS-boot probe remains slower and is not a fallback.
  Ceres 1 beats TAS settled_gs8 by 2f (309 vs 311). First-control inbound is
  **1606f** ×2 (+83 vs TAS 1523). Ridley same-pin **1611f** ×2 (was 1666).
  Full station from first-control: countdown **3255f**, Ceres leave **5364f /
  01:29.40** (was 5419 / 01:30.32), landing **8052f** ×2 / **02:14.20**
  (was 8107 / 02:15.12). Elev leftover is still the 571 chain: pose-25 window
  is reached, y=628 WJ misses the ``$E23F`` door overlay.
- **Follow-on:** TAS WJ is still the leftover vs 2246f elev_to_landing
  (**3239 / +993**). Pin is falling (vy=+4 at y=637); LEFT-into dumps the
  pit. Absorb shown steam (``$E1FF``, bit 2 clear) as a d-boost; do not idle
  debris. Ceres 2 exit-hop/steam, Ceres 3 jump-1, Ceres 4 x=467 stall. Do not
  STATUS-promote from the pin bench.

### Reproduce

```bash
uv run python snes/super_metroid/scripts/record/continuous.py --to morph --no-video
# TAS boot probe only: play_boot_to_ceres_tas (not product)
# TAS movies: uv run python -m super_metroid.tas.fetch_refs
```

### TAS 100% reference foundation (2026-08-07) — not STATUS

The old snes9x annotation is research-only Ceres thrash. Native lsnes now
replays Sniq 100% #4010M on its authoring bsnes v085 core and reaches Landing
GREEN at f15198. The compact reference is
`tas/bodies/sniq_100_ceres_lsnes_hops.json`; the gitignored full oracle is
`recordings/tas_oracle/sniq_100_lsnes/`. This is comparison evidence, never
STATUS or an open-loop product body. See
[`tasks/LSNES_100_PLAN.md`](tasks/LSNES_100_PLAN.md).

### Inbound Ceres TAS alignment audit / plan — 2026-08-31

**Scope:** first settled Ceres elevator control → settled Ridley door, no
fight and no reverse. The result is now also composed through the promoted
continuous Morph prefix; Gravity and boot style remain out of scope.

Current main-line inbound is `play_ceres_to_ridley_door`: Ceres 1 moonfall,
Ceres 2 magnet-feet, Ceres 3 jump-before-ledge, Ceres 4 no-jump run, then Flat.
The old room-1/2/3 tape and its switch are deleted; this reactive chain is the
only main policy. Natural-predecessor baseline from
`scratch/ceres_first_control.state` is **1606f** ×2 to Ridley `(39,139) p17`
(was 1626f). Maintained comparison:
`recordings/room_timings/ceres_vs_tas.json` on `settled_gs8`.

| Hop (`gs=8` → `gs=8`) | Product | TAS | Δ | First residual |
|---------------------------|--------:|----:|--:|----------------|
| Elevator → Falling | 309 | 311 | −2 | Hold; already aligned |
| Falling → Magnet | 301 | 282 | +19 | Magnet-feet plants y=171; leftover is exit-hop + door steam |
| Magnet → Scientist | 377 | 345 | +32 | Jump-2 holds A like TAS; remaining dwell is jump-1/stairs |
| Scientist → Flat | 279 | 264 | +15 | Jump removed; leftover is the x=467 east-ledge stall |
| Flat → Ridley | 340 | 321 | +19 | Dwell versus fixed door fade |
| **Total** | **1606** | **1523** | **+83** | Ceres 3 leftover, then Ceres 2 exit / Ceres 4 |

#### 1. Make the comparison clock authoritative — done

`tas/bodies/sniq_100_ceres_lsnes_hops.json` is `sm_tas_ceres_hops_v2` with
both `room_flip` and `settled_gs8` (source frames kept). Default comparator
clock is `settled_gs8`. Selecting `room_flip` against RoomTimer totals sets
`clock_mismatch`. Elev pad is 8639, not ride gs=8 at 8538. Maintained
`scripts/probe/ceres.py` owns the dual comparison and full-station clock proof.

#### 2. Build a frame-aligned TAS/product trace, then extract skills

- Join the already parsed LSMV SNES-12 input stream to the native oracle by
  movie frame. Extend the oracle series/checkpoint schema with x/y subpixels,
  x/y velocity, momentum/speed counter, facing, movement type, game state,
  door transition, knockback/damage state, and RNG where available. Keep raw
  L+R exactly as authored.
- Record the same minimal fields and applied buttons from a product room probe.
  Align traces by semantic landmarks rather than absolute indices: settled
  entry, jump press/release, ledge leave, apex, landing, hit/knockback, door
  leave, room flip, and destination settle.
- Extend extraction beyond the current arm-pump/mockball button detectors.
  Ceres needs candidates for run-off, short spin jump, air turn, controlled
  drop, pre-ledge jump, hit/knockback recovery, and door-speed carry. Each
  candidate must include its input window, entry/exit kinematics, landmark
  frames, room geometry reference, and confidence/evidence source.
- Do not automatically turn a pose cluster into a reusable skill. Promote a
  helper under `routes/skills/` only when the trace identifies a parameterized
  mechanic with a second consumer; otherwise keep the policy in
  `routes/kpdr/ceres/`.

#### 3. Tune Ceres 3 from the real Ceres 2 leave — steam tank gone, −8f

- Pin `scratch/post_ceres_falling_magnet.state` `(39,139) p9 mom=2`: **377f**
  ×2 to Scientist `(39,139) p9` from first-control (was 382f). Fade 162f
  matches TAS 162f; remaining **+32f is dwell**. Leave pin
  `scratch/post_ceres_magnet_scientist.state`.
- Wired knobs: jump-2 window later (`y≥255 x≤135`, TAS ~124,262); 1f LEFT
  then 15f A then DOWN+A (TAS jump-2); 1f `LEFT+B+X` at x~177 then RIGHT.
  Magnet-room pose 137 is gone. Jump-1 idle-coast was tried and rejected
  (pose-41 139-ledge plant, 444f).
- Still open on this room: +32f dwell. Do not reintroduce L/R on the stairs.
  Jump-1 TAS idle-coast needs a different gate than "no buttons at y=139".

#### 4. Tune Ceres 4 from the real Ceres 3 leave — jump gone, −22f

- Pin `scratch/post_ceres_magnet_scientist.state` `(39,139) p9`: **279f** ×2
  to Flat `(39,139) p17`. Fade 161f matches TAS 161f; remaining **+15f is
  dwell** (118 vs 103). Inbound from `ceres_first_control.state` is
  **1627f** ×2; elev/falling/magnet/flat unchanged. Leave pin
  `scratch/post_ceres_scientist_flat.state`. Flat still joins (341f).
- Wired knob: never jump. TAS dwell (f9577–9680) never presses A; RIGHT+B+L/R
  walks the entry lip, the y=187 pit, and the right stairs. The old floor
  takeoff (x 350–410 spin jump) caught x=467 pose 137 and cost ~20f. A on the
  alcove still bonks. `RIGHT+B` on this pin does **not** catch the entry lip
  (TAS holds B from gs=8).
- Still open on this room: east-ledge stall at x=467 (pose 207, ~14f, no KB
  under L/R pump). TAS runs 457→492 in 7f. Do not reintroduce the pit jump.
  Then inspect Flat→Ridley's +20f; do not fold Flat tuning into Scientist.

#### 5. Reconcile main-line ownership after alignment — done

- Deleted the policy toggle, old outbound room tape, obsolete fixture, and
  open-loop reverse escape body. `CERES_SPINE` has one reactive Ceres policy.
- Promoted continuous Morph at **24,187f** ×2 exact with zero state loads,
  progression writes, or capacity writes (`recordings/morph.json` and
  `morph_dual.json`). Previous published Morph was 24,475f.
- Full first-control → elevator exit is **5,520f / 01:32.00** ×2 exact;
  displayed countdown is **00:38.26**. Landing final settle is **8,207f /
  02:16.78**. Evidence is in the maintained Ceres timing report.

### Reverse Ceres TAS alignment — 2026-08-31

**Scope:** reverse Ceres 5 (Flat) and reverse Ceres 4 (Scientist) from the
product Ridley-leave pin. Ridley fight not tuned. Gravity, boot-style,
`DEFAULT_CONTINUOUS_TIP`, `recordings/morph.json`, reverse Magnet Stairs
policy, and STATUS are out of scope.

Sniq 100% lsnes never presses A in reverse Flat (gs=8 f11939–12036) or
reverse Scientist (f12198–12299): LEFT+B+L/R, same never-jump as inbound
Ceres 4. Product stuck-jump on those corridors is leftover versus TAS.

| Hop (`gs=8` → `gs=8`) | before | after | TAS | Δ vs TAS |
|---------------------------|-------:|------:|----:|---------:|
| Flat → Scientist | 293 | **292** | 259 | +33 |
| Scientist → Magnet | 284 | **281** | 263 | +18 |
| Magnet → Falling | 599 | **407** | 331 | +76 |

Seconds @ 60.0988. Dual-exact. Fade already matched (162 vs 162 both rooms).
Did not STATUS.

Pin in: product tail-tank leave → Flat gs=8
`scratch/post_ceres_ridley_flat.state` `(472,119) p82` airborne (TAS
grounded `(472,139) p18`). Reverse Flat leave
`scratch/post_ceres_flat_scientist.state` `(472,139) p18`. Reverse
Scientist leave `scratch/post_ceres_scientist_magnet.state` `(216,395)
p16`. Reverse Magnet leave `scratch/post_ceres_magnet_falling.state`
`(472,139) p16 mom=2`; dedicated dual is **407f** ×2. The successor still
composes through Falling/elevator to Landing (252f to elev, 919f to Ceres
leave, 3110f to Landing from the Falling pin). Inbound Ceres 4 still **279f**.

Flat/Scientist knobs: never jump. TAS dwell never presses A. Classic L↔R pump
stays on (dropping L/R at mom=0 lost 1–3f and inbound Ceres 4 went
279→281). Ridley exit stays product LEFT+A 24f.

Rejected: TAS Ridley-exit copy (skipped); dropping B/L/R on the door lip
(slower).

Reverse Magnet fix: the old policy dropped the Scientist-door run to plain
LEFT (88f to x=43), coasted into the west wall for a knockback before jumping,
then repeatedly pressed A at the west exit because enemy0 was nearby. It now
arm-pumps to x≈55, takes off immediately, uses three landing-gated hops, and
runs through the open door. This removes 192f without changing the predecessor
duals; `scratch/ceres_magnet_to_falling_dual.json` is the practice evidence.

Still open: reverse Flat leftover is the airborne Ridley-exit entry
(+33f dwell 130 vs 97). Reverse Scientist leftover is the x=45 west-ledge
stall (pose 208, ~14f, inbound x=467 analog). TAS runs the west lip without
that freeze. Reverse Magnet remains +76f: the product kinetic state cannot
replay the TAS lower-slope jump window, so the reactive climb is still longer.

---

## Finish Spazer K2.2 (mainline — always collect)

**Product path:** `play_below_spazer_to_west` → `play_spazer_detour` always when
Spazer missing (floor entry included). **No Charge-only West skip.** Continuous
warehouse without Spazer bit is RED until residual pure is green — intentional.

**Done (do not re-prove):**

| Fact | Evidence |
|------|----------|
| Charge on continuous K1 | `play_big_pink_to_ghz`; `below_spazer_with_charge.json` **84,880f** |
| Spazer door / collect / return pure | `below-spazer-to-spazer`, `spazer-collect`, `spazer-return-to-below` |
| Mid band → West pure | `spazer-top-to-west` from y≥220 |
| Mainline wired | `play_below_spazer_to_west` always → `play_spazer_detour` |
| Historical Charge-only West | `warehouse_with_charge.*` **85,992f** beams `0x1000` — **not** product |

**Done this epic:**

1. Climb / top→West / morph-tunnel Super door / pure detour — **GREEN**.
2. **SM-SPAZER-CONT** — continuous `--to warehouse` **GREEN** **89,416f**,
   beams **`0x1004`**, integrity 0 loads/prog/deaths. Floor Cacatac clear
   (Charge-cadence UP+X) before spin — spike knockoff was continuous fail.
   Video: `recordings/warehouse_with_spazer.mp4` (from supers frame).
3. **SM-SPAZER-CONT dual** (`rr-jx9`) — second warehouse integrity match
   **90,904f**, beams `0x1004`, room `0xA6A1`, outcome `warehouse_entry`
   (`warehouse_with_spazer_dual.json`). Frame +1,488 = Spore combat variance.
4. **SM-SPAZER-STATUS** (`rr-4wg`) — STATUS/MILESTONES warehouse Spazer dual
   promoted (2026-08-06). Later folded into Speed dual STATUS (`rr-cd0`).
5. **Speed dual + STATUS** (`rr-d20` / `rr-cd0`) — continuous `--to speed`
   **130,388f** ×2 exact match, beams `0x1004`, items `0x3105`, room
   `0xADDE`; `DEFAULT_CONTINUOUS_TIP = wave` (Speed prefix still 130,388f).

**Optional / later:** dual Spazer `bat_cave` tip STATUS alone (single
**127,806f** in `bat_cave_spazer_cwu.json`) — superseded by Speed dual tip.

**Sources:** [tasks/EARLY_SPAZER_HUMAN.md](tasks/EARLY_SPAZER_HUMAN.md) ·
[tasks/SM-SPAZER-HUMAN-CHUNKS.md](tasks/SM-SPAZER-HUMAN-CHUNKS.md).

---

## Open work by epic

### K4 remaining (Norfair items)

- [x] Pure **Bat → Speed Hall** (residual GREEN)
- [x] Pure Speed Hall → Speed Booster room + collect (residual GREEN)
- [x] Spine graph edges continuous for Speed tip (`--to speed` wired)
- [x] Continuous compose + dual re-verify for `speed` (`rr-d20`, 130388f ×2)
- [x] STATUS promote `speed` (`rr-cd0`, default CLI tip)
- [ ] Stabilize wave after Speed continuous (`rr-07b`)
- [ ] Pure Speed return → Bubble (`rr-g4i`) → Wave → Ice chain
- [ ] Continuous compose + dual for `wave` / `ice` tips
- [ ] (Parked) Speedway → Farm → Bubble post-Speed shortcut

### K5 — Alpha PB

- [x] Natural Alpha PB `0xA3AE` collect after Ice (not Pink PB)
- [x] Scratch dual `--to moat` **175526f** ×2 (rr-2r06; default CLI still `ice`)
- [x] Scratch dual `--to ws` **176141f** ×2 (rr-p2bw; default CLI still `ice`)
- [ ] Planner STATUS promote `--to moat` / `--to alpha_pb` / `--to ws`

### K6 — Ship / Phantoon / Gravity

- [x] Moat shinespark pure from `post_kihunter_pre_moat_spark` → West Ocean
  (store→spin→UP unspin→spark + RIGHT+X door; probe hop + controller pure GREEN;
  West handoff `scratch/post_moat_west_ocean_spark.state`; harness B=dash A=jump;
  residual purged after pure green; **not** continuous / STATUS)
- [x] Landing Site shine practice gym + diagnose/drill
  — [tasks/SHINE_PRACTICE.md](tasks/SHINE_PRACTICE.md)
  (`routes/skills/shinespark.py` + `kpdr.py`; store trap documented)
- [x] West Ocean edge-turn-hop pure → mid-right door `0xC98E` (Bowling)
  — [tasks/SHINE_PRACTICE.md](tasks/SHINE_PRACTICE.md) / `kpdr.py pure west-ocean-to-bowling`
  (practice only; free-place spit bootstrap)
- [x] West Ocean over-ocean spark → green Super WS `0xCA08` pure
  — `kpdr.py pure west-ocean-to-ws` / `play_west_ocean_over_ocean_spark`
  (natural Moat handoff ~(49,1163); stutter dual-green from the power-on
  `--to moat` leave **627f** ×2 probe / **615f** ×2 spine hop; pin
  `scratch/post_moat_poweron_wo_to_ws.state`; `--to ws` scratch dual
  **176141f** ×2, **not** STATUS)
- [x] Product WS pin + human record setup (`--from ws-entrance` /
  `practice_takes --segment ws-entrance`) for ship free-record
- [x] Scratch dual WS Main Shaft → basement **1208f** ×2 `0xCC6F` (rr-4btp;
  `play_ws_main_to_basement`; pin `scratch/post_ws_main_to_basement.state`;
  **not** STATUS; `--to ws` still ends `0xCA08`)
- [x] Scratch dual WS Basement → Phantoon *room* **718f** ×2 `0xCD13` (rr-cjpp;
  `play_ws_basement_to_phantoon`; pin `scratch/post_ws_basement_to_phantoon.state`;
  **not** STATUS; `--to ws` still ends `0xCA08`; fight not started)
- [x] Human Gravity path + tail pin Caterpillar `0xA322` items `0x3125`
  (`scratch/post_gravity_caterpillar.state`; `--from post-gravity`)
- [ ] Natural climb onto West Ocean dry spit (only if reusing edge-bowling path)
- [x] Moat → West Ocean → Wrecked Ship pure compose (`play_moat_to_ws` /
  `kpdr.py compose moat-to-ws`;
  dual pin sources; **pin-only** — not power-on continuous STATUS)
- [x] Compose wired to Phantoon ship recording (`--from ws-entrance` after
  `kpdr.py compose moat-to-ws`; Phantoon hop ← `ws_ship_human_end` ← Gravity free-record)
- [x] `--to phantoon` wired (rr-gyla: Entrance→Main→Basement→room→fight
  wrapper; unit-green; `--to ws` still ends `0xCA08`; not STATUS)
- [x] Wiki KPDR Phantoon pin-benches (rr-7lc5). Doppler wired (rr-asyg)
  **12118f** ×2 + loot/exit **337f** → basement `(1240,139)` p10; compose
  **12455f** ×2. Charge-only 20537f / charge+missiles / Ice-on X-Factor
  stay research.
- [x] Power-on `--to phantoon` dual **195,336f** ×2 (rr-8g2u) + STATUS
  living tip (`rr-b926`). Ice is prefix CI. Tip ends `0xCC6F`.
- [ ] Natural Phantoon leave / WS power-on → Gravity (`rr-kw8t`)

- [x] Grapple side-trek + Maridia free-record from post-gravity pin
  (`tasks/maridia_grapple_human.json` 44039f → Main Street trace end;
  Grapple ~f24720 items `0x7125`; hops extract offline;
  Main Street **binary end pin LOST** — re-lock with anchors from
  `--from post-grapple`; see `docs/tasks/SM-MARIDIA-GRAPPLE-HUMAN.md`)
- [x] Anti-desync human recording: live room/item anchors + F6 + end fingerprint
  (`human_tape.py` / `guided_human` default ON / `extract_human_tape.py`)
- [x] Re-lock Main Street pin from post-grapple with anchors + F6
  (`tasks/maridia_main_street_human` **14170f**, end `0xCFC9` ~(391,1979)
  items `0x7125`, pin `scratch/post_grapple_main_street.state`, end_fp OK;
  `--from main-street`)
- [x] Fix room_enter anchor swallow during door_transition (`human_tape.py`)
- [ ] Continuous tips only after natural doorway entry

### K7 — Maridia

- [x] Human Main Street → Botwoon → Draygon → Space Jump free-record
  (`tasks/maridia_botwoon_path_human` **58670f**, SJ @ f52049 items `0x7325`,
  pins `post_space_jump` / `post_draygon_precious`; shape only — sloppy grapple;
  `docs/tasks/SM-MARIDIA-BOTWOON-HUMAN.md`)
- [ ] Tube / Everest / Botwoon / Draygon pure + continuous (natural entry each)
  (human start: `--from main-street`; post-SJ: `--from post-space-jump`;
  post-Plasma: `--from plasma-beam` / `scratch/full_start_v1_plasma.state`)

### K8 — Lower Norfair / Ridley

- [x] Human post-SJ → Spring + Plasma → LN Main Hall free-record
  (`tasks/post_sj_exit_human` **80368f**, end `0xB236` ~(1152,648) items
  `0x7327` beams `0x100F`; pin `post_ln_main_hall` / `--from main-hall`;
  `docs/tasks/SM-POST-SJ-EXIT-HUMAN.md`)
- [x] Human Main Hall → Screw → Ridley → Landing Site free-record
  (`tasks/post-main-hall` **121220f**, Screw f10857 items `0x732F`, Ridley
  Norfair bit 6→7, end `0x91F8` ~(1152,1088); pins `post_bosses_landing_site`
  / `post_screw_attack` / `post_ridley_tank`; `--from post-bosses`;
  `docs/tasks/SM-POST-MAIN-HALL-HUMAN.md`)
- [ ] LN pure geometry + Ridley combat natural-entry (shape from human tape)

### K9 — Tourian / MB / Escape / Credits

- [ ] Human G4 statues → Tourian → Mother Brain free-record
  (start: `--from post-bosses`)
- [ ] G4 statues → Tourian → Mother Brain pure + continuous (zebetites, phases)
- [ ] Escape timer + geometry + ship / ending-credits evidence (M8)

Boss order and phase rules: [BOSS_PIPELINE.md](BOSS_PIPELINE.md). Template:
Kraid → Varia continuous.

### PRACTICE (dual-track, planner opt-in)

- [ ] `ROOM_WORK_QUEUE` + `farm_room_waves.sh` when planner opts in (not default P0)
- [ ] Combat unit scaffolds (no full fights before natural entry)
- [ ] Early Spazer walljump detour + 100% board (parallel; does not block K4)
  — [routes/TRACK_100.md](routes/TRACK_100.md)

Practice greens ≠ continuous evidence and **not** product next-work
(`STATUS` + beads own that). Own-files only; width ≤ 8.

### CLEAN (parallel)

- [x] Morph Clean continuous green (26,824f prefix); assisted tip now 24,187f
- [ ] ★ Bombs / Torizo Clean (`SM-CLEAN-BOMBS`)
- [ ] Later Clean tips only after bombs green
- Never mutate default CLI assists or demote assisted baselines

### ARCH (planner-serial on hot modules)

Highest leverage for whole-game length — see Structure below and
[ARCHITECTURE.md](ARCHITECTURE.md).

### Maturity targets

| Gate | Target | Notes |
|------|--------|-------|
| **M5** | Bronze observation; Survival continuous tip | **Current** (Phantoon). Sticker, not a rung. |
| **M6** | Complete route graph with owners/predicates | In progress |
| **M7** | Continuous dry-run invariants (power-on → credits path) | Open |
| **M8** | Verified capture + ending/credits evidence | Open |

Observation-class migration (Bronze → Silver) is a separate workstream after
continuous reliability.

---

## Structure & API (open only)

Current layers (CLI → continuous + catalog + segment → pure kpdr → graph →
ram/assist → combat) are correct. Planner-serial when touching
`continuous.py` / `progression.py` / `catalog.py`.

### Selective RAM + StateCache

- [ ] Profile frame time on long tips / full runs (WRAM-copy rate)
- [ ] Optional linter: forbid bare full `parse_env_state` inside `routes/kpdr/`

### Declarative continuous composition

- [x] Move hop tables out of continuous (`routes/kpdr/hops.py` + tip-spec bind)
- [x] Early morph→supers extracted to `routes/early_continuous.py`; shared
  prefix conditions + room-timing helpers in `runtime`
- [x] Morph/Ceres SpineHop orchestration (`routes/kpdr/early_spine.py`); seeds unchanged
- [x] Controller shims deleted (`kpdr_controller` / `post_spore_controller`)
- [x] Fold bombs/spore/supers play onto SpineHop (`early_post_morph.py`; no timing risk)
- [ ] Optional: further collapse early run_* finish_report boilerplate

### Source-state & pure-probe diagnostics

- [ ] Short video clip + PLM/door RAM snapshot on pure RED
- [ ] Dispatch auto-suggest `--source` from card schema
- [ ] Provenance on checkpoints (parent tip, command, capabilities)

### Controller structure

- [x] Bubble skills extraction (`routes/skills/` + hop policy `bubble_to_bat`; product `to_bat_cave`)
- [x] K4 knockback / Super-door plant frames → `routes/skills/knockback.py` + `door.super_door_pressure_frame`
- [ ] Prefer `wait_ordinary_room` handoff bands (`y_range` etc.) over airborne
  settle hope
- [x] Remove thin helper aliases (`_hold = hold`) from KPDR segment modules
- [x] Rename room-named KPDR segments to hop names (`pink_to_ghz`, `red_stack`, `to_kraid`, …)
- [x] De-nest progression stage tables (`progression/stages/`)
- [x] Continuous `*RunReport` aliases removed; Super+ tips via `play_tip` / `run_to` only

### Graph first-class

- [ ] Typed path-summary model (not ad-hoc `dict[str, object]`)
- [ ] Stop mechanical multi-line DoorEdge reformats; extract edge data if needed
- [ ] Work-queue / tracker export reads graph verification + dwell ranks
- [ ] Planner “next pure” CLI: path summary + SOURCE suggest together

### Hygiene

- [ ] Keep `legacy/` and `dev/` (door-warps) strictly fenced
- [ ] Normalize artifact naming (semantic states)
- [ ] Promote shared adventure patterns to `retro_harness.adventure` **only after**
  SM + ALTTP both prove the abstraction

### Agent process improvements

- [ ] Stronger pre-dispatch schema validation + auto-skeleton residual.md
- [ ] Mandatory residual metrics: frames, dwell, exact pose/x/y/`door_transition`
- [ ] Ownership / file-locking so parallel waves stay safe
- [ ] Dual-track room farming remains planner opt-in (never starves serial pure)

**Do not** relax pure-first / one-knob / residual rules.

---

## Risks

| Risk | Mitigation |
|------|------------|
| Long-horizon nav fragility / high-dwell segments | Tighten offline secondary; stabilize after each tip |
| Architecture debt (multi-registry tip wire, full WRAM hot paths) | Highest-leverage ARCH first |
| Residual / card proliferation | Archive-after-successor + one-knob schema |
| Process drift (practice claiming continuous) | Dual-track gates; planner owns STATUS |
| Endgame (Zebetites regen, escape geometry, timer/WRAM) | Deferred until natural entry |

## Non-goals (now)

- Door-warp / hybrid tours as continuous evidence (Track A topology **done** —
  stop expanding warp product work)
- Ship-first / PRKD continuous route
- Pink PB maze (first PB is Alpha after Ice)
- Vision-based boss combat in `legacy/`
- Claiming Clean greens as the program M5/M8 assisted gate
- Full endgame fight code before natural entry

---

## Recommended next waves

```text
★ NOW   Gravity on the Phantoon tip (rr-kw8t). Power-on is the rung.
THEN    Maridia (tube / Botwoon / Draygon / SJ) on the tip
        LN + Ridley → Tourian + MB + escape + credits
Parallel Chip prefix slop under Sync (re-pin; couple rooms if the seam fails)
Publish every 20–30 working sessions to the living tip (not a calendar)
Later   rewrite toward sub-hour; drop convenience majors; 100%
```

Live dispatch: `bd ready -l super_metroid`.
Milestone names: [routes/MILESTONES.md](routes/MILESTONES.md).

---

## Pointers

| Doc | Role |
|-----|------|
| [`CONTEXT.md`](../CONTEXT.md) | Language |
| [STATUS.md](STATUS.md) | Verified tip + prefix frames |
| `bd ready -l super_metroid` | Ready / in-flight work |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Layers + structural debt |
| [BOSS_PIPELINE.md](BOSS_PIPELINE.md) | Boss natural-entry rules |
| [CLEAN_TRACK.md](CLEAN_TRACK.md) | Clean track process |
| [routes/ROUTE_KPDR.md](routes/ROUTE_KPDR.md) | KPDR route text |
