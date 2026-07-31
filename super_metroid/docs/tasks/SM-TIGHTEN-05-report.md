# SM-TIGHTEN-05 Report — Spore Spawn fight split dwell (12,182f + 5,170f + 1,499f + 1,540f)

**Source:** `recordings/start_to_varia.json` (outcome=`varia_collected`, total=101,954f)
**Controller:** `routes/spore_spawn_controller.py` → `play_main_shaft_to_spore_spawn`
**Window:** `green_brinstar_main_shaft` @52,778 → `spore_supers_collected` @73,169 = 20,391f

---

## Split structure

| Split | Room | Frame | Dwell | Role |
|-------|------|-------|-------|------|
| `green_brinstar_main_shaft` | 0x9AD9 | 52,778 | — | Approach start (elevator settle) |
| `spore_spawn_activated` | 0x9DC7 | 64,960 | 12,182f | Pre-fight: main shaft → Dachora → Big Pink → Kihunter → entry |
| `spore_spawn_defeated` | 0x9DC7 | 70,130 | 5,170f | Fight: activation → HP zero |
| `spore_spawn_exit` | 0x9B5B | 71,629 | 1,499f | Post-fight: death settle → exit climb → exit room |
| `spore_supers_collected` | 0x9B5B | 73,169 | 1,540f | Supers item collection |

---

## Phase map (pre-fight: 12,182f)

```
Window [52,778 → 64,960]  (12,182f)
│
├─ main_shaft_entry_settle (1,000f)  ── fixed hold after elevator lands
├─ main_shaft_descent (240f)  ── 4×60f RIGHT/LEFT bounces
├─ main_shaft_descent_settle (50f)
├─ main_shaft_dachora_level (400f)  ── 5×80f RIGHT/LEFT bounces
├─ main_shaft_dachora_door_settle (30f)
├─ select_missiles (11f)
├─ open_dachora_red_door (255f)  ── 15×(2f X + 15f wait)
├─ enter_dachora (100f)  +  dachora_entry_settle (250f)
├─ cross_dachora (350f)
├─ dachora_tunnel_morph / bomb (15×(45+15) = 900f)
├─ exit_dachora (160f)
├─ big_pink_entry_settle (300f)
├─ unmorph (12f) + big_pink_climb (260f) + big_pink_map_guided_climb (1,056f)
├─ big_pink_red_door_approach (100f)
├─ open_kihunter_red_door (255f)  ── 15×(2f X + 15f wait)
├─ enter_kihunter (150f) + kihunter_entry_settle (300f)
├─ clear_spore_kihunters (1,440f)  ── 8×180f fixed sweeps
├─ aim_at_spore_kihunters (2,400f)  ── 240×10f fixed aim loop
├─ kihunter_clear_settle (300f)
├─ kihunter_boss_door_runway/jump (190f)
├─ door alignment (180f)  ── align/center/release
├─ open_spore_spawn_door (180f)  ── 15×(2f X + 10f)
├─ enter_spore_spawn (120f) + spore_spawn_entry_settle (300f)
└─ room check + enemy0 HP verify (~5f)
```

## Phase map (fight: 5,170f)

```
Window [64,960 → 70,130]  (5,170f)
│
└─ fight_spore_spawn (5,170f)  ── floor-bounce loop (spore_spawn_controller.py:370-406)
    ├─ Floor bounce: jump when y≥710, hold 44-52f, direction LR
    ├─ Fire: mouth_open && index%2==0 → UP+X
    └─ Airborne: aim UP toward core, face direction, hold jump
```

## Phase map (post-fight: 1,499f)

```
Window [70,130 → 71,629]  (1,499f)
│
├─ spore_spawn_death_settle (600f)  ── fixed hold after HP zero
├─ spore_exit_map_guided_climb (816f)  ── 51-step × 16f fixed sequence
├─ open_spore_exit_door (200f)  ── 20×(2f X + 8f wait)
└─ spore_spawn_exit_settle (300f)
```

## Controller ownership

| Sub-segment | Function | Lines | Action reason(s) |
|-------------|----------|-------|------------------|
| Main shaft settle | `play_main_shaft_to_spore_spawn` | 254-257 | `main_shaft_entry_settle` |
| Descent + Dachora | `play_main_shaft_to_spore_spawn` | 258-282 | `main_shaft_descent`, `open_dachora_red_door`, etc. |
| Dachora room | `play_main_shaft_to_spore_spawn` | 284-292 | `cross_dachora`, `bomb_dachora_tunnel`, `exit_dachora` |
| Big Pink climb | `play_main_shaft_to_spore_spawn` | 296-303 | `big_pink_climb`, `big_pink_map_guided_climb` |
| Kihunter room | `play_main_shaft_to_spore_spawn` | 304-340 | `clear_spore_kihunters`, `aim_at_spore_kihunters`, `kihunter_clear_settle` |
| Boss door entry | `play_main_shaft_to_spore_spawn` | 341-356 | `kihunter_boss_door_*`, `center_under_spore_spawn_door`, `open_spore_spawn_door` |
| Fight loop | `play_main_shaft_to_spore_spawn` | 370-406 | `fight_spore_spawn` |
| Death settle | `play_main_shaft_to_spore_spawn` | 414 | `spore_spawn_death_settle` |
| Exit climb | `play_main_shaft_to_spore_spawn` | 415-418 | `spore_exit_map_guided_climb` |
| Exit door | `play_main_shaft_to_spore_spawn` | 419-422 | `open_spore_exit_door`, `spore_spawn_exit_settle` |

---

## Top suspected waste

### 1. Fixed kihunter clear loop (3,840f)

The Kihunter room uses two fixed-length loops:
- `clear_spore_kihunters`: 8 × 180f = 1,440f ── fixed RIGHT/LEFT sweeps regardless of when kihunters die
- `aim_at_spore_kihunters`: 240 × 10f = 2,400f ── fixed 240-iteration fine-aim loop

These are designed to reliably kill 3 kihunters with the missile-aim pattern. But if they die faster, the remaining frames are idle. The actual `enemies_killed >= 3` check at line 339 happens *after* both loops complete.

**`spore_spawn_controller.py:312-322`**: `clear_spore_kihunters` — 8×180f fixed RIGHT/LEFT
**`spore_spawn_controller.py:323-337`**: `aim_at_spore_kihunters` — 240×10f fixed aim

**Estimated waste:** 500-1,500f (variable depending on how fast kihunters die). If the aim loop terminates early when `enemies_killed >= 3`, the 2,400f fixed loop could be cut to 1,000-1,900f. The clear sweeps (1,440f) could also exit early if enemies are already dead.

**Risk:** Medium. The kihunters are on different horizontal planes and the missile pattern needs to cover all positions. Early termination could miss the last kihunter if it spawned late or is off-screen.

### 2. Fixed 1,000f main shaft settle (1,000f)

The `main_shaft_entry_settle` at line 257 is a hard 1,000f hold. The comment says it was relaxed from 360f to 1,000f for continuous reliability (the guarded settle with x/y/pose was too tight for elevator land variance).

**`spore_spawn_controller.py:257`**: `_hold(session, 1_000, reason="main_shaft_entry_settle")`

**Estimated waste:** 400-640f (the elevator lands and settles well before 1,000f). The settle could be replaced with `wait_ordinary_room` + standing check, or reduced to 500-600f with a predictive settle.

**Risk:** Low-medium. The original 360f settle timed out at x=128 y=680 pose=0 — too tight for elevator variance. A wider guard (`x in [100, 140]` and `y == 680` or `standing`) could replace the fixed hold. But the 1,000f was chosen deliberately for reliability.

### 3. Fixed bomb tunnel loops (900f + 900f)

The Dachora bomb tunnel (`dachora_tunnel_morph` + `bomb_dachora_tunnel`: 15 × 60f = 900f) and the Big Pink bomb tunnel (similar pattern at `big_pink_tunnel_bomb_wait`: 700f + `big_pink_tunnel_roll`: 251f) use fixed-iteration bomb loops. These could be replaced with reactive patterns that exit as soon as the tunnel is cleared.

**`spore_spawn_controller.py:285-291`**: Dachora tunnel — 15×(45f bomb + 15f roll)
**`spore_spawn_controller.py:296-301`** (Big Pink section): similar bomb tunnel

**Estimated waste:** 100-300f (combined). The tunnels usually clear in 10-12 bomb cycles, not 15.

**Risk:** Low. The bomb tunnel exit condition (`room_id == next_room` or `samus_x > threshold`) is easy to check per-iteration.

### 4. Fixed 600f death settle (600f)

The `spore_spawn_death_settle` at line 414 is a hard 600f hold after the boss HP reaches zero. This covers the death animation, fanfare, and item drop.

**`spore_spawn_controller.py:414`**: `_hold(session, 600, reason="spore_spawn_death_settle")`

**Estimated waste:** 100-250f (the death animation + fanfare is ~350-500f, not 600f). But this is fragile — the death settle must be long enough for the item PLM to spawn.

**Risk:** Low. The death settle could be replaced with a `wait_until` for the item PLM bit or the door to appear. But the 600f is a deliberate safety margin.

### 5. Fixed 300f entry/exit settles (1,200f cumulative)

Multiple 300f fixed settles: `big_pink_entry_settle` (300f), `kihunter_entry_settle` (300f), `kihunter_clear_settle` (300f), `spore_spawn_entry_settle` (300f), `spore_spawn_exit_settle` (300f) = 1,500f total.

**`spore_spawn_controller.py`**: lines 293, 309, 338, 353, 422

**Estimated waste:** 100-300f cumulative (each settle could be 200-250f instead of 300f, or replaced with `wait_ordinary_room`).

**Risk:** Low. Room settles are naturally ~100-200f; the 300f is a safety margin. Reducing to 200-250f is safe.

---

## Patch recipes (future cards)

### P1: Reactive kihunter clear — exit aim loop when enemies_killed >= 3

- **Files:** `routes/spore_spawn_controller.py` lines 323-337
- **Change:** Add an `enemies_killed` check inside the aim loop (lines 323-337). Break out of the loop when `session.state.enemies_killed >= 3`, then skip the remaining `clear_spore_kihunters` iterations if already dead. Or replace the fixed 240-iteration loop with a `for` loop that checks `enemies_killed` each iteration and breaks early.
- **Risk:** Medium. The kihunters spawn at different positions; the 240-iteration aim pattern is designed to cover all angles. Early termination could leave a stray kihunter alive. The `enemies_killed` counter must be handled carefully (it resets per room but may include the kihunters killed in the sweeps).
- **Expected band:** 500-1,500f speculative.
- **Acceptance:** `uv run python super_metroid/scripts/record/continuous.py --to spore --no-video` + verify `enemies_killed >= 3` in report.

### P2: Reduce main shaft entry settle from 1,000f to 600f with wider guard

- **Files:** `routes/spore_spawn_controller.py` line 257
- **Change:** Replace `_hold(session, 1_000, ...)` with a `wait_until` loop that polls `x in [100, 140]` and `standing` (or `velocity_y == 0`), with a 600f timeout. The original 360f timeout was too tight; the 1,000f is too generous. A 600f timeout with a wider x band (100-140 instead of 118-126) should be reliable.
- **Risk:** Low-medium. The elevator land position has inherent variance. The wider band (100-140) should cover all natural land positions. Verify with 3+ `--to spore` runs.
- **Expected band:** 400f speculative.
- **Acceptance:** `uv run python super_metroid/scripts/record/continuous.py --to spore --no-video` green.

### P3: Reduce fixed 600f death settle to 400f + item PLM wait

- **Files:** `routes/spore_spawn_controller.py` line 414
- **Change:** Replace `_hold(session, 600, ...)` with a wait for the Spore Spawn item PLM to appear (or the boss bit to be set). Use a 400f minimum settle + optional early exit when the door reappears. The death animation is ~350f; the fanfare is ~100f. Total ~450f would be safe.
- **Risk:** Low. The death settle is a safety margin. The 600f was chosen to cover the worst-case death animation. Reducing to 500f or adding an early-exit condition (item PLM bit check) would save 100-200f.
- **Expected band:** 100-200f speculative.
- **Acceptance:** `uv run python super_metroid/scripts/record/continuous.py --to spore --no-video` green. Verify `spore_supers_collected` is still achieved.

### P4: Batch-reduce fixed 300f entry/exit settles to 200f

- **Files:** `routes/spore_spawn_controller.py` lines 293, 309, 338, 353, 422
- **Change:** Reduce each 300f settle to 200f, or replace with `wait_ordinary_room` with a 200f timeout. The room transition settles naturally in 100-200f; 300f is a generous safety margin.
- **Risk:** Low. Each settle is a room transition settle. The `wait_ordinary_room` function (controller_common.py) already handles game state 8. Reducing to 200f saves 500f across 5 settles.
- **Expected band:** 500f speculative (5×100f).
- **Acceptance:** As above.

### Combined P1+P2+P3+P4

If all four patches land, the combined expected band is ~1,500-2,700f. Re-record `--to spore` and `--to varia` to verify integrity and measure the actual delta.

---

## Caveats

- **No frame savings claimed without re-record.** The "expected band" is an estimate from counting action_reason frames, not a measurement.
- **The pre-fight window (12,182f) includes the elevator ride, room transitions, and fixed settles.** The actual controllable navigation is ~7,000-8,000f of the 12,182f.
- **The fight (5,170f) is owned by the boss pipeline.** The floor-bounce + missile pattern is already efficient for a natural-entry fight. The 5,170f is ~86s at 60fps, which is reasonable for a 960 HP boss with missile-only damage.
- **Continuous-hardening was intentional:** the 1,000f settle, 300f room settles, and 600f death settle were added to fix prior fail-to-proceed blockers. Any tightening must maintain the natural-entry reliability.
- **The `start_to_kraid.json` file was overwritten during analysis** (shows `failed:TimeoutError` at 54,004f). All data in this report is from `start_to_varia.json` which completed successfully.

## Acceptance command for a future implement card

```bash
# Full tip re-record (must be integrity green, 0 state loads, 0 progression writes)
uv run python super_metroid/scripts/record/continuous.py --to spore --no-video

# Then verify dwell reduction
uv run python super_metroid/scripts/export/split_dwell.py \
  super_metroid/recordings/start_to_spore.json --top 20

# Also verify the full chain is unaffected
uv run python super_metroid/scripts/record/continuous.py --to varia --no-video
```

Or for a pure-probe first pass (tests only the approach + fight, not the full chain):

```bash
uv run python super_metroid/scripts/probe/kpdr.py pure ghz-to-noob \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/dev_kpdr_ghz.state
```

## Non-claims

- Did not STATUS-promote.
- Did not forge progression RAM.
- Did not edit any controller files.
- No frame savings claimed without re-record.

## Residual

- `start_to_kraid.json` overwritten/missing during analysis; all data from `start_to_varia.json`.
- Action_reason data is summed across the entire run, not per-split; the pre-fight approach (12,182f) includes some action_reasons that also appear in other windows (e.g., `policy_pit_to_post_torizo` at 13,143f is in the pre-spore window but not in the spore split).
- The Kihunter early-exit recipe (P1) has the highest potential gain (500-1,500f) but also the highest risk of leaving a stray kihunter alive.