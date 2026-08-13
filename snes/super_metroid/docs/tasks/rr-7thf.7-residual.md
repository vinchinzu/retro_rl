## Residual — rr-7thf.7 Wave D pure green (MB approach + Mother Brain fight)

### Result
**PARTIAL → acceptance met**

- **Approach (bb hops 0–7):** dual hop-replay **GREEN** (boot-settle **0**, contract assist default).
- **MB leave → Escape 1 (`g4_tourian_human_mb` hop 0):** dual **GREEN** with contract assist.
- **BB hop 8 (MB room full dwell entry → mid-rainbow end pin):** dual **RED** even with assist — combat open-loop desync; still in `0xDD58` but not at end xy/pose. **No fake mid anchors invented.**

No STATUS promote.

### GREEN/RED table

| Hop | Tape | Room | Leave / goal | Dual | End pin (both runs) | Notes |
|----:|------|------|--------------|:----:|---------------------|-------|
| 0 | bb | Big Boy `0xDCB1` | → Seaweed `0xDCFF` | **GREEN** | room=`0xDCFF` xy=`[20, 118]` pose=82 `ROOM_TRANSITION` | boot anchor f0 |
| 1 | bb | Seaweed `0xDCFF` | → Recharge `0xDD2E` | **GREEN** | `0xDD2E` `[20, 395]` pose=10 | enter f1801 |
| 2 | bb | Recharge `0xDD2E` | → Seaweed `0xDCFF` | **GREEN** | `0xDCFF` `[236, 139]` pose=9 | enter f2224 |
| 3 | bb | Seaweed `0xDCFF` | → Eye Door `0xDDC4` | **GREEN** | `0xDDC4` `[236, 390]` pose=81 | enter f2569 |
| 4 | bb | Eye Door `0xDDC4` | → Rinka `0xDDF3` | **GREEN** | `0xDDF3` `[1004, 139]` pose=9 | enter f3010; 1130f slice |
| 5 | bb | Rinka `0xDDF3` | → Save `0xDE23` | **GREEN** | `0xDE23` `[20, 395]` pose=10 | enter f4260 |
| 6 | bb | Save `0xDE23` | → Rinka `0xDDF3` | **GREEN** | `0xDDF3` `[236, 139]` pose=9 | enter f4649 |
| 7 | bb | Rinka `0xDDF3` | → MB room `0xDD58` | **GREEN** | `0xDD58` `[20, 651]` pose=10 | enter f4872; **MB room entry** |
| 8 | bb | MB room `0xDD58` | end pin (no leave) mid-rainbow stun | **RED** | got `0xDD58` xy=`[299, 195]` pose=1 (want `[205, 195]` pose=233) | 14557f open-loop combat; see phase-split |
| 0 | mb | MB room `0xDD58` | → Escape1 `0xDE4D` | **GREEN** | room=`0xDE4D` xy=`[20, 138]` pose=82 `ROOM_TRANSITION` | boot f0 (= bb end stun); 10805f; **leave-to-escape** |

### Assist (required for combat dual-green)

Human tapes were recorded with `UnlimitedResourcesAssist` (rainbow drain suspend + energy/ammo refill). Hop-replay previously stepped **without** assist → MB hops died and ended in **Tourian Elevator `0xDAAE`** (both dual runs).

**Code fix this card:**

- `human_tape_replay.replay_hop` / `run_hop_replay`: default `assist=True` (record-path parity / ASSIST_CONTRACT).
- CLI: `--no-assist` for clean-track experiments.
- Without assist: mb hop 0 and bb hop 8 **RED** → `0xDAAE`.
- With assist: mb hop 0 **GREEN**; approach hops unchanged **GREEN**; bb hop 8 still **RED** (geometry/pose desync, not death).

### MB phase-split needs (no fake anchors)

Live anchors today:

| Anchor | Tape | Kind | Meaning |
|--------|------|------|---------|
| f005415 enter `0xDD58` | bb | room_enter | Natural MB entry (post Rinka leave) |
| **f008135 mid_lockstep** | bb | mid_lockstep | Lockstep dump enter→mid (xy≈`[322,130]` pose 84); dual enter→mid GREEN |
| f019972 end `0xDD58` pose 233 | bb | end | Mid rainbow/stun handoff (= mb boot) |
| f000000 boot `0xDD58` pose 233 | mb | boot | Same pin; fight resume → escape leave |
| f010924 enter Escape1 | mb | room_enter | Post-leave |

**Phase status:**

| Phase | Trace cue | Gate | Status |
|-------|-----------|------|--------|
| A Entry / early combat | bb hop 8 early dwell | lockstep mid f8135 | **enter→mid dual GREEN** (not full leave) |
| B Brain body mid→stun | combat spray after f8135 | need more F6 / pure | mid→end open-loop **RED** |
| C Rainbow / baby latch | pose 84 / 233 / 235 | stun pin = end/boot | handoff GREEN as mb boot |
| D Escape leave | leave `0xDE4D` | dual hop-replay mb hop 0 | **GREEN** |

Lockstep scan (bb hop 8): `contiguous_last_match=8135`, first mismatch f8136.
Materialize: `scripts/tools/materialize_hop_mid.py …_bb.json --hop 8 --materialize`.

Trace windows (mb hop 0, from task `trace` — informational only):

- pose 233 (stun): f0–236, …, long run ~f3418–6238
- pose 235: f1749–3417 (~1669f)
- pose 84 (rainbow): 20 short runs, total ~448f, last ~f9476–9491
- energy drops during early rainbow (499→79) then assist refill; leave f10805 en=499

**Do not** promote bb hop 8 full dwell as pure skill until phase A–C mid anchors exist **from natural path** and each sub-slice dual-greens. Leave-to-escape skill is already dual-green from the live mid-stun boot.

### Commands (re-verify)

```bash
# Critical: --boot-settle 0 (default). Assist ON by default.

# Approach 0–7
for h in 0 1 2 3 4 5 6 7; do
  uv run python snes/super_metroid/scripts/tools/replay_human_hop.py \
    snes/super_metroid/tasks/g4_tourian_human_bb.json --hop $h --dual
done

# MB leave → Escape 1 (combat)
uv run python snes/super_metroid/scripts/tools/replay_human_hop.py \
  snes/super_metroid/tasks/g4_tourian_human_mb.json --hop 0 --dual

# Full MB room from entry (expected RED until phase splits)
uv run python snes/super_metroid/scripts/tools/replay_human_hop.py \
  snes/super_metroid/tasks/g4_tourian_human_bb.json --hop 8 --dual

# Death path without assist (documents assist necessity)
uv run python snes/super_metroid/scripts/tools/replay_human_hop.py \
  snes/super_metroid/tasks/g4_tourian_human_mb.json --hop 0 --dual --no-assist
```

### Seeds written

Under `snes/super_metroid/tasks/g4_tourian_human_bb_seeds/`:

| File | Mode | Frames after trim |
|------|------|------------------:|
| `big_boy_safe.json` | safe | 1109 |
| `seaweed_a_safe.json` | safe | 403 |
| `recharge_safe.json` | safe | 361 |
| `seaweed_b_safe.json` | safe | 439 |
| `eye_door_safe.json` | safe | 1147 |
| `rinka_a_safe.json` | safe | 364 |
| `save_safe.json` | safe | 86 |
| `rinka_b_safe.json` | safe | 504 |
| `mb_room_safe.json` | safe | 14489 |
| `mb_room_combat.json` | combat | 14489 |

Under `snes/super_metroid/tasks/g4_tourian_human_mb_seeds/`:

| File | Mode | Frames after trim |
|------|------|------------------:|
| `mb_fight_safe.json` | safe | 10619 |
| `mb_fight_combat.json` | combat | 10619 |

**Open-loop:** prefer `*_safe.json` for approach hops (validated dual-green on **raw** hop slices, not re-validated as trimmed seeds). MB fight seeds are combat-long; use only after dual-green on the raw hop (mb hop 0 GREEN). Do not open-loop `mb_room_*` as a skill until phase-split greens exist.

### Files touched (code)

- `snes/super_metroid/human_tape_replay.py` — default contract assist in `replay_hop` / `run_hop_replay`
- `snes/super_metroid/scripts/tools/replay_human_hop.py` — `--no-assist`
- Seeds dirs above
- This residual

Unit: `uv run pytest snes/super_metroid/tests/test_human_tape_replay.py -q` → 10 passed.

### Acceptance

- [x] Approach hops (bb 0–7) dual GREEN
- [x] MB leave-to-escape dual GREEN (mb hop 0) from mid-stun boot
- [x] Full fight entry→end RED documented; phase-split needs listed; **no fake anchors**
- [x] Safe/combat trim seeds for bb approach + mb fight
- [x] Residual with commands + results
- [ ] STATUS promote — **not done** (per card)
- [ ] bb hop 8 dual GREEN — **deferred** (needs natural mid-phase anchors)

### Residual risks

1. Hop-replay greens are **anchor + frame slice**, not continuous G4 → credits.
2. Assist is now default on hop-replay; clean-track duals must pass `--no-assist`.
3. bb hop 8 RED is deterministic combat drift (same both runs), not flaky RNG alone.
4. Trimmed seed frame lists not dual-green re-validated.
5. Wave C (`rr-7thf.6` G4→Big Boy) still separate; approach here starts at Big Boy.

### Next action

- Close `rr-7thf.7` on this residual (acceptance met).
- Optional follow-up bead: re-record / dump **natural** MB mid-phase anchors (zebetite clear, brain body, pre-rainbow) then dual-green phase slices — still no invented pins.
- Promote approach safe seeds + mb leave to pure controllers under `routes/` only after product asks.

### Non-claims

- Did not STATUS-promote MB fight or continuous credits
- Did not invent mid-fight anchors
- Did not package pure controllers under `routes/kpdr`
- Did not dual-green-validate trimmed `*_safe.json` frame lists as standalone skills
- Did not force-push / commit (not requested)

---

## Wave2 residual probe (2026-08-10)

Deepen bb hop 8 mid→stun RED without multi-minute invent. Assist ON, boot-settle **0**.
Leave-to-escape (`g4_tourian_human_mb` hop 0) remains dual GREEN (not re-run this wave).

### Re-verify enter→mid

**Trap:** `--from-frame 5415 --to-frame 8135` desyncs (replays the dump frame).
Use **replay_start = anchor_frame+1 = 5416**.

```bash
uv run python snes/super_metroid/scripts/tools/replay_human_hop.py \
  snes/super_metroid/tasks/g4_tourian_human_bb.json \
  --from-frame 5416 --to-frame 8135 --dual --boot-settle 0
```

**GREEN dual** — both runs `0xDD58` xy=`[322, 130]` pose=84 (mid pin). Anchor
`f005415_enter_0xDD58`. Confirms Wave1 enter→mid.

### Post-mid short windows (from mid_lockstep pin)

Boot `f008135_mid_lockstep_0xDD58.state`, step f8136..end, dual **end-geometry only**
(xy_tol=24 — not frame-lockstep):

| Window | Dual end-band | Notes |
|--------|:-------------:|-------|
| +50 / +100 / +200 / +300f | **GREEN** | soft band; dual-consistent drift near tape |
| +350 / +400f | **RED** | e.g. got `[313,144]` want `[343,144]` |
| +500f | **GREEN** | coincidence — desync re-enters band |
| +600 / +700f | **RED** | |
| +800 / +900f | **GREEN** | coincidence again |
| +1000 / +1600 / +3200 / +5000f | **RED** | diverges harder |

**Contiguous soft end-band from mid ≈ +300f**, then **oscillates** GREEN/RED as
open-loop position wanders near/far from tape end_xy. **Not** a stable pin chain
and **not** lockstep.

**Lockstep from mid:**

| xy_tol | last_match | first_mismatch |
|-------:|-----------:|----------------|
| 12 | None | **f8136** got `[318,125]` p84 want `[329,144]` p40 |
| 24 | **8138** | **f8139** got `[308,116]` want `[329,144]` |

Lockstep-safe extension past mid ≈ **0–3 frames**. Cannot materialize a later
`mid_lockstep` from this recording.

### Offline propose — mid→end only (f8136..19972)

37 candidates (combat_pose heavy). Highest-value re-record / F6 targets:

| i | kind | xy | pose | note |
|--:|------|----|-----:|------|
| 8143 | combat_pose | [329, 139] | 84 | first post-mid (already past lockstep) |
| 8829 | combat_pose | [261, 155] | 138 | knockback-class |
| 10177 | combat_pose | [229, 195] | 138 | floor |
| 18520 | combat_pose | [234, 124] | **233** | first long stun |
| 19525 | combat_pose | [234, 124] | **233** | stun return |
| 19972 | pre_leave | [205, 195] | **233** | end / mb boot handoff |

Stun pins (18520 / 19525 / 19972) are the phase-C handoff class; mb hop 0 already
greens leave-to-escape from the live end/boot pin.

### Safe / traversal trim notes

- **Safe full hop 8:** 14721 → 14489 (trailing_idle=232). Re-exported
  `mb_room_safe.json` / `mb_room_combat.json` (RED full dwell; not a skill).
- **Safe enter→mid:** 2720 → 2709 (leading_idle=11). Seed
  `tasks/g4_tourian_human_bb_seeds/mb_enter_to_mid_safe.json` dual-green prefix.
- **Traversal thrash (edit hints only):** 14721 → 12695; mid_idle=1085 + **2
  retry loops / 709f**. Non-contiguous kept_ranges — do **not** open-loop as
  mid→stun skill.

### Recommendation (bb hop 8 MB fight entry→stun)

| Option | Fit | Why |
|--------|-----|-----|
| Pure rewrite full 14k dwell | Heavy | thrash exists but fight is multi-phase RNG |
| Soft end-band extend / more open-loop mid | **No** | lockstep dies f8136; +300f soft is coincidence |
| More mid_lockstep pins at next **natural** F6 index | **Preferred** | need live pins for phase B (post-zebetite / brain body) then dual each sub-slice |
| Rely on leave-to-escape only | Already done | mb hop 0 GREEN from stun boot |

**Exact next command (after F6 mid-phase dump on natural path):**

```bash
# List offline targets for guided F6 (no invent):
uv run python snes/super_metroid/scripts/tools/materialize_hop_mid.py \
  snes/super_metroid/tasks/g4_tourian_human_bb.json --hop 8 --propose

# After new live mid pins exist in *_anchors.json, dual each phase window, e.g.:
uv run python snes/super_metroid/scripts/tools/replay_human_hop.py \
  snes/super_metroid/tasks/g4_tourian_human_bb.json \
  --from-frame 5416 --to-frame 8135 --dual --boot-settle 0   # phase A (known GREEN)
# then from each new mid pin → next pin / stun end
```

Until F6 phase pins: treat enter→mid as dual-green prefix; leave-to-escape as
separate GREEN skill; do **not** open-loop full `mb_room_*` as pure.
