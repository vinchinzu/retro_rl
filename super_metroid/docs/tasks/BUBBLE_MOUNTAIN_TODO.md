# Bubble Mountain — residual product backlog

**Last updated:** 2026-08-03 (**R19 pure GREEN** Bubble→Bat 2012f → `0xB07A`;
next: graph compose / Bat→Speed)

> **Scope note:** Structural refactor is landed. Pure GREEN Bubble→Bat is
> **closed** (R19 residual). Continuous/STATUS tip advance is planner compose —
> not automatic from pure GREEN alone.

---

## Structural refactor (landed 2026-08-03 — not open product work)

Do **not** re-do in pure R-series cards:

- [x] Extract `routes/kpdr/bubble_mountain.py` + `bubble_mountain_mid.py` (mid budget loop)
- [x] Phase helpers: lower → repin → **one** `bubble_run_mid` → door
- [x] One `true_ground` vs `stand_pin` model
- [x] Product surface: `play_bubble_to_bat_cave(session)` only; recon via probe wrappers
- [x] Delete dead `_BUBBLE_PEAK_CROSS_Y`, `_BUBBLE_ALT_WJ_PERIOD`, `_BUBBLE_DOOR_X`
- [x] Unit tests for ground / phase predicates
- [x] Fix `test_k4_speed_branches` for Cathedral-first Bubble path
- [x] **Lip / approach use stand_pin** (not true_ground-only) — extract
      regression fixed 2026-08-03; pure back to R11 envelope
      (`launched=True` max_x=349 min_y=260 frames≈7227). Climb reseat stays
      true_ground (R11). Test:
      `test_bubble_launch_lip_uses_stand_pin_not_true_ground_only`.
- [x] Code-quality pass: `bubble_mountain_params.py` (shared constants, no
      cycle), single mid entry (`start=launch|climb`), mid control locals not
      on `BubbleTrack`, one `_play_bubble_full`, public `bubble_on_launch_lip`.

---

## P0 — spine pure (`SM-K4.4-PURE` / R13+)

Baseline (R13): full pure from CATH-04 source → `max_x=408` `min_y=260`
**`phase_c_hit=True`** top red. Phase C pin marginal ~`(301,429)` — place
finish still holds from higher shelf only.

- [x] **`SM-K4.4-PURE-R12`**: lip stand_pin extract restore (R11 envelope)
- [x] **`SM-K4.4-PURE-R13`**: floor-reclimb after height class → Phase C pure
      green. Residual:
      [`SM-K4.4-PURE-R13-residual.md`](SM-K4.4-PURE-R13-residual.md)
- [x] **`SM-K4.4-PURE-R14`**: human save-door runway (maprando left climb, no
      Ice) wired; pure holds R13 envelope. Residual:
      [`SM-K4.4-PURE-R14-residual.md`](SM-K4.4-PURE-R14-residual.md)
- [x] **`SM-K4.4-PURE-R15`**: double-WJ open-loop clears Phase D on human
      runway pin (min_y≈76–138, x≥300 @ y≤200); pure still lip path R13
      envelope. Residual:
      [`SM-K4.4-PURE-R15-residual.md`](SM-K4.4-PURE-R15-residual.md)
- [x] **`SM-K4.4-PURE-R16`**: pure seats fire solid + fires R15 open-loop;
      min_y=228 free-air (was 260 lip); Phase D still red (mx200 pocket).
      Residual:
      [`SM-K4.4-PURE-R16-residual.md`](SM-K4.4-PURE-R16-residual.md)
- [x] **`SM-K4.4-PURE-R17`**: velocity RECON + `bubble_mountain_primitives`
      (double-WJ named); pure still no p132 / Phase D red. Residual:
      [`SM-K4.4-PURE-R17-residual.md`](SM-K4.4-PURE-R17-residual.md) ·
      techniques [`BUBBLE_TECHNIQUES.md`](BUBBLE_TECHNIQUES.md)
- [x] **`SM-K4.4-PURE-R18`**: max-left seat + no arm-pump + R18 WJ2; pure
      earns p132+pose84 min_y≈159; Phase D red (enemy AI phase). Residual:
      [`SM-K4.4-PURE-R18-residual.md`](SM-K4.4-PURE-R18-residual.md)
- [x] **`SM-K4.4-PURE-R19`**: enemy-phase fire + sticky Super door → full pure
      GREEN ordinary `0xB07A` (**2012f** ×2). Residual:
      [`SM-K4.4-PURE-R19-residual.md`](SM-K4.4-PURE-R19-residual.md)
- [x] **Phase E** ordinary `0xB07A` (R19 sticky right WJ + Super)
- [x] Successor `scratch/post_bubble_to_bat_pure.state`

Phase map: [`SM-K4.4-PHASE-LADDER.md`](SM-K4.4-PHASE-LADDER.md) · stagnation:
[`HARD_ROOM_SPLITS.md`](HARD_ROOM_SPLITS.md).

---

## P1 — after pure GREEN only

Planner-serial; never in a farm batch or while pure is RED.

- [x] **`SM-K4.4-GRAPH`**: Graph edge Bubble→Bat `controller_dev` (R19 closeout)
- [ ] Continuous tip compose (`--to` bat / speed / …) + dual re-record + STATUS
- [ ] ★ Bat Cave → Speed Hall → Speed pure stack (next serial from
      `post_bubble_to_bat_pure`)

---

## P2 / parked

- [ ] Post-Speed shortcut: Farm→Bubble scaffold (`play_farm_to_bubble` still
      `_scaffold_exit`) — not first Bubble path
- [ ] Wave branch Bubble→Single Chamber pure
- [ ] Dual-track room practice `room_acb3_from_b07a_to_aedf` (separate track;
      not continuous integrity)
- [ ] Freeze unit/pure probes for green **R5/R6** as automated regression —
      if structural fix only unit-tests ground predicates, remaining: emulator
      pure phase-pin tests (lower + lip hold under full pure)

---

## Explicitly rejected (do not re-ship without new pin evidence)

From R11 residual (load-bearing):

| Rejected | Why |
|----------|-----|
| Lip walk-left + dash / pre-charge run | Height regress (`min_y≈365`) |
| Mid-iso dash without enemy clearance | Natural KB thrash + height regress |
| Left-column top hunt | Free-air ceiling ~y228; not top band |
| Period / mid-high window-only on current lip arc | No Phase C; stagnation class |
| Floor WJ / bomb chains / save-room run-out | No height class / not product |

---

## Process pointers

| Doc | Role |
|-----|------|
| [`SM-K4.4-PHASE-LADDER.md`](SM-K4.4-PHASE-LADDER.md) | Phases A–E + capture/climb probes |
| [`HARD_ROOM_SPLITS.md`](HARD_ROOM_SPLITS.md) | Hard-room split + stagnation @ 3 |
| [`SM-K4.4-PURE-R12.md`](SM-K4.4-PURE-R12.md) | Open pure card (trajectory class) |
| [`SM-K4.4-PURE-R12-residual.md`](SM-K4.4-PURE-R12-residual.md) | R12 RECON residual |
| [`SM-K4.4-PURE-R11-residual.md`](SM-K4.4-PURE-R11-residual.md) | Rejected list + spin-apex fix |
| [`../SOURCE_STATES.md`](../SOURCE_STATES.md) | Cathedral-first Bubble sources |
| Code home | `routes/kpdr/bubble_mountain{,_params,_mid}.py` |

### Full pure verify (product proof only)

```bash
uv run python super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bubble_to_bat_pure.state \
  --pin-json super_metroid/debug/bubble_to_bat_pure_pin.json
```
