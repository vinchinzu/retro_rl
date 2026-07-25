# Final Fight — Plan

Ladder #2. Beat-’em-up benchmark: movement, melee, health, screen locks.

## Working rule

Not gated on uninterrupted evaluation early. Prefer:

1. Dev `.state` files under `custom_integrations/FinalFight-Snes/`
2. Segment policies (menu → Stage1 start; fight nearest; walk-right unlock)
3. Retries from save on failure
4. Chain segments once a screen clear is reliable
5. Continuous title-to-credits only as later hardening

## Milestones

1. **Scaffold** — ROM setup, integration stubs, docs, BT skeleton, RAM probe
2. **Stage1.state** — past character select, ready to fight
3. **RAM** — player X/Y/health, enemy slots (diff probes + tentative map)
4. **Segment clear** — from Stage1.state, clear one locked screen / fight ✅
   (`run_stage1_segment.py`; inverted-Y + attack cadence)
5. **Stage chain** — more screens ✅ (room-1 alley unlock cam 1536→1729).
   Waves 1–2 clean; w3/w4 punch trades only; unlock at lives≥2 / HP **38**
   (`Stage1_PostUnlock_L2`). CLEAR_AREA ≠ stage clear
6. **Damnd spawn** ✅ — from `Stage1_PostUnlock_L2`, cam **2304** / room 2,
   `0x11E0=01`, saved `Boss.state` (+ `Stage1_BeforeBoss`). Report:
   `recordings/survive_r1_postunlock_l2h/`. Prefer Boss refresh from
   `Stage1_Clear_w3_cam1728` → **HP 40**.
7. **Damnd fight** — door thugs 1–5 (peaks 36/60/64/42/82) + Damnd HP44 ✅
   at HP40 → `Boss_Drawn` / `Stage1_Clear` (kill-frame). Prefer segment from
   `Boss_PostThug5`. Open: legitimate `0x0CD2` death (UF leaves flag 0).
8. **Stage 2 subway** — CLEAR_AREA bridge from `Stage1_Clear` → `Stage2` ✅
   Prefer `Stage2_Clear_w2_cam537`. HP148 clear **HP54/L2** /
   **HP67/L1**. Cam994 pack clear ✅; area0 cam994 softlock → area1
   cam1792 ✅. Area1 dual-pack clear ✅ (hit-and-run). Cam2561 softlock
   → area2 cam3840 ✅ (**HP54** entry; scroll chip fixed). e69 early
   **JD90+toward+Y** → pack clear @**HP54** ✅; HP134 wave @HP37 ✅;
   cam4130 softlock → area3 / **`Boss2`** (`0x11E0=01`) ✅; Drawn ✅;
   UP+Y throw kill → **`Stage2_Clear`** ✅ (HP37 / boss UF)
   (woken Mid flee-save blocked by chains).
9. **Stage 3 West Side** — CLEAR_AREA bridge from `Stage2_Clear` →
   Break Car bonus (`round=06`) → **`Stage3`** ✅ (HP80 / cam619 /
   round **02**). Wave1 @**HP80** (JD-left edge) ✅; Mid_p66 ✅;
   wave2 clear from Mid @**HP31** (`Clear_w2`) ✅; wave3 → Clear_w3
   @**HP31** ✅; wave4 Andore HP216 → Clear_w4 @**HP31** ✅ (edge-JD).
   Wave5 dual 142+96 → **cleared** ✅ (`Clear_w5_real_p48`; split+heal
   + LEFT+Y wait-KD). Cam931 softlock → Area1 cam2560 ✅. Area1 HP250
   thug / Boss3 open (`ENTITY_HP_MAX=252`). Continuous still dies
   mid-wave2 after Mid.
10. **Longer run** — optional title → Stage 1 without relying on mid-run saves

## Behavior tree (segment policy)

```text
if player_dead / continue: handle_continue()
elif level_complete: advance()
elif boss_active: boss_segment()
elif enemies_present: align_vertical → fight_nearest
else: walk_right (stall → UP/DOWN+RIGHT lane nudge)
```

Implemented in `final_fight/policy.py` using `snes_oneshot.combat.build_segment_tree`
(`align_vertical_action`, `fight_nearest_action`, `WalkProgress`).
`Sequence` is reactive (re-checks gates each tick) so clears fall through to
walk.

Reusable combat / chain helpers: `snes_oneshot/combat.py`,
`snes_oneshot/segment_runner.WaveChainTracker`.
