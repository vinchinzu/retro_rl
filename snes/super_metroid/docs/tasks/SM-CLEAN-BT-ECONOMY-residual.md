## Residual — rr-5of / SM-CLEAN-BT-ECONOMY

### Result
GREEN

Clean continuous `--to bombs --clean` clears Bomb Torizo on natural ammo
(entry often ~3/10 missiles) with **0 resource writes**, hybrid BT fight +
hash-pinned exit tail to Parlor. Dual clean integrity **49,321f** ×2.
Assisted bombs re-verify GREEN **45,711f** (116 missile refill writes; full
hash-pinned policy unchanged). Assisted stems **not** overwritten.

### Files changed
- `combat/bomb_torizo.py` — clean kite defaults; HP-only idle (no open-bus
  boss_bits trap); wait for Crateria boss bit after HP 0
- `routes/kpdr/early_post_morph.py` — clean-only hybrid: policy → fight →
  policy exit tail (assisted path stays full hash-pinned replay)
- `policy.py` — `play_policy` optional `stop_when` / `action_slice` /
  `require_exit`
- `tests/test_bomb_torizo_strategy.py` — defaults + open-bus fire regression
- `docs/tasks/SM-CLEAN-BT-ECONOMY-residual.md` — this residual
- `docs/tasks/SM-CLEAN-BOMBS-residual.md` — superseded pointer
- `docs/CLEAN_TRACK.md`, `docs/STATUS.md` — Clean bombs secondary GREEN
- `docs/routes/BACKLOG.csv`, `docs/routes/MILESTONES.csv` — C-BOMBS / economy
- local (gitignored): `recordings/bombs_clean.json`,
  `bombs_clean_reverify.json`,
  `bombs_assisted_reverify_after_bt_economy.json`

### Verify paste
```bash
uv run python snes/super_metroid/scripts/record/continuous.py \
  --to bombs --clean --no-video \
  --report snes/super_metroid/recordings/bombs_clean.json
# [GREEN] tip=bombs frames=49321 room=0x92FD items=0x1004
# integrity clean_resources_zero; missile writes=0
# dual: bombs_clean_reverify.json also 49321f GREEN

uv run python snes/super_metroid/scripts/record/continuous.py \
  --to bombs --no-video \
  --report snes/super_metroid/recordings/bombs_assisted_reverify_after_bt_economy.json
# [GREEN] frames=45711; missile refill writes=116
# start_to_bomb_torizo.json SHA-256 unchanged:
#   8c72099469165e6bed8b5177365336b68f4808b5eb67b33de43f1689ae632ac2
```

**Clean splits (2026-08-06):**

| Split | Frame | Notes |
|-------|------:|-------|
| morph_ball | 26,824 | matches assisted morph tip |
| first_missiles / blue | 27,678 / 29,440 | cap 5 → 10 |
| pit_natural_entry | 32,560 | |
| bombs | 41,243 | item bit 0x1000 |
| bomb_torizo_defeated | 46,946 | hybrid fight; entry often mis≈3/10 |
| bomb_torizo_exit | 48,638 | Flyway |
| pit_to_post_torizo | 49,321 | Parlor settle 0x92FD |

### Acceptance
- [x] Clean GREEN bombs tip integrity (dual 49,321f)
- [x] Assisted GREEN after shared combat/policy edit (45,711f)
- [x] Residual SM-CLEAN-BT-ECONOMY written
- [x] Assisted stems not overwritten
- [x] One-knob: BT strategy kite defaults + clean hybrid only

### Residual risks
- Hybrid is **clean-only**; assisted still depends on hash-pinned spray + ammo
  refill (116 writes). Promoting hybrid for assisted is a separate card.
- Exit uses last **2000** frames of `pit_to_post_torizo.json` after boss bit —
  brittle if that tail is re-recorded.
- Continuous BT entry often has **~3 missiles**; strategy farms drops while
  kiting — do not tighten fire_period without re-probe from continuous entry.
- Open-bus `boss_bits` from `env.get_ram()` still false-positive
  `enemy_defeated` in features; fight path keys on HP only.

### Next action (required)
- **Next card ID:** SM-CLEAN-STAB (or planner STATUS secondary only — dual already green)
- **One change:** optional STATUS/MILESTONES dual promote wave; then spore clean park lift
- **Source state:** continuous clean power-on; natural_bomb_torizo_active still valid pure probe

### Non-claims
- Did **not** change assisted `start_to_bomb_torizo.json` / `bombs.json` stems
- Did **not** STATUS-promote Clean as primary tip / M5 gate
- Did **not** claim assisted hybrid or frame-parity with assisted bombs
- Did **not** re-label assisted morph/bombs greens as Clean

### Probe pin (if pure/geometry)
```text
# residualPinLine: continuous_entry room=0x9804 mis=3/10 hp=90 x=131 y=192
#   sm=0xAA12 enemy_hp=800; clean kite min=100 max=160 jump_hold=28 period=40
#   fight~5338f to boss bit; exit tail policy[-2000:]; parlor 0x92FD @49321
```
