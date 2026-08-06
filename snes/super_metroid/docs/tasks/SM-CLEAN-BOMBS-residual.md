## Residual — rr-siz / SM-CLEAN-BOMBS

### Result
RED

Clean continuous `--to bombs --clean` reaches Bomb Torizo with zero resource
writes, then dies inside the hash-pinned `pit_to_post_torizo` segment. Assisted
`start_to_bomb_torizo.json` / any assisted stem **not** overwritten.

### Files changed
- `docs/tasks/SM-CLEAN-BOMBS-residual.md` — this residual (ephemeral)
- `docs/CLEAN_TRACK.md` — C-BOMBS ★ → RED pin (honest; no green claim)
- `docs/STATUS.md` — Clean track secondary only (prefix facts + RED)
- `docs/routes/MILESTONES.md` — C-BOMBS row reflects RED residual
- `docs/routes/BACKLOG.csv` — SM-CLEAN-BOMBS residual notes; BT-ECONOMY ready
- `docs/tasks/SM-CLEAN-BOMBS.md` — acceptance checkboxes for this run
- local (gitignored): `recordings/bombs_clean.json`, `bombs_clean_run.log`

### Verify paste
```bash
uv run python snes/super_metroid/scripts/record/continuous.py \
  --to bombs --clean --no-video \
  --report snes/super_metroid/recordings/bombs_clean.json
# exit non-zero
# RuntimeError: bombs run failed; report=…/bombs_clean.json
# root: pit_to_post_torizo exit mismatch: room 0x9804 != required 0x92FD;
#   phase boot_or_menu; items 0x0000 missing 0x1004; ammo (0,0,0) below (10,0,0)
# intervention Clean; energy/ammo writes=0; deaths counter=0 (no continue)
# assisted baseline SHA-256 (unchanged):
#   start_to_bomb_torizo.json
#   8c72099469165e6bed8b5177365336b68f4808b5eb67b33de43f1689ae632ac2
```

**Prefix (clean, this run, 2026-08-06):**

| Split / event | Frame | Notes |
|---------------|------:|-------|
| morph_ball | 26,824 | matches assisted morph tip after Ceres shave |
| first_missiles (cap 5) | 27,678 | progress `max_missiles` |
| blue_brinstar_missiles (cap 10) | 29,440 | progress `max_missiles` |
| pit_natural_entry | 32,560 | hp=84, mis=5/10, sel=0 |
| bombs item bit `0x1000` | 41,243 | progress `collected_items` 4→4100 |
| BT activated / peak HP 800 | — | integrity final_conditions |
| BT HP 0 / parlor settle | — | **fail** |

**Fail pin:** after bombs collect, hash-pinned policy still in room `0x9804`
(Bomb Torizo); final phase `boot_or_menu`, pose 216, inventory wiped
(health 99 / missiles 0/0 / items 0) — death during BT portion of
`pit_to_post_torizo` under Clean (no ammo refill). Assisted baseline used
**116** missile refill writes for the same tip.

### Acceptance
- [x] Ran clean continuous `--to bombs --clean` with `*_clean` report only
- [x] Resource writes all zero (clean integrity extras true until wipe)
- [x] Assisted bombs baseline files unchanged
- [ ] Clean report success + bomb_torizo outcome — **RED**
- [x] Residual PROCESS fields → SM-CLEAN-BT-ECONOMY

### Residual risks
- Failure mode is **BT ammo/death economy** on the accepted hash-pinned
  `pit_to_post_torizo` policy (not early-room geometry). Assisted path sprays
  missiles with unlimited ammo.
- Older gitignored `start_to_bomb_torizo_clean.json` (2026-08-01) also RED:
  fight timeout, `min_hp=800` (zero damage), death, 3/10 missiles remaining —
  same economy class.
- Do not swap continuous default or assisted stems while fixing economy.
- Any shared `combat/bomb_torizo.py` / early policy knob requires **assisted
  bombs re-verify** after the one-knob change.

### Next action (required)
- **Next card ID:** SM-CLEAN-BT-ECONOMY (bead for one-knob ammo/death fix)
- **One change:** make Clean BT clear with natural 10-missile capacity (and
  natural packs) without ammo writes — prefer existing combat model over
  re-solving geometry; re-verify assisted `--to bombs` after any shared edit
- **Source state:** natural continuous entry preferred; optional capture after
  bombs PLM if isolating fight (`natural_bomb_torizo_active` pattern)

### Non-claims
- Did **not** claim Clean bombs tip green or dual re-verify
- Did **not** STATUS-promote Clean as primary tip / change M5 gate
- Did **not** overwrite assisted `bombs.json` / `start_to_bomb_torizo.json`
- Did **not** implement BT economy fix in this card
- Did **not** re-label assisted morph/bombs greens as Clean

### Probe pin (if pure/geometry)
```text
# residualPinLine: room=0x9804 phase=boot_or_menu pose=216 x=234 y=187
#   door_transition=0 frames=45703 last_event=bombs@41243 pit_entry_hp=84 mis=5/10
# failure=pit_to_post_torizo_exit_mismatch clean_ammo_writes=0
```
