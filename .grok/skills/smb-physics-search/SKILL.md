---
name: smb-physics-search
description: >
  Search Super Mario Bros. windows in smb.approx, grade per-frame claims with
  smb.predict, and halt live replay at the first miss. Residual R(τ) is the
  search-model keep/reject. Use for 8-3 phase search, jump polish, or TAS
  adapt — not for recording a new human tape.
---

# SMB physics search

`smb.approx` is the physics engine. The emulator is the grader.

1. Roll the window with `approx.step` / `rollout` (no ROM).
2. Attach a claim per frame (`smb.predict.predict_step` / `player_claim`).
3. Keep the stepper as a search model only while
   `compute_residual_profile(...).can_keep_as_search_model()`.
4. Live commit: `grade_trajectory` then `halt_plan` / `first_miss_index` —
   stop at the first missed claim. Do not halt_plan on residual `fd_pi`.
5. Prefer TAS adapt (`docs/TAS_ADAPT.md`) over emulator hill-climb when a
   public movie already has the trick.

Shared grammar: `retro_harness.predict`. Oracle 8-3 still gates first
obstacle → x900 → x1600 → flag; max_x alone is not success.
