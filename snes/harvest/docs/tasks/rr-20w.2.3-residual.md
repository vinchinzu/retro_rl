## Residual — rr-20w.2.3 D2 field clearing (CLOSED)

**Status:** GREEN, then gutted. Bead is closed. Do not leftover smash. Do not
ship from the 18:09 pin. No `--video`. No STATUS.

### Verified this session

- **Clean power-on D2 LIVE GREEN.** `--stop-after-d2-clear`
  `recordings/power_on_d2_farm_clear.json`: **393223f / 371309 planner /
  1059.6s**. End D2 **18:01** farm `(41,58)` stam 52 **$100**. Debris 0,
  planted 8, wet 8, `shipped_before_17=True`. Clean `ram_writes=0`,
  `mid_run_state_loads=0`. Leave pin `Y1_D2_PowerOn_FarmClear`.
- **Composer gut.** Dead `phase_already_clear` / tactic `_skip` /
  `skip_chunks` / unused leftover_exec re-exports cut. Leftover order is
  only `next_d2_spec`. `d2_work.py` 985 LOC. Source-grep tests and static
  leftover lists in tests deleted. Report contract locked in
  `PowerOnD2FarmClearReportTests`.

### Exact next action

Claim **one** from `bd ready -l harvest -l spine`: `rr-20w.2.4` or `rr-3ae8`.
Delete this residual when that bead is claimed.

### Non-claims

- No STATUS promotion
- Did not start from `Y1_D2_Morning_After_D1`
- Did not leftover smash
- Did not ship from the 18:09 leftover pin
- Did not treat CrossMap origin-return as shop success
