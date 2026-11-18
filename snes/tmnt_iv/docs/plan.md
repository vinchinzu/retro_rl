# Plan: TMNT IV: Turtles in Time

Verified result: [STATUS.md](STATUS.md). Contract: [ASSIST_CONTRACT.md](ASSIST_CONTRACT.md).

The Clean power-on Hard clear and independent native-audio video replay
are verified at M8. No completion blocker remains.

## Next

1. **Preserve the regression baseline.** Keep the original Leo I tape,
   complete Clean tape, dry report, published replay report, and integrity
   checks. A normal reproduction is:
   `uv run python -m tmnt_iv.scripts.run --replay
   snes/tmnt_iv/recordings/tmnt_iv_full_hard_clean.tape.npz --video
   --native-video --report snes/tmnt_iv/recordings/tmnt_iv_full_hard_clean.json`.
2. **Reduce route time.** Measure Technodrome and Slash improvements
   against the 01:13:47.558 / 74-damage baseline. Require zero life losses,
   zero assists, and a complete natural power-on replay before promotion.
3. **Reduce search cost.** Gate rescues on threat proximity, reduce parser
   work in rollouts, and trim branches per lead. Keep search wall time
   separate from committed game time.

## Decided

- Safety lives in the guard. The base policy owns progress tactics; new
  dodges belong in the macro search rather than another reactive stack.
- One CLI (`scripts/run.py`), one loop (`run_trial`), one wrapper
  (`GuardedPolicy`). No sibling runners.
- Default turtle is Leo. Mike's larger health bar lost to Leo's reach at
  Metalhead; revisit only with a measured A/B.
- Ending recovery may append neutral input only after the ending starts.
  Missing gameplay still fails; the recovered tape must pass a normal
  power-on replay before publication.
- Credits demonstrations reuse HP/lives/stage RAM. Exclude their changes
  from gameplay accounting while retaining all intervention audits.
