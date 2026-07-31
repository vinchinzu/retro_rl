# SM-TIGHTEN-02C Implementation Note

Recipe C only was applied to `hj_return_gray_exit`.

- Before: periodic `RIGHT+B+X` for four frames every 30, otherwise `RIGHT+B+A`.
- After: `RIGHT+B+X` for the first four frames, then continuous `RIGHT+B`.
- Preserved the 600-frame timeout and `hj_shaft_to_business: gray door failed` error label.
- Preserved Sova cleanup and all Wave-3 02B bomb-tunnel and settle knobs.

No frame savings are claimed without a re-record. Planner verification remains:

```bash
uv run python super_metroid/scripts/record/continuous.py --to kraid --no-video
```

The 02B interaction risk remains: if the stacked gray-door change fails in the
continuous run, roll back this Recipe C loop independently before changing the
02B settings.

This task did not STATUS-promote and did not forge progression RAM.
