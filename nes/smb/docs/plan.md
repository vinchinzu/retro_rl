# Plan

## Goal

Keep the Clean power-on any% warp gate in `docs/STATUS.md`. Later work is
a faster Clean seed, and a separate 32-exit route. A hybrid showcase or a
warpless extract does not replace that gate.

## Warp claw-back

1. Fold a faster body only after a Clean power-on 3/3 with zero mid-attempt
   loads. Until then the gate stays
   `models/smb_1_1_to_ending_natural_82.json` (21,559 frames).
2. Do not absolute-stitch a HappyLee or flamexx slice into that seed.
   Retime from control. See `docs/TAS_ADAPT.md`.
3. Pure HappyLee (#1715M) under fceumm is still open. A native FCEUX or
   BizHawk capture of the untouched movie is not a stable-retro Clean result.
4. Re-record the power-on MP4 only when the Clean seed changes. The
   preserved capture manifest is the older 22,005-frame file.

## 32-exit route

Movie is HappyLee and Mars608 warpless #3728M, not warp #1715M.

The deleted handoff (not re-run here) said 1-4 and 2-1 had been extracted,
2-2 had not, and the unique TAS peak took a Cheep-cheep hit. Do not patch
that body, and do not start 2-3 from a miss. Open command:
`annotate_fm2 --search 2-2 --from-pred --export`, after
`record_warpless --to 2-1`.

## Residual stepper

`docs/RESIDUAL.md`. Next physics work is collision as a `World` query.
That is not a route-clear claim.

## Parked commands

```bash
uv run python -m smb.scripts.run_warp_finish --mode poweron --record
uv run python smb/scripts/run_1_1.py --natural-entry --trials 3
uv run python -m smb.scripts.run_1_2 --predecessor stairs --trials 3
uv run python -m smb.scripts.run_reactive_warp --retime-4-1 --retime-4-2 --retime-8-2
uv run python -m smb.scripts.fold_continuous_policy
uv run python -m smb.scripts.run_1_2_flag --record --trials 2
uv run python -m smb.scripts.record_happylee --to ending
uv run python -m smb.scripts.pure_hl status
uv run python -m smb.scripts.measure_residual
```
