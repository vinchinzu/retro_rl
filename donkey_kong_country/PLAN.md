# DKC Harness Plan

## Autosplit + Timing
- Lock down level start using in-game timer + movement change.
- Confirm level ID -> name mapping for the first few levels (Jungle Hijinks).
- Add more RAM addresses for transitions (goal, bonus, death) to tighten splits.

## Recording + Playback
- Add record mode in `run_bot.py` with `.bk2` output directory.
- Add replay helper (load `.bk2` and play back).
- Track recording metadata (state, timestamp) in a manifest.

## Training + Playback
- Standardize recording paths and metadata to align with training scripts.
- Add a small CLI for listing recordings and converting to MP4.
