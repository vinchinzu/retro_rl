# Plan: The Great Waldo Search

The continuous five-scrolls clear is in `docs/STATUS.md`. This file is only
unfinished work.

## Next

1. Isolate a scene-complete or scene-id byte that does not track the camera.
2. Read score only after the bonus animation settles. Scene 5 needs at least
   200 frames.
3. Re-record `scripts/record_full_run.py` when the capture should be refreshed.
   Do not rebuild Scenes 3 through 5 from Cleared states and call that the
   continuous path.

Cursor addresses stay in `docs/ram_map.md`. Open-loop coordinates stay in the
scene scripts.
