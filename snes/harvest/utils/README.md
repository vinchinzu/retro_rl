# Utils

Development and debugging utilities. These are not required for normal bot operation.

## Files

- `find_ram.py` - RAM exploration utility for finding game memory addresses
- `state_builder.py` - Build save states by replaying task sequences
- `capture_task_ram.py` - Replay a recorded task and save a focused RAM trace (`trace.csv` + `summary.json`)
- `capture_harvest_mode_ram.py` - Capture the trimmed Day 9 harvest route and derive shipped-value/deposit-count metrics
- `dialogue_branch_probe.py` - Replay a dialogue segment from a task anchor, branch inputs, and compare RAM outcomes
- `record_l2_house_wife_bed.sh` - Canonical recorder wrapper for tracing House L2, talking to wife, then going to bed
- `extract_recording_walkable_tiles.py` - Extract observed player tiles and A-press windows from a task trace
- `merge_tasks.py` - Merge multiple task recordings into one
- `check_map.py`, `map_dump.py`, `map_analyzer.py`, `map_visualizer.py` - Map data exploration

## Example

Capture a saved RAM trace for the recorded TV weather interaction:

```bash
uv run python utils/capture_task_ram.py --task tv_weather --state latest
```

That writes artifacts under `debug_alignment/ram_capture/tv_weather_latest/`.

For romance/event analysis, you can watch the full `Romance` section and isolate a task slice:

```bash
uv run python utils/capture_task_ram.py \
  --task get_berry \
  --state latest \
  --start-frame 7210 \
  --end-frame 7576 \
  --watch-section Romance
```

For berry-route analysis, remember the payment cutoff: produce has to be in the
shipping bin before `17:00` (`5:00 PM`) or the shipper has already collected
for the day and you will not get paid that evening.

For the current Day 9 harvest-mode baseline, capture the trimmed farm-only
route plus derived shipping metrics with:

```bash
./.venv/bin/python utils/capture_harvest_mode_ram.py
```

That writes the usual trace under
`debug_alignment/ram_capture/harvest_mode_day9/` and appends
`harvest_metrics` to the summary JSON.

To compare dialogue branches from the same anchor:

```bash
uv run python utils/dialogue_branch_probe.py \
  --task get_berry \
  --state latest \
  --anchor-frame 7429 \
  --end-frame 7570 \
  --watch-section Romance \
  --branch recorded_down \
  --branch flattering_default \
  --override flattering_default@7430-7436=none
```
