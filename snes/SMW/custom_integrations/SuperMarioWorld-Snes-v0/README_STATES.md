# SMW custom states

| State | Status | Notes |
|-------|--------|-------|
| `Chained_YoshiIsland3` | OK | Natural entry after YI2 clear + OW up. Big Mario. YI3 hillclimb seed clears. |
| `Chained_YoshiIsland4` | OK | Natural entry after YI3 clear + OW right. Big Mario, exits=2. |
| `Chained_AfterYI4_OW` | OK | OW after natural YI4 clear (exits=3); castle path open. |
| `IggysCastle` | OK | Outdoor castle door (`trans=0x25`) after chained YI4 + enter. Not DP1. Clear: `smw_iggys_castle/recording_001_clear.json`. |
| `DonutPlains1_from_yi4_rec002` | OK | Outdoor DP1 (`trans=0x15`); was mislabeled `IggysCastle`. |
| `DonutPlains1_chained` | OK | DP1 after first pipe (`trans=0x0A`). |

## Yoshi Island chain

```bash
# Rebuild chained YI3/YI4 entry, after-YI4 OW, and IggysCastle
uv run python -m SMW chain-yi

# YI4 natural-entry clear (already promoted as recording_004_chained_clear.json)
uv run python -m SMW -l yi4 play --state Chained_YoshiIsland4

# Verify YI4 on chained entry
uv run python -m SMW -l yi4 verify \
  --actions snes/SMW/optimizer/runs/smw_yoshi_island_4/recording_004_chained_clear.json \
  --state Chained_YoshiIsland4

# Record / verify Iggy from natural entry
uv run python -m SMW -l iggy play
uv run python -m SMW -l iggy verify \
  --actions snes/SMW/optimizer/runs/smw_iggys_castle/recording_001_clear.json
```

Route id: `smw_yoshi_island` / `yi_chain` (YI2→YI3→YI4→Iggy).

YI4 seed notes:

- `recording_002_*` / `recording_003_*` — package `YoshiIsland4` only (small Mario).
- `recording_004_chained_clear.json` — natural entry; 1 idle frame pads Evaluator free-frame resync.
