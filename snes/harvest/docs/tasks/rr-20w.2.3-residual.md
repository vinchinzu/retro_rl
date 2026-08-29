## Residual — rr-20w.2.3 D2 field clearing

**Status:** IN PROGRESS. Farm→spa from Partial is live-green. Last
boulder is gone. Wood is 36→5. Do not record `--video`.

### Verified this session

- `farm_to_west_gate_waypoints` lives in `harvest/maps/farm_gate.py`
  (map_routes was already past 1k). Start-aware prefixes:
  - **SE leftover** `(54,42)` ~(879,686): y=39 dirt between stump
    belts, then x=13 join. South-field hop 0 `(136,600)` run-right
    overshoots from x≥216.
  - **NE wood checkpoint** `(48,13)`: y=13 x=46–50 cannot DOWN
    (`EAST_SPUR_FA_SOUTH_OPEN_X=51`), then west to `(39,17)`.
  - Stump-chunk success at stam-low now `insert_spa` before the next
    2×2 smash (not only after rocks).
- Live farm→spa→return from `Y1_D2_Leftover_Partial` **GREEN**:
  4→100 in **4579f** (`recordings/hot_spring_partial.json`), pin
  `Y1_D2_Partial_Spa`. BEFORE leftover spa was 20000f FAILURE still
  on farm `(51,38)`. Δ **−15421f** and it actually left.
- Same corridor from `Y1_D2_Wood_Checkpoint` `(48,13)` **GREEN**
  (`recordings/hot_spring_wood_checkpoint.json`).
- Last boulder from `Y1_D2_Partial_Spa` **GREEN**: 1→0 in 1623f, pin
  `Y1_D2_After_Last_Rock` (stam 100→88).
- Wood from that pin: 36→5 (NW/NE/SW/SE), two leftover spas GREEN,
  then ditch-lip spa red. Pin `Y1_D2_Wood_Progress` `(11,25)` stam 4,
  **5 stumps** `(34,42) (42,44) (36,51) (60,55) (42,58)`.
- Unit: hot_spring + map_config + d2_work + d2_farm_chunks **94+**
  passed (full related set 102 earlier). No STATUS.

### Exact next action

Do not third-20k spa from `Y1_D2_Wood_Progress` `(11,25)` (two serial
reds: house-south, then ditch-lip join still hugs LEFT at `(11,25)`).
Remaining 5 stumps need a pocket leave that is not another leftover
`--section stumps` from that pin.

SE rocks from `Y1_D2_After_Stones` `--no-spa` cleared **6** then
`stamina_low` (76→4, 47→41). 13 SE 2×2 do not fit in 76 stam; drop
`--no-spa` so leftover spas via the greened SE y=39 row:

```bash
HEADLESS=1 uv run python -m harvest.scripts.d2_leftover_probe \
  --section rocks --chunk se --state Y1_D2_After_Stones \
  --timeout 200000 --out recordings/d2_leftover_smash.json
```

Do not start from `Y1_D2_Morning_After_D1`. Do not STATUS. Do not
400k `--section all`.

### Non-claims

- No STATUS promotion
- No natural power-on Day 2 farm-clear
- No D2 movie / `--video`
- 5 stumps remain on `Y1_D2_Wood_Progress`
- After_Stones SE rocks still need spa after the first 6
- Ditch-lip `(11,25)` spa is a live hug, not a leftover 12k cap
