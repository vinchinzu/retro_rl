## Residual — rr-20w.2.3 D2 field clearing

**Status:** IN PROGRESS. SE rocks leftover from After_Stones is live-green
with spa. Last boulder is gone. Wood is 36→5. Do not record `--video`.

### Verified this session

- SE leftover spa from After_Stones `--no-spa` was 6 then `stamina_low`
  at `(59,40)`. With spa, y=39 dirt reached the ditch lip then **LEFT-held
  into A6** `(9–10,25)` (`route_mountain` 20k timeout, leftover `(11,25)`).
- RAM dump After_Stones: y=25 x=9–10 is `0xA6` (not in `FARM_WALKABLE`);
  y=24 x=9–16 is dirt/`A0`. Fix (no tape): drop south-field
  `run_direction="left"` on `(136,392)`; hop `(216,392)` `(13,24)` then
  west on y=24. Ditch-lip prefix UPs to `HOUSE_COLUMN_DIRT_Y_PX` instead
  of east to `(200,408)`.
- Live SE rocks **GREEN** (`recordings/d2_leftover_smash.json`):
  47→34 (**13/13 SE**) in **14540f** / 04:02.33. Spa 4→100 in **4519f**
  (`HOT_SPRING` 5041→9560). Before spa hug was 20000f FAILURE still on
  farm `(11,25)`. Δ spa **−15481f** and it actually left. End `(59,51)`
  stam 16, hammer. No pin saved (checkpoint 15k not reached).
- Unit: hot_spring + map_config + d2_work + d2_farm_chunks **95** passed.
  No STATUS.

### Exact next action

Do not third-20k spa from `Y1_D2_Wood_Progress` `(11,25)` (two serial
reds last session; ditch-lip LEFT is now the y=24 join, un-benched from
that pin). Remaining **5 stumps** there still need a pocket leave that
is not another leftover `--section stumps` from that pin.

SE is empty on After_Stones geometry; remaining **34** boulders are
nw/ne/sw. Start SW (farmer already `(29,35)`):

```bash
HEADLESS=1 uv run python -m harvest.scripts.d2_leftover_probe \
  --section rocks --chunk sw --state Y1_D2_After_Stones \
  --timeout 200000 --out recordings/d2_leftover_smash.json
```

Do not start from `Y1_D2_Morning_After_D1`. Do not STATUS. Do not
400k `--section all`. Do not redo SE.

### Non-claims

- No STATUS promotion
- No natural power-on Day 2 farm-clear
- No D2 movie / `--video`
- 5 stumps remain on `Y1_D2_Wood_Progress`
- 34 boulders remain (nw/ne/sw); SE 13 are gone only if you re-run
  After_Stones with spa (no end pin this sitting)
- Wood_Progress ditch-lip spa is unit-fixed, not live-green from that pin
