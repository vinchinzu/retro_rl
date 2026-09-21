# Residual — isolated F1 room_engine edges (`rr-ccxt.4`)

Planner owns `docs/STATUS.md`. This sitting is **state-load / isolated** only.
Did not STATUS-promote. Did not treat a save-state pin as power-on. Did not
write `$F3CC` / sword / keys. Did not copy approach xy into Python. Did not
overwrite `recordings/verified_tip_run.json`. Did not promote
`room_50_east_to_0x01` off `natural_entry` (and did not add 0x01/0x52 hops).

Maps already had measured approach + landing; landings were **not** rewritten.
Door-push xy matched the JSON landings; `settle_destination` then walks a
band further (quoted below). Geometry authority stays `maps/room_{50,01,52}.json`.

## Commands

```bash
uv run python alttp/scripts/room_engine.py show room_50
uv run python alttp/scripts/room_engine.py show room_01
uv run python alttp/scripts/room_engine.py show room_52
SDL_VIDEODRIVER=dummy uv run python alttp/scripts/room_engine.py run room_50 \
  --edge east_to_0x01 --state CastleRoom50
SDL_VIDEODRIVER=dummy uv run python alttp/scripts/room_engine.py run room_01 \
  --edge east_to_0x52 --state CastleRoom01
SDL_VIDEODRIVER=dummy uv run python alttp/scripts/room_engine.py run room_52 \
  --edge south_to_0x62 --state CastleRoom52
uv run pytest alttp/tests/test_room_engine_edges.py -q
```

JSON/PNG from this sitting: `recordings/probe_room_engine/`.

## Isolated results (source=`state_load_dev`)

| Edge | State | ok | frames | dest room | door-push xy | settled xy |
|------|-------|----|--------|-----------|--------------|------------|
| `room_50` `east_to_0x01` | `CastleRoom50` | **True** | 1301 | `0x01` | (499, 120) | (560, 120) |
| `room_01` `east_to_0x52` | `CastleRoom01` | **True** | 130 | `0x52` | (1011, 2680) | (1072, 2680) |
| `room_52` `south_to_0x62` | `CastleRoom52` | **True** | 2327 | `0x62` | (1144, 3060) | (1144, 3128) |

`room_01` east is cheap (clear 0f + 30f RIGHT push). `room_52` south is **not**
cheap under default clear: spawn is south of hostiles, so clear walks north
(~1985f) before the 246f DOWN push. `--no-clear` from this pin would be the
short south door (~map note 34f). Isolated dest still reached.

### room_50 east_to_0x01 — RAM leftover

Start (`CastleRoom50`): room `0x50`, module `$10=0x07`, submodule `$11=0x00`,
xy=(448, 2680), sword `$F359=1`, follower `$F3CC=0`, keys `$F36F=0`, control.

Clear: 1108f, `before=4 defeated=(0, 1, 10) remaining_near=0` at (389, 2713).

Exit: 93f RIGHT push → room `0x01` xy=(499, 120).

Final (after 100f `settle_destination`): **ok=True**, frames=1301, room
`0x01`, module `0x07`, submodule `0x00`, xy=(560, 120), indoors, sword=1,
`$F3CC=0`, keys=0, `has_control=true`. Acceptance `at_door_dest=true`.

### room_01 east_to_0x52 — RAM leftover

Start (`CastleRoom01`): room `0x01`, module `0x07`, submodule `0x00`,
xy=(960, 120), sword=1, `$F3CC=0`, keys=0.

Clear: 0f, no hostiles.

Exit: 30f RIGHT push → room `0x52` xy=(1011, 2680).

Final: **ok=True**, frames=130, room `0x52`, module `0x07`, submodule `0x00`,
xy=(1072, 2680), sword=1, `$F3CC=0`, keys=0, `has_control=true`.

### room_52 south_to_0x62 — RAM leftover

Start (`CastleRoom52`): room `0x52`, module `0x07`, submodule `0x00`,
xy=(1144, 2993), sword=1, `$F3CC=0`, keys=0.

Clear: 1985f, `before=3 defeated=(1, 2) remaining_near=0` at (1128, 2701).

Exit: 246f DOWN push → room `0x62` xy=(1144, 3060).

Final: **ok=True**, frames=2327, room `0x62`, module `0x07`, submodule `0x00`,
xy=(1144, 3128), sword=1, `$F3CC=0`, keys=0, `has_control=true`.

## Graph / tests leftover

- `room_50_east_to_0x01` remains `natural_entry` (not continuous). Isolated
  re-run from `CastleRoom50` is not a clean-chain claim.
- Map doors `east_to_0x52` / `south_to_0x62` are **not** graph hops.
  `room_01_to_zelda_cell` stays `planned`.
- Offline tests: `tests/test_room_engine_edges.py` (maps load, door labels,
  dest helper, already-at-dest `run_room_edge`, graph not promoted). Opt-in
  `@pytest.mark.rom` live pins skip when ROM/state missing.

## Next

B1 stairs after the `0x01` chain still open. Dense scan already said no
stairs in 0x01/0x52/0x62; drive `f1-stairs` / `CastleB2Landing` reverse, not
another isolated F1 door replay. Natural 0x01 predecessor is the settled
(560, 120) band, not the `CastleRoom01` spawn (960, 120).
