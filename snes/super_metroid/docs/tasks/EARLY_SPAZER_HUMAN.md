# Early Spazer — human wall-jump recording (one page)

**Goal:** demo pure path for K2.2 Spazer mainline (wall-jump climb) from the
continuous Charge Below Spazer source. Spazer is **mainline** — continuous
Below→West always detours for Spazer (no floor skip). Floor→top climb + top→mid
drop remain residual; continuous warehouse is RED until those pure-green.

Default continuous tip is still `--to bat_cave` (past Warehouse).

## Run (human record — **no guide**)

Guide overlay messes play/camera for this room. Always pass `--no-guide`.

**Preferred start: Charge (Big Pink)** — not Bat, not Below Spazer scratch
(those pins are ahead or often corrupt). Play natural: Charge collect → Red →
Bat → Below Spazer climb → Super door → Spazer → return.

```bash
# From Charge — play the whole path into Spazer climb (recommended)
uv run python snes/super_metroid/scripts/record/guided_human.py \
  --from charge-to-spazer --name spazer_from_charge_human --no-guide

# Same start (alias)
uv run python snes/super_metroid/scripts/record/guided_human.py \
  --from big-pink --route early-spazer --name spazer_from_charge_human --no-guide

# Avoid unless you know the pin is good:
# --from below-spazer   (scratch continuous pin — often unplayable)
# --from bat            (not a preset; Bat is past Charge and still before climb)
```

| Control | Action |
|---------|--------|
| **F5 / F1** | Save task JSON + end state, exit |
| **ESC / Q** | Cancel without saving |
| `[` `]` / TAB | Speed / turbo |
| HUD | room, xy, nearest guide waypoint |

Assist (unlimited energy/ammo) is **on** by default (practice only). Use
`--no-assist` for hard practice. `--no-guide` hides the polyline.

## Source (Charge on continuous spine — 2026-08-04)

| Field | Value |
|-------|-------|
| Continuous checkpoint | `scratch/post_below_spazer_with_charge_continuous.state` |
| Pre green-door | `scratch/pre_spazer_door_with_charge.state` (~460,139) |
| Room | `0xA408` Below Spazer |
| Pin (continuous) | ~`(49, 395)` pose 1, game state 8 |
| Items | Morph + Bombs (`0x1004`) |
| Beams | **`0x1000` Charge** (main-line K1 detour via `play_big_pink_to_ghz`) |
| Ammo | missiles 10, **5 supers** (green door) |
| Continuous frames | **84,880f** integrity-green `--to below_spazer` |
| Pure door hop | `below-spazer-to-spazer` GREEN from pre-door → `0xA447` |
| Pure collect | `spazer-collect` GREEN → beams `0x1004` |
| Pure return | `spazer-return-to-below` GREEN → Below top handoff ~(380,155), clear of Super door |
| Residual | floor→top WJ climb; top→floor→West (do not RIGHT from handoff — re-enters Spazer) |

### Legacy no-Charge pin (still valid for power-only practice)

| Field | Value |
|-------|-------|
| Path | `scratch/post_below_spazer_for_spazer_pure.state` |
| Beams | `0x0000` — shoot **X** power only |
| Provenance | pure `bat-to-below` from `continuous_like_bat` (668f) |

Refresh source:

```bash
uv run python snes/super_metroid/scripts/probe/kpdr.py pure bat-to-below \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/continuous_like_bat.state \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_below_spazer_for_spazer_pure.state
```

### Loadout / “door looks wrong” notes

| Symptom | Cause | What to do |
|---------|--------|------------|
| **No Charge beam** | Expected — Charge is a K1 Big Pink detour, not on this KPDR pin | Shoot with **X** (power). Charge optional practice only if you re-source after Charge collect |
| Left door looks open / sucks you into Bat | Left door is Bat Room (`door_ptr 0x9102`). WJ on the door face triggers transition; human demo hit `x=65535` wrap + Bat ping-pong | Keep **x ≥ 40** on the left wall; short RIGHT bump if door_tr=1 |
| Bottom water | Going right off the dry lip drops into water/spikes | Climb is **up the left shaft**, not across the pool |
| West blue door | Bottom-right `(~496, 368)` | any% skip — **trap** for this demo |

## Technique (probe-backed)

Probes in `debug/early_spazer_*` (2026-08-04):

1. **Shoot** a few power shots (X) from the lip — clear Cacatac if needed.
2. **Floor → mid (~y 260–300):** spin-jump LEFT+A into the left shaft
   (human peak pure y≈**284**; auto hop ~339). Hard part; stay off the door.
3. **Mid → top (double WJ):** once above ~y 260, consecutive wall-jumps work.
   Probe: place/mid start y=260 → **min_y ≈ 125**, land top-left ledge ~`(59, 126)`.
   Pattern: into wall → A pulse → flip A (shared `WallJumpTiming` / Bubble double).
4. **Top left → green Super:** bomb the top gap (sm-json 4→3 needs bombs), run
   right, SELECT to supers, X the **top-right green door**, enter Spazer Room.
5. Collect Chozo → return left to Below Spazer.

## Guide path (overlay = validate while recording)

### Below Spazer `0xA408` (2×2 screens)

| Waypoint | ~xy | Notes |
|----------|-----|-------|
| entry | 49, 395 | start / left door lip — **do not hug door** |
| off-door | 55–70, 395 | small RIGHT first if needed |
| spin-peak | 37–45, 280–340 | floor→mid spin (human ~284) |
| wj-mid | 45–60, 260→180 | double WJ zone (probe-proven) |
| top-left | 59–110, 120–130 | node 4 ledge |
| bomb-gap | 200, 120 | morph bombs |
| green-door | 480, 120 | Super → Spazer (`block [31,7]`) |

**Traps:** left door → Bat; bottom-right blue → West Tunnel.

### Spazer Room `0xA447` (1 screen)

| Waypoint | ~xy | Notes |
|----------|-----|-------|
| entry | 40, 121 | just inside left door |
| chozo | 176, 144 | Spazer pedestal (`block [11,9]`) |
| exit-left | 40, 121 | return to Below Spazer top right |

## Success (human task)

1. Leave `0xA408` via **top-right green** door (not Bat left, not West bottom-right).
2. Collect Spazer (`beams` bit `0x04`).
3. Return to `0xA408` ordinary gameplay.
4. **F5** → `tasks/spazer_human.json` + `spazer_human_end.state`.

Prior `spazer_human` (3596f) stayed in Bat↔Below door loop — not a clear; re-record.

## Clean TOP-MID recording (prefer this over full re-climb)

Door / collect / return are pure-green. **Do not** re-record Charge→climb just
to capture a drop — start from the pure handoff pin.

```bash
# Refresh pure post-Spazer states (no human)
uv run python snes/super_metroid/scripts/probe/kpdr.py pure spazer-collect \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_spazer_entry_pure.state \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_spazer_collect_pure.state
uv run python snes/super_metroid/scripts/probe/kpdr.py pure spazer-return-to-below \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_spazer_collect_pure.state \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_spazer_return_pure.state

# Human: clean drop only from return handoff ~(380,155)
uv run python snes/super_metroid/scripts/record/guided_human.py \
  --from post-spazer-return --route spazer-top-drop --name spazer_top_drop_human --no-guide
```

| Control | Action |
|---------|--------|
| Goal | Mid band y≥220 (or floor) without enemy catch / thrash |
| Trap | **RIGHT** re-enters open Super door into Spazer |
| F5 | Save `tasks/spazer_top_drop_human.json` + end state |

## Next pure / tip splice

Door hop + collect + return are pure green. **Spazer is mainline K2.2** —
`play_below_spazer_to_west` always runs `play_spazer_detour` (no floor skip):

| Piece | Status |
|-------|--------|
| `play_below_spazer_to_west` | **Mainline** → always `play_spazer_detour` |
| `play_spazer_detour` | Climb → door → collect → return → West |
| Spazer held + top handoff | `play_spazer_top_to_west` (no RIGHT into Super) |
| Mid → West | pure-green |
| Return Spazer → Below top | pure-green ~(380,155) |
| Floor → solid top climb | **GREEN** (401f; FLOOR-MID + CLIMB-LAND) |
| Top handoff → West | **GREEN** (1281f ×2; `spazer_top_drop_human` guide) |
| Solid top → Super door | **mainline** morph-tunnel in `play_below_spazer_to_spazer` |
| Full floor detour | **GREEN** `spazer-detour` 3675f ×2 → West beams `0x1004` |

**Product continuous** is warehouse **with Spazer bit** (`0x1004`). Historical
`warehouse_with_charge` (beams `0x1000` only) is not the product path.

```bash
# Pure mainline detour from floor (RED until climb land green)
uv run python snes/super_metroid/scripts/probe/kpdr.py pure spazer-detour \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_below_spazer_with_charge_continuous.state
```

Historical climb residual purged (Spazer mainline promoted). Product:
`routes/kpdr/spazer/`. Board: `docs/plan.md` / `docs/STATUS.md`.

```bash
# Return pure (green)
uv run python snes/super_metroid/scripts/probe/kpdr.py pure spazer-return-to-below \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_spazer_collect_pure.state \
  --output snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_spazer_return_pure.state

# Door hop pure (green; top ledge source)
uv run python snes/super_metroid/scripts/probe/kpdr.py pure below-spazer-to-spazer \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/pre_spazer_door_with_charge.state

# Climb (still residual from floor)
uv run python snes/super_metroid/scripts/probe/kpdr.py pure below-spazer-climb \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_below_spazer_with_charge_continuous.state
```

## Refs

| Asset | Role |
|-------|------|
| `docs/tasks/refs/early_spazer_red_room.png` | left-shaft mid climb visual |
| `debug/early_spazer_full_try/` | probe shots (mid WJ top, ledge) |
| sm-json `Below Spazer.json` | link 1→4 `canWallJump`; 4→3 bombs |
| `docs/routes/TRACK_100.md` | 100% board / Spazer insert policy |
| `docs/SOURCE_STATES.md` | `post_below_spazer_for_spazer_pure` |
| `routes/kpdr/guides/` | `early-spazer` presets |
| `routes/kpdr/spazer/` | pure door/collect/return + climb/drop (package) |
