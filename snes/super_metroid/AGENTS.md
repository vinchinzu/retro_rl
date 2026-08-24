# Agent Instructions — Super Metroid

Scripted full-clear package `super_metroid` (disk: `snes/super_metroid/`).
Docs: `docs/STATUS.md`, `docs/plan.md`, `docs/ASSIST_CONTRACT.md`,
`docs/ram_map.md`. Tracker: `bd ready -l super_metroid`.

## Evaluation contract

- Primary: unlimited energy + ammo only
  ([`docs/ASSIST_CONTRACT.md`](docs/ASSIST_CONTRACT.md)); no free
  items/doors/map/bosses/capacity.
- Natural ending/credits required; final boss alone is not a clear.
- Clean track: [`docs/CLEAN_TRACK.md`](docs/CLEAN_TRACK.md);
  `*_clean` stems only — never overwrite assisted baselines.
- Pure-first: one hop or one residual change per session; dual-track
  (spine continuous vs room practice). Practice greens are not
  continuous evidence. Planner owns STATUS.

## Immediate goal

**Verified tip:** continuous power-on → Ice Beam (default `ice`).
STATUS dual **148,167f** ×2 (2026-08-10). Ceres-successor reverify dual
**146,937f** ×2 (2026-08-22, scratch `ice_ceres_successor.json` + `_dual.json`;
STATUS promote `rr-ucl9`). `--to moat` power-on dual is scratch-green
**175526f** ×2 `0x93FE` `(49,1163)` p1 max PB 5 (rr-2r06; Ice prefix
**146937f**; post-Ice **28589f**). `--to ws` is wired (rr-p2bw): power-on
scratch dual **176141f** ×2 `0xCA08` `(57,139)` p1 gs=8 items `0x3105`
beams `0x1007` max PB 5 (spark hop **615f** ×2 from the Moat leave). Do
**not** STATUS-promote — default CLI stays `ice`. Entrance→Main is scratch
dual **403f** ×2 `0xCAF6` `(1063,907)` p9 gs=8 (rr-ahjo). Main Shaft → basement
scratch dual **1208f** ×2 `0xCC6F` `(657,92)` p24 gs=8 (rr-4btp). Basement →
Phantoon *room* scratch dual **718f** ×2 `0xCD13` `(39,124)` p81 gs=8 (rr-cjpp);
Phantoon fight (`rr-tlaq`) assist dual-green **20537f** ×2 HP 0 + `$D82B`
bit 0 gs=8 (residual `docs/tasks/rr-tlaq-residual.md`; pin
`scratch/post_phantoon_poweron.state`; do not STATUS-promote; not on `--to ws`).
Ice-pin compose through WS Entrance **29212f** (28597+615). Product `play_red_to_hellway` is the Ice-pin
checkpoint climb to ordinary Hellway left-door **5846f** ×2 `(39,139)` p11
(keep RIGHT until gs=8 x≤80; 163f/`(237,139)` was the Red Tower door-slot
fire). Successor `hellway_to_caterpillar` **2110f** ×2;
`caterpillar_to_alpha_pb` **1372f** ×2 (compose hop **1418f**). Mid→thin is
still the 2974f period WJ. Tape body 6199f remains fallback when the Ice+HJ
floor seat is absent. Tip history lives in STATUS — do not STATUS-promote
or rewrite the route from this file.

## Commands

```bash
bd ready -l super_metroid

# Continuous default (ice) / named tips
uv run python snes/super_metroid/scripts/record/continuous.py --no-video
uv run python snes/super_metroid/scripts/record/continuous.py --to ice --no-video
uv run python snes/super_metroid/scripts/record/continuous.py --to wave --no-video
uv run python snes/super_metroid/scripts/record/continuous.py --to speed --no-video

# Pure hop (example; source under scratch/)
uv run python snes/super_metroid/scripts/probe/kpdr.py pure bat-cave-to-speed-hall \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bat_cave_continuous.state

# Human record / hop replay (do not start a tape session unless that is the bead)
./snes/super_metroid/play
uv run python snes/super_metroid/scripts/tools/replay_human_hop.py \
  snes/super_metroid/tasks/full_start_v1.json --hop 0 --dual
```

Reports vs video: `recordings/<tip>.json` is machine integrity (on disk
only). Quote frames, room, beams, integrity flags, and the `.mp4` path
— do not paste the JSON body. Shine / TAS / practice CLIs stay in
`docs/plan.md` and `docs/tasks/SHINE_PRACTICE.md`.

## Layout

| Path | Role |
|------|------|
| `routes/continuous.py`, `early_continuous.py`, `catalog.py` | Power-on chain + tip registry |
| `routes/kpdr/` | Pure movement/combat controllers |
| `routes/kpdr/spazer/` | Gold-standard multi-hop package |
| `tas/` | Sniq movies + harness replay (`docs/TAS_ADAPT.md`) |
| `custom_integrations/SuperMetroid-Snes/` | Anchors; probes → `scratch/` |

Multi-hop → package from day 1 (no megafiles), room-prefixed geometry,
RLE as JSON data, ≤~500 lines/file.

## Traps

- Door-warp settle: wait for **game state 8**; state 11 can last 50–100+f.
- High WRAM (`$7E:D820+`): `read_bank7e_wram` / `write_wram_u8` — raw
  `get_ram()[0xD820]` is open-bus garbage.
- Named anchors in `SuperMetroid-Snes/`; probe noise only in `scratch/`.
- Dual-track / door-warp / boss probes are **not** continuous evidence.
- Morph bombs are **X** while morph (not A).
- **D-pad vs shoulders:** `LEFT`/`RIGHT` walk; `L`/`R` are shoulders.
  Never use `L` as a hop side.
- Door entry speed/position matter. Practice doorway bootstrap **zeros**
  momentum — never treat those fixtures as natural leave-speed evidence.
- **Shinespark store:** harness **B**=dash, **A**=activate, **DOWN**=store.
  After echoes=4, press DOWN **while still holding RIGHT**. Idle or **B
  alone** dumps echoes **4→0 in one frame**. Drill:
  `shine_practice.py drill`.
- **Ceres elev:** Falling→elev mid-transition can still read **y≈139**;
  ordinary **gs=8 remaps to bottom y≈651**. Ledge pin **y=571**. Product
  stays legacy boot **26,824f** until TAS ≤ that.
- **Spazer + HJ pillar:** single-frame peak down-shot fails under Spazer;
  `play_hj_room_collect` multi-taps DOWN+X at peak.
