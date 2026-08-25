# Agent Instructions — Super Metroid

Scripted full-clear package `super_metroid` (disk: `snes/super_metroid/`).
Docs: `CONTEXT.md`, `docs/STATUS.md`, `docs/plan.md`,
`docs/ASSIST_CONTRACT.md`, `docs/ram_map.md`. Session loop:
`.grok/skills/sm-session/SKILL.md`.
Tracker: `bd ready -l super_metroid -l spine`.

## Evaluation contract

- Primary: unlimited energy + ammo only
  ([`docs/ASSIST_CONTRACT.md`](docs/ASSIST_CONTRACT.md)); no free
  items/doors/map/bosses/capacity.
- Natural ending/credits required; final boss alone is not a clear.
- Clean track: [`docs/CLEAN_TRACK.md`](docs/CLEAN_TRACK.md);
  `*_clean` stems only — never overwrite assisted baselines.
- Dual-track: spine continuous vs room practice. Practice greens are
  not continuous evidence. Planner owns STATUS.

## Immediate goal

Living tip: `--to phantoon` ([STATUS.md](docs/STATUS.md),
[CONTEXT.md](CONTEXT.md)). Next spine bead: `rr-kw8t` Gravity on that tip
(power-on). Default CLI is `phantoon`. Living residual:
[`docs/tasks/rr-kw8t-residual.md`](docs/tasks/rr-kw8t-residual.md).

## Commands

```bash
# Watch any hop (bot on). --headed is retro_harness.headed, not a per-probe loop.
uv run python snes/super_metroid/scripts/probe/kpdr.py pure <hop> --source <pin> --headed
./snes/super_metroid/play <pin> --headed --assist-full

bd ready -l super_metroid -l spine

uv run python snes/super_metroid/scripts/record/continuous.py --to phantoon --no-video

uv run python snes/super_metroid/scripts/probe/kpdr.py pure <hop> \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/<pin>.state

./snes/super_metroid/play
uv run python snes/super_metroid/scripts/tools/replay_human_hop.py \
  snes/super_metroid/tasks/full_start_v1.json --hop 0 --dual
```

`--no-video` on continuous/probe duals. Leave proof is RAM + dual JSON
(`super_metroid.hop_glance`), not an MP4. Quote frames, room, beams,
integrity flags — do not paste report JSON. Shine / TAS / practice CLIs:
`docs/plan.md`.

## Layout

| Path | Role |
|------|------|
| `routes/continuous.py`, `early_continuous.py`, `catalog.py` | Power-on chain + tip registry |
| `routes/kpdr/` | Pure movement/combat controllers |
| `routes/kpdr/spazer/` | Gold-standard multi-hop package |
| `tas/` | Sniq movies + harness replay (`docs/TAS_ADAPT.md`) |
| `custom_integrations/SuperMetroid-Snes/` | Anchors; probes → `scratch/` |

Multi-hop → package from day 1. Split a file **before 500 lines**; refuse
a new knob on a file **≥800**. Continuous hops only via `tips.play_hops`.

## Traps

- Door-warp settle: wait for **game state 8**; state 11 can last 50–100+f.
- High WRAM (`$7E:D820+`): `read_bank7e_wram` / `write_wram_u8` — raw
  `get_ram()[0xD820]` is open-bus garbage.
- Named anchors in `SuperMetroid-Snes/`; probe noise only in `scratch/`.
  Overwrite `scratch/<hop>_dual.json` — do not mint `_vN` / `_window_*`.
- Dual-track / door-warp / boss probes are **not** continuous evidence.
- Morph bombs are **X** while morph (not A).
- **D-pad vs shoulders:** `LEFT`/`RIGHT` walk; `L`/`R` are shoulders.
  Never use `L` as a hop side.
- **RED dual does not trash the hop.** Overwrite `scratch/<hop>_dual.json`
  only. Do not revert the controller because it missed the pass gate.
- Practice doorway bootstrap **zeros** momentum — not natural leave-speed.
- **Shinespark store:** harness **B**=dash, **A**=activate, **DOWN**=store.
  After echoes=4, press DOWN **while still holding RIGHT**. Idle or **B
  alone** dumps echoes **4→0 in one frame**.
- **Ceres elev:** Falling→elev mid-transition can still read **y≈139**;
  ordinary **gs=8 remaps to bottom y≈651**. Product stays `_BOOT_STYLE =
  "legacy"` until TAS ≤ the published morph tip (STATUS).
- **Spazer + HJ pillar:** single-frame peak down-shot fails under Spazer;
  `play_hj_room_collect` multi-taps DOWN+X at peak.
