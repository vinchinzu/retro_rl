# SMB — architecture hygiene

Rules that keep warp / TAS / 32-exit tooling from growing clone CLIs.
Program: [CODING_STANDARDS.md](../../../CODING_STANDARDS.md),
[REPO_HYGIENE.md](../../../docs/REPO_HYGIENE.md).

## Layers

| Layer | Module(s) | Owns |
|-------|-----------|------|
| RAM | `ram.py` | Snapshots, addresses, ending / world-4 gates |
| Neuro obs | `obs.py` | 210-dim PPO vector (legacy 189) |
| Physics lattice | `observation.py`, `approx.py`, `predict.py`, `residual.py` | `R(τ)`; search in `approx`, grade with `predict` |
| Composer | `tas/stages.py` | One `StageSpec` row per TAS body; `slice` / `chain` / `replay` consume it |
| Warp replay | `policy.py` | NES-9 RLE seeds, settle/boot/ending constants, `play_1_1_until_clear` |
| Warp 1-2 | `reactive_12.py`, `reactive_route.py` | State-gated W4 + successor tracker |
| 32-exit 1-2 flag | `flag_12.py` | Lift/pipe tail + UG floor-pipe truth table |
| Menus / boot | `menus.py` | Title script, `boot_to_ready`, `idle_n` |
| Capture | `rta_panel.py` | RTA panel + `VideoWriter` (not a per-CLI ffmpeg) |
| TAS import | `tas/fm2.py`, `tas/bk2.py`, `tas/warpless.py` | Movies; warpless legs table |
| Local search | `tas/search.py` | 1-1 Evaluator polish + 8-3 jump mutations |
| Scripts | `scripts/` | Thin CLIs — env / assist / report only |

`obs.py` (neuro vector) and `observation.py` (residual lattice) are
different layers. Do not merge them.

## Hard rules

1. **No new per-stage CLI.** Add a `StageSpec` / `WARPLESS_LEGS` row.
   `annotate_fm2 --search <id>` already drives warpless extract.
2. **No path logic in `scripts/`.** Controllers live in `flag_12`,
   `reactive_*`, `tas/*`, `policy`. Scripts call those and write JSON/MP4.
3. **Scripts do not import scripts.** Shared boot/idle/1-1/video live in
   `menus.py`, `policy.py`, `rta_panel.py`.
4. **Probe CLIs are not product.** A solved hop's probe is deleted (git
   restores). Durable CLIs: `run_warp_finish`, `run_1_2_flag`,
   `record_warpless`, `annotate_fm2`, `./play smb`.
5. **Prefer TAS adapt over hill-climb** (`docs/TAS_ADAPT.md`). Local
   mutations live in `tas.search`. Probe / polish / import_fm2 /
   stitchless CLIs are deleted (git restores).
6. Soft max **~1000 LOC**. Crossing 1k means merge into the Composer /
   owner or delete. No sibling extract (`foo_2.py`).

## Tracks (do not mix seeds)

| Track | Seed stem | Notes |
|-------|-----------|-------|
| Clean warp M8 | `smb_1_1_to_ending_natural_82.json` | STATUS; never overwrite |
| Hybrid TAS | `smb_happylee_hybrid_v2_fx84.json` | HL…8-2 + nat 8-3 + flamexx 8-4 |
| Stitchless 8-3 | `smb_8_3_stitchless_skills_leave.json` | Skills leave; not pure FM2 |
| Pure HappyLee | `models/pure_hl/` only | Parked |
| 32-exit warpless | `smb_*_warpless_*.json` | #3728M only |

## Artifact retention

- STATUS evidence JSON: keep under `recordings/warp_finish/`,
  `recordings/reactive_warp/`, named warpless play JSON.
- Lab probes, PNG dumps, one-off agent logs: gitignore; do not commit
  bulk PNGs. `recordings/` is local evidence, not product source.
- `**/probe*.png` is already gitignored.

## Over-bar leftovers (next Gut)

None over 1k. Showcase stitch render uses `rta_panel.VideoWriter` (not a
per-CLI ffmpeg). Next compression: `run_warp_finish.py` (~873) if it
grows a second encode path, or fold `tas/slice.py` per-stage probe
wrappers onto `probe_from_control`.
