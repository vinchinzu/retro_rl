# Architecture — TMNT IV

Language: [CONTEXT.md](../CONTEXT.md). Commands: [AGENTS.md](../AGENTS.md).

One production tick (`Stage1Policy`) for every stage. New behavior is a
`next(state)` tactic or a spec row, not a second policy and not a copied
script.

## Layout

| Path | Owns |
|------|------|
| `ram.py` `stages.py` `menus.py` `paths.py` | Observation, boot, files |
| `assist.py` | Emergency HP + form-2 iframe hold |
| `policy.py` | Dispatcher (`tick`); pizza/hazard import seam |
| `tactics/` | Stage- and boss-specific `next(state)` |
| `grind_knobs.py` | Overlay knobs (Alleycat/Sewer bands are not grindable) |
| `run/` | Wave-chain segment, stage bridge, Clean suite |
| `lab/` | Slash research adapters (KEEP ≠ production) |
| `local_grind/` | Knob-search agent |
| `scripts/` | Thin CLIs over `run/` / `lab/` / continuous recorder |

## Tick order

pizza → pack → spikes → Baxter → Technodrome → cave → Slash → form-2 →
combat tree → stall escape. `HazardAvoid` is not in that order.

## Overlap rules

- One table, one loop: `StageSpec`, `CleanProbeSpec`, `BridgeSpec`.
- Slash production is `tactics/slash.py` (spin dodge adx **52**). Lab
  patterns stay in `lab/` until a stage suite and a continuous dry-run
  both hold.
- Add a spec; do not clone a probe or segment CLI.

## Tests

ROM-free tests cover **finish**, **time**, and **damage** — not file
layout. Spec tables (`StageSpec`, `CleanProbeSpec`, `BridgeSpec`) are
the loop contract. Do not add tests that require a cloned CLI or PNG
dump. Hygiene only forbids the old clone names
(`run_stageN_segment.py`, `probe_stageN_clean.py`,
`run_stageN_bridge.py`).

## Artifacts

JSON + mp4 under `recordings/` (gitignored). PNG dumps are opt-in
(`--screenshots` on segment/bridge). Exceptions: grind-trial frames,
boot probe, continuous freeze-abort.
