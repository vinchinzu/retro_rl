# Agent Instructions — smb

NES Super Mario Bros. (**M8** Clean power-on → 8-4 ending). Shared:
`retro_harness.platformer`. Docs: `docs/STATUS.md`, `docs/plan.md`,
`docs/HYGIENE.md`. Physics search: `.grok/skills/smb-physics-search/`.
Tracker: `bd ready -l smb`.

## Immediate goal

Bead `rr-g2ht` (32-exit warpless #3728M from 2-2). Recipe:
[`docs/HANDOFF_32EXIT.md`](docs/HANDOFF_32EXIT.md). Isolated 1-3 pits:
`rr-tb15`. Do not touch the warp any% seed.

## Commands

```bash
uv run python smb/scripts/setup_rom.py
uv run pytest nes/smb/tests -q

# M8 Clean power-on → ending
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.run_warp_finish --mode poweron --trials 3

# 32-exit (warpless #3728M only — not HappyLee warps #1715M)
uv run python -m smb.tas.fetch_refs
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.annotate_fm2 --search 2-2 --from-pred --export
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.record_warpless --to 2-1
./play smb --list
```

Parked polish / TAS / oracle CLIs: `docs/plan.md` § CLI catalog.

## Layout

| Path | Role |
|------|------|
| `ram.py`, `obs.py`, `observation.py` | RAM snap / neuro vec / physics lattice |
| `policy.py` | Replay seeds + 1-1 play helper (Composer inputs) |
| `tas/stages.py` | **Composer**: `StageSpec` rows (TAS adapt) |
| `reactive_12.py` / `reactive_route.py` | Warp 1-2 + route tracker |
| `flag_12.py` | 32-exit 1-2 flag body (not W4) |
| `rta_panel.py` | HUD capture (`VideoWriter`) |
| `scripts/` | Thin CLIs — env/report only |

Soft max ~1000 LOC: merge into `StageSpec` / `policy` or delete
([CODING_STANDARDS.md](../../CODING_STANDARDS.md)).

## Traps

- Power-on: **350** boot + **16** idle. Level1_1 continuous: **14** idle.
  Natural 1-1 alone: **1** idle (`NATURAL_SETTLE_FRAMES`).
- World 4 = world index **3**. 32-exit clock is `$075C` LevelNumber — never
  default `_smb_level` AreaNumber (`$0760`).
- Ending = 8-4 + `oper_mode=2` held **120** idle. Recordings hold **780f**
  through Peach (`ENDING_PEACH_HOLD_FRAMES`).
- Do not absolute-stitch a faster 1-1 into old 1-2. Retime from control.
  Warpless is **#3728M**, not warps #1715M. Pin boot: `set_state` →
  `reset()` → `set_state`. RAM y is head/top (floor stand y=176).

## Pointers

[docs/STATUS.md](docs/STATUS.md) · [docs/plan.md](docs/plan.md) ·
[docs/HYGIENE.md](docs/HYGIENE.md) · [docs/TAS_ADAPT.md](docs/TAS_ADAPT.md) ·
[docs/HANDOFF_32EXIT.md](docs/HANDOFF_32EXIT.md)
