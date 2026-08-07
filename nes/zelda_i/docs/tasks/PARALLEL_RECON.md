# Parallel recon wave — L3–L9 + OW (2026-08-06)

**Status: complete (scaffold + live entry for L3/L5/L6; L8 bush; L4/L7/L9 plan).**

**Goal:** Prep every dungeon after L2 in parallel while L2 tip agents finish
Moon. Deliver **planning docs + live OW door IDs + entry checkpoints** so
isolated pure + reactive splice can land as soon as each tip arrives.

## Strategy

Zelda I first quest allows **out-of-order dungeons** (L9 needs all TF bits).
So we can recon L3/L5/L6/L8 live without finishing L2. Gated dungeons (L4
raft, L7 whistle, L9 full TF) get planning docs now; live entry after item
or **dev-only** inventory poke (never Clean STATUS).

Dual track: default Clean; `--infinite-life` Survival assist for first-pass
geometry only (`docs/ASSIST_CONTRACT.md`).

## Beads (this wave)

| Bead | Scope | Live? |
|------|-------|-------|
| `rr-2mi` | L3 Manji OW + entry | Yes (any order) |
| `rr-k0w` | L4 Snake raft island | Plan (+ raft gate RAM) |
| `rr-aec` | L5 Lizard Lost Hills | Yes |
| `rr-1yu` | L6 Dragon | Yes |
| `rr-7vc` | L7 Demon whistle pond | Plan (+ whistle gate) |
| `rr-s8o` | L8 Lion candle bush | Yes (buy candle) |
| `rr-c8v` | L9 Death Mountain | Plan (+ TF gate) |
| `rr-2nx` | OW door screen table | Aggregate |
| `rr-mmq` | NamedRoute stubs L3–L9 | Scaffold only |

## File ownership (no collisions)

| Agent | Own (create/edit) | Do **not** touch |
|-------|-------------------|------------------|
| L3 | `docs/LEVEL3_ROUTE.md`, `level3_overworld.py`, `scripts/probe_level3_entry.py`, `Level3*.state` | L2 modules, STATUS |
| L4 | `docs/LEVEL4_ROUTE.md`, `level4_overworld.py`, `scripts/probe_level4_entry.py` | same |
| L5 | `docs/LEVEL5_ROUTE.md`, `level5_overworld.py`, `scripts/probe_level5_entry.py` | same |
| L6 | `docs/LEVEL6_ROUTE.md`, `level6_overworld.py`, `scripts/probe_level6_entry.py` | same |
| L7 | `docs/LEVEL7_ROUTE.md`, `level7_overworld.py` | same |
| L8 | `docs/LEVEL8_ROUTE.md`, `level8_overworld.py`, `scripts/probe_level8_entry.py` | same |
| L9 | `docs/LEVEL9_ROUTE.md`, `level9_overworld.py` | same |
| OW table | `docs/OVERWORLD_DOORS.md` only | overworld.py hops |
| Graph | `routes_later.py` or append **only** stub section in `routes.py` if safe | chain.py, level2_* |

**Hot modules (other agents):** `level2_overworld.py`, `level2_clean_door.py`,
`chain.py`, `dungeon.py`, `docs/STATUS.md`, `assist.py`, L2 probe scripts.

## Deliverable template (each LEVELN_ROUTE.md)

```markdown
# Level N — Name (route notes)

Status: planning | assisted-entry | isolated-pure | natural-entry

## Overworld
- Door screen: 0x?? (live | source)
- Path from start / post-L1: hops...
- Required items: ...

## Interior (source → live)
| Room id | Enemies | Key/item | Doors |

## Boss / Triforce
- Boss type id, policy notes
- triforce bit 0x..

## Checkpoints
- LevelNEntrance.state provenance

## Evidence
- recordings/*.json
```

## Live probe recipe

```bash
# From Level1.state (sword not yet) or post-sword; use assist for OW combat
uv run python zelda_i/scripts/probe_levelN_entry.py --infinite-life --save-state
```

1. Boot / load known state (`Level1` or natural boot + sword).
2. Enable `UnlimitedHealthAssist`.
3. Walk hops to door screen; UP/bomb/burn as required.
4. Confirm `level==N`, mode 5, entry room id; `save_state(..., "LevelNEntrance")`.
5. Probe N/E/S/W for 200–400f each; log room ids + object types.
6. Write JSON under `recordings/lN_*_recon.json` + route doc.

## Triforce / item RAM (already in ram.py)

| Item | ADDR | L dungeon |
|------|------|-----------|
| TF bits | `0x0671` | 0x01…0x80 |
| Raft | `0x0660` | L3 |
| Ladder | `0x0663` | L4 |
| Whistle | `0x065C` | L5 |
| Rod | `0x065F` | L6 |
| Candle | `0x065B` | shop / L7 red |
| Book / Magic Key | `0x0661` / `0x0664` | L8 |
| Ring / Silver arrows | ring `0x0662` | L9 |

## Reactive splice later

When tip reaches L_n: isolated pure from `LevelNEntrance` → natural-entry from
real predecessor → NamedRoute promote. Do **not** claim Clean natural until
predecessor TF bits are real (no poke).
