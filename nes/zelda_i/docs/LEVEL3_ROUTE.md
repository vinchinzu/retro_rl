# Level 3 — Manji (route notes)

Status: **assisted-entry** (not Clean STATUS)

Assist track only (`UnlimitedHealthAssist` / `--infinite-life`). Do not promote
natural-entry or Clean gates from this doc.

## Overworld

| Field | Value | Evidence |
|-------|-------|----------|
| Door screen | **`0x74`** | **live** — exit spawn from entry; re-enter level==3 |
| Door approach | UP @ **x≈128**, approach from **y≳130** | live exit spawn (128, 125) |
| Entry room | **`0x7c`** | **live** `level==3` mode 5 |
| Checkpoint | `Level3Entrance.state` | `custom_integrations/LegendOfZelda-Nes/` |

### Path from start / post-sword

**Source** (Zelda Dungeon — *not* walkable as stated):

```
From start: up, left 4, down, right 1
→ screens 0x77 → 0x67 → 0x66 → 0x65 → 0x64 → 0x63 → 0x73 → 0x74
```

**Live blocker:** screen **0x67** is a fully enclosed tree pocket (enterable
from 0x77 north) with **no west exit**. Source hop path fails after the first
screen.

**Live prefix** (assisted; pieces verified 2026-08-06):

```
0x77 E@y≈140 → 0x78 N@x≈48 → 0x68 N → 0x58 W@y≈155
→ 0x57 W → 0x56 W@y≈133 → 0x55 S → 0x65
```

**Live door suffix** (from `OW_66` or 0x65; assisted enter succeeded):

```
0x66/0x65 W@y≈141 → 0x64 → 0x63 S → 0x73 E@y≈117 → 0x74
→ tour/UP hunt → level 3 room 0x7c
```

Code: `zelda_i.level3_overworld` (`LEVEL3_PATH_HOPS`, `LEVEL3_DOOR_HOPS_FROM_66`).

Required items to *enter*: wooden sword (potion recommended by walkthrough;
not required for assisted entry).

## Interior (source → live)

| Room id | Enemies | Key/item | Doors | Status |
|---------|---------|----------|-------|--------|
| **0x7c** entry | Keese (visual pack; type RAM often 0 / type-liveness residual) | — | S mouth (exit OW), **W open** (visual) | **live** entry + screenshot |
| West of entry | Zols + key (source) | key | — | source only |
| North chain | Zols + key → Darknuts… | bombs / Raft path | — | source only |

### Source interior (Zelda Dungeon L3)

- Entry → **LEFT** → Zols (split to Gel with wooden sword) + key → UP → Zols + key → UP
- Darknuts (side/back hits); bombs reward; bomb RIGHT = boss shortcut
- LEFT → Keese + **Compass** → key LEFT → Darknuts → DOWN
- Staircase → Keese path → **Raft**
- Backtrack toward boss: Bubbles, Keese, Zols → UP **Manhandla** (bombs best)
- Heart Container → Triforce shard 3

### Live residual

- West door from 0x7c is **visually open** but automated push often stops at
  x≈32 (`open_doorway_mask==0`). Isolated pure clear of adjacent rooms not yet
  logged. Prefer live re-enter + combat policy before claiming room IDs east of
  entry.

## Boss / Triforce

| Field | Value |
|-------|-------|
| Boss | Manhandla (bombs preferred) |
| Item | Raft (`ADDR_RAFT=0x0660`) |
| Triforce bit | **`0x04`** |

## Checkpoints

| State | Provenance |
|-------|------------|
| `Level3Entrance.state` | Assisted enter 2026-08-06; `level==3` room **0x7c** ~(120, 205); Survival health poke only |

## Evidence

- `recordings/l3_recon.json` — door/entry facts, path notes
- `recordings/l3_entrance_live.png` / `l3_final_entry.png` / `l3_confirm_state.png` — entry room
- `recordings/l3e_74_x115_y133.png` — exterior mouth geometry (green statues)
- Probe: `uv run python nes/zelda_i/scripts/probe_level3_entry.py --infinite-life --from-state OW_66 --save-state`
- Map-only: `… --from-state Level3Entrance --map-only --infinite-life`

## Sources

- [Zelda Dungeon — Level 3: The Manji](https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/level-3-the-manji/)
- Local: `docs/research/DUNGEON_WALKTHROUGHS.md`, `docs/tasks/PARALLEL_RECON.md`
