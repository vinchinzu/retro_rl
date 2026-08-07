# Parallel pure wave 2 — results (2026-08-06)

**Status: complete for L3/L5/L6 first pure segments; L8 candle residual open.**

## Closed

| Bead | Result | Track | Checkpoint | Module |
|------|--------|-------|------------|--------|
| `rr-g4p` | 0x7c → **0x7b** west key (6× Zol `0x13`) | Clean 2/2 | `Level3WestKey` | `level3_dungeon.py` |
| `rr-vqw` | 0x76 → **0x66** clear (3× `0x30`); east→0x67 | Clean 2/2 | `Level5Cleared66` | `level5_dungeon.py` |
| `rr-r9l` | 0x79 → **0x7a** key (5× `0x24`) | assisted 2/2 | `Level6EastKey` | `level6_dungeon.py` |

## Open / partial

| Bead | Status | Notes |
|------|--------|-------|
| `rr-q8a` | open | Bush `0x6D` live; need shop + 60R + burn → `Level8Entrance` |

## Ownership (kept)

Per-dungeon modules only (`levelN_dungeon.py`, runners, LEVELN_ROUTE). Did **not**
edit L2 tip modules / `dungeon.py` L2 specs / STATUS.

## Next wave seeds (ready)

| Bead | Next action | Start |
|------|-------------|-------|
| `rr-65w` | L3 north/key chain → Raft | `Level3WestKey` |
| `rr-87a` | L5 0x67 + dark rooms | `Level5Cleared66` |
| `rr-076` | L5 entry east Pols Voice | `Level5Entrance` |
| `rr-miy` | L6 post-key graph → Rod | `Level6EastKey` |
| `rr-ccx` / `rr-q8a` | Candle shop + burn 0x6D enter | OW / `Level8BushOW` |

Blocked until parent pure: `rr-vpl` (L3 TF), `rr-28p` (L5 TF), `rr-d6v` (L6 TF).
