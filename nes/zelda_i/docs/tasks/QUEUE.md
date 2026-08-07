# Queue — Zelda I

**Source of truth:** `bd ready -l zelda_i`

Human snapshot. Expand room beads when that dungeon is the tip (or for
parallel isolated pure from checkpoints).

## Live tip (L2 — other agents)

| Gate | Status | Notes |
|------|--------|-------|
| Power-on → L1 Triforce | **Clean green** | M5 |
| L2 interior → TF `0x02` | **active tip** | epic `rr-ci7` |
| Ready L2 leaves | `rr-ebe`, `rr-n5i`, `rr-bcd` (and related) | do not collide |

## Parallel wave results (2026-08-06)

### Wave 1 — recon (OW door + entry)

| L | Door OW | Entry room | Checkpoint | Track |
|--:|---------|------------|------------|-------|
| 3 | **0x74** | **0x7c** | `Level3Entrance` | assisted enter |
| 4 | 0x45 hyp | — | plan (needs raft) | source |
| 5 | **0x0B** (hills **0x1B**) | **0x76** | `Level5Entrance` | assisted enter |
| 6 | **0x22** | **0x79** | `Level6Entrance` | assisted enter |
| 7 | 0x42 hyp | — | plan (needs whistle) | source |
| 8 | **0x6D** bush | — | `Level8BushOW` | candle blocked |
| 9 | 0x05 hyp | — | plan (needs TF `0xFF`) | source |

Docs: `LEVEL{3–9}_ROUTE.md`, `OVERWORLD_DOORS.md`, `PARALLEL_RECON.md`.

### Wave 2 — isolated pure (closed)

| Bead | Result | Track | Checkpoint | Runner |
|------|--------|-------|------------|--------|
| **rr-g4p** L3 west key | **0x7c→0x7b** 6× Zol `0x13`, keys 0→1 | **Clean** 2/2 | `Level3WestKey` | `run_level3_west_key.py` |
| **rr-vqw** L5 clear 0x66 | 3× `0x30` Gibdo, doors east→**0x67** | **Clean** 2/2 | `Level5Cleared66` | `run_level5_clear66.py` |
| **rr-r9l** L6 east key | **0x79→0x7a** 5× `0x24`, keys 0→1 | **assisted** 2/2 | `Level6EastKey` | `run_level6_east_key.py --infinite-life` |
| **rr-q8a** L8 candle enter | PARTIAL — bush only | — | `Level8BushOW` | candle residual |

Door traps burned:

- L3 west: pure LEFT sticks x≈32 → use **LEFT+UP** @ y≈149
- L6 east: y≈157 → x≈208 → y≈144–149 RIGHT (no A while aligning)

## Next beads (spawned after wave 2)

### Parallel pure (from green checkpoints — no L2 wait)

| Bead | Title | Start state |
|------|-------|-------------|
| **rr-65w** Z3.2 | North/key chain toward Raft | `Level3WestKey` |
| **rr-vpl** Z3.3 | Raft → Manhandla → TF `0x04` residual | after Z3.2 |
| **rr-87a** Z5.2 | 0x67 + dark-room graph | `Level5Cleared66` |
| **rr-076** Z5.3 | Entry 0x76 east Pols Voice + key | `Level5Entrance` |
| **rr-28p** Z5.4 | Whistle + Digdogger + TF `0x10` residual | after graph |
| **rr-miy** Z6.2 | Post-east-key graph (no Old Man key waste) → Rod | `Level6EastKey` |
| **rr-d6v** Z6.3 | Rod + Gohma + TF `0x20` residual | after Rod |
| **rr-q8a** Z8.1 | Candle → burn 0x6D → `Level8Entrance` (open) | `Level8BushOW` |
| **rr-ccx** Z8.1b | Map Blue Candle shop + rupee farm path | OW |

```bash
bd ready -l zelda_i
```

### OW prep (when free)

| Bead | Scope |
|------|-------|
| `rr-38p` | White sword + candle + bomb bag capabilities |
| `rr-dnp` | Lost Hills (done live) + whistle pond L7 |
| `rr-yhr` | Raft dock L4 + bracelet Armos + magical sword |

### Gated (after items / L2 tip)

| Epic | Gate |
|------|------|
| L4 Snake | Raft from L3 |
| L7 Demon | Whistle from L5 + bait shop |
| L9 Death Mountain | Full TF `0xFF` |
| Continuous dry run | assisted then Clean stack |

## Reactive splice (later)

For each pure green segment:

1. Isolated pure from checkpoint (done for L3 west / L5 0x66 / L6 east key)
2. Natural-entry from real predecessor (no mid-run state load)
3. NamedRoute / `routes_later.py` promote when tip arrives
4. Clean STATUS only after natural 2/2 — never from assist-only

Assisted L6 east key needs Clean combat harden before STATUS.

## Process

- [`PROCESS.md`](PROCESS.md) — dual track, pure-first
- [`PARALLEL_RECON.md`](PARALLEL_RECON.md) — wave 1
- [`PARALLEL_PURE.md`](PARALLEL_PURE.md) — wave 2
- [`ASSIST_CONTRACT.md`](../ASSIST_CONTRACT.md)
- [`STATUS.md`](../STATUS.md) — Clean claims only (do not invent promotes here)

## Commands (wave 2 green)

```bash
uv run python nes/zelda_i/scripts/run_level3_west_key.py --trials 2 --save-state
uv run python nes/zelda_i/scripts/run_level5_clear66.py --trials 2
uv run python nes/zelda_i/scripts/run_level5_clear66.py --from-entrance --save-state
uv run python nes/zelda_i/scripts/run_level6_east_key.py --infinite-life --trials 2 --save-state
```
