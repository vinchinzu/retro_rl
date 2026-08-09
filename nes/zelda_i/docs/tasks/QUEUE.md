# Queue — Zelda I

**Source of truth:** `bd ready -l zelda_i`

Human snapshot. Expand room beads when that dungeon is the tip (or for
parallel isolated pure from checkpoints).

## Agent priority (finish → tune)

1. **Pathfinding + puzzles** (hops, doors, keys, bomb walls, blocks, items)
2. **Assisted full route** (`--infinite-life`; damage heatmap in report)
3. **Clean harden later** — rank rooms by `assist.damage_by_location`; combat
   polish is residual, not tip-blocking

Infinite life is intentional for first-pass agents. Do not stall L2 tip on
sword kiting once geometry is known.

## Live tip (post-L3 — assist-first)

| Gate | Status | Notes |
|------|--------|-------|
| Power-on → L1 Triforce | **Clean green** | M5 |
| L2 Boom → TF `0x02` | **assisted green 2/2** | `rr-5dk` / `rr-n5i`; `l2_complete_assisted.json` |
| Post-L2 → L3 enter | **assisted green 2/2** | `rr-rnx`; `l2_to_l3_assisted.json`; epic `rr-ci7` closed |
| L3 Raft → Manhandla → TF `0x04` | **assisted green 2/2** | `rr-vpl` / epic `rr-wmv` closed; `level3_to_boss_assisted.json` |
| Checkpoints | **`Level3Raft`**, **`Level3Boss`**, **`Level3Complete`** (raft=1) | L3 epic closed |
| L4 OW entry | **assisted green 2/2** | `rr-0fx`; dock **0x55** island **0x45** room **0x71**; `l4_entry_recon.json` |
| Checkpoints L4 | **`Level3ExitOverworld`**, **`OW_L4Dock`**, **`Level4Entrance`** | not Clean |
| **Tip leaf** | **`rr-5lu` Z4.2** L4 interior | from `Level4Entrance` Stepladder path |
| Parallel free | **`rr-38p`** early OW caps | white sword / candle / bomb bag |
| Deferred (blocked on tip) | L5/L6 TF residual, Clean L2 heatmaps, bracelet/mag sword | P4; not tip-blocking |

### Architecture (agent monitor)

```
tip spine:  L1 Clean → L2 assist TF → L3 assist TF+Raft → L4 entry ✓ → **L4 interior (rr-5lu)**
parallel:   OW early caps (rr-38p); isolated pure only from green checkpoints
defer:      combat Clean harden, later-dungeon TF residual until tip arrives
process:    path/puzzle first → --infinite-life full clear → damage heatmap Clean
```

Claim one tip leaf: `bd update rr-5lu --status in_progress`.

### All-night wave results (2026-08-06 night)

| Bead | Result |
|------|--------|
| **rr-lzk** | **2/2 Clean** bomb N 0x6f→0x5f; `run_level2_bomb_north.py` |
| **rr-etl** | **2/2 Clean** Goriya 0x5e; `run_level2_clear5e.py` |
| **rr-fvt** | 0x5f = 5× Gel; doors DOWN-only; clear ≠ open R/U |
| **rr-cjf** | **LIVE** 0x5f bomb-UP→**0x4f**; 0x5e UP→0x4e→0x4f |
| **rr-3pz** | `level2_puzzles.py` catalog |
| **rr-mhl** | `door_graph.py` + L2 seed |
| **rr-65w** | **2/2 Clean** L3 north to 0x5b Darknuts |
| **rr-vpl** tip | Raft **2/2** + Manhandla→TF **2/2 assisted** (`run_level3_to_boss.py` / `Level3Complete`) |
| **rr-miy** | **2/2 assisted** L6 west wizzrobes 0x78 |
| **rr-87a** / **rr-076** | L5 graph PARTIAL (0x67 dead-end; Pols door residual) |
| **rr-iri** / **rr-ccx** / **rr-q8a** | OW hops + candle shop 0x5E; burn residual |

## Parallel wave results (2026-08-06)

### Wave 1 — recon (OW door + entry)

| L | Door OW | Entry room | Checkpoint | Track |
|--:|---------|------------|------------|-------|
| 3 | **0x74** | **0x7c** | `Level3Entrance` | assisted enter |
| 4 | **0x45** (dock **0x55**) | **0x71** | `Level4Entrance` | assisted enter |
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

## Next beads (tip + ready)

```bash
bd ready -l zelda_i   # tip: rr-5lu; parallel: rr-38p
```

| Bead | Role | Start / notes |
|------|------|---------------|
| **`rr-0fx` Z4.1** | ✓ live L4 entry | dock `0x55` island `0x45` room `0x71`; `run_level4_entry.py` |
| **`rr-5lu` Z4.2** | **TIP** interior | from `Level4Entrance`; first rooms + stepladder |
| **`rr-38p` ZOW.1** | parallel free | white sword + candle + bomb bag |
| **`rr-q3n`** | L4 epic container | more children after interior recon |
| Deferred P4 | `rr-28p` L5 TF, `rr-d6v` L6 TF, `rr-4oz` Clean L2, `rr-yhr` bracelet/mag sword | blocked on tip |

### Closed wave history (keep for evidence)

L3 west key / Raft / Manhandla TF, L5 clear 0x66, L6 east key, L2 bomb/Goriya,
OW recon table — see closed beads + `LEVEL{2,3,5,6}_ROUTE.md`.

### Gated (after items)

| Epic | Gate |
|------|------|
| L4 Snake clear | live entry ✓; interior `rr-5lu` |
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
