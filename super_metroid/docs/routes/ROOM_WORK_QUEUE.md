# Room work queue — easiest first

Isolated room-clear practice board (teleport → policy → promote).
Not continuous-run evidence. Source catalog: `maps/room_problems.json`.

Regenerate:

```bash
uv run python super_metroid/scripts/export/room_work_queue.py
```

## Units

Ranked units are the 262 canonical room problems (one per room). The full graph has 583 directed edges; those are topology hops, not separate practice harness entries. Prefer this queue for easiest-first teleport practice; use KPDR_TRACKER for continuous spine milestones.

## Percent complete (practice harness)

| Scope | Ready % |
|-------|--------:|
| Easy + standard (queues 0–2) | **47.2%** |
| Non-boss (queues 0–3) | 20.3% |
| All 262 room problems | 19.5% |
| Completion-path rooms only | 8.4% |
| Easy+standard teleport fixtures | 51.9% |
| All teleport fixtures | 21.4% |

## Counts

| Metric | Count |
|--------|------:|
| Room problems | 262 |
| Directed edges (full graph) | 583 |
| Completion-path rooms | 107 |
| Teleport-ready (entry `.state`) | 56 |
| Run-ready (state + verified policy) | 51 |
| Easy+standard total | 108 |
| Easy+standard ready | 51 |
| Tough/late (queue 3) | 143 |
| Boss deferred (queue 4) | 11 |

### By queue

| Queue | Meaning | Count |
|------:|---------|------:|
| 0 | state + verified policy ready | 3 |
| 1 | easy / small rooms | 67 |
| 2 | standard traversal | 38 |
| 3 | tough / late / unresolved geometry | 143 |
| 4 | bosses held for later | 11 |

### By tier

| Tier | Count |
|------|------:|
| `boss_late` | 11 |
| `easy` | 69 |
| `late_special` | 27 |
| `standard` | 38 |
| `tough` | 117 |

## How to work top-down

1. Export / refresh this board.
2. Bootstrap entry states for easy rooms (door-warp fixtures):

```bash
uv run python super_metroid/scripts/room/run_problem.py bootstrap --queue 1
```

3. Scaffold a policy, iterate, then promote on a green isolated run:

```bash
uv run python super_metroid/scripts/room/run_problem.py scaffold PROBLEM_ID
uv run python super_metroid/scripts/room/run_problem.py teleport PROBLEM_ID
uv run python super_metroid/scripts/room/run_problem.py run PROBLEM_ID --promote
```

4. Leave queue 3 large rooms and queue 4 bosses until easy+standard % is solid.

## Next open easy (top of queue)

| Rank | Score | Room | Problem | Teleport |
|-----:|------:|------|---------|:--------:|
| 52 | 50 | Blue Brinstar Boulder Room `0xA1AD` | `room_a1ad_from_9f64_to_a1d8` | yes |
| 53 | 50 | Crab Hole `0xD21C` | `room_d21c_from_d3b6_to_d08a` | yes |
| 54 | 50 | Ice Beam Tutorial Room `0xA865` | `room_a865_from_a815_to_a8b9` | yes |
| 55 | 50 | Grapple Tutorial Room 2 `0xABD2` | `room_abd2_from_ab64_to_ac00` | yes |
| 56 | 50 | Metal Pirates Room `0xB62B` | `room_b62b_from_b482_to_b5d5` | yes |
| 57 | 10183 | Varia Suit Room `0xA6E2` | `room_a6e2_from_a59f_to_a59f` | no |
| 58 | 10183 | Post Crocomire Power Bomb Room `0xAADE` | `room_aade_from_aa82_to_aa82` | no |
| 59 | 10184 | Etacoon Super Room `0xA051` | `room_a051_from_a011_to_a011` | no |
| 60 | 10186 | Space Jump Room `0xD9AA` | `room_d9aa_from_da60_to_da60` | no |
| 61 | 10186 | Ice Beam Room `0xA890` | `room_a890_from_a8b9_to_a8b9` | no |
| 62 | 10186 | Speed Booster Room `0xAD1B` | `room_ad1b_from_acf0_to_acf0` | no |
| 63 | 10186 | Wave Beam Room `0xADDE` | `room_adde_from_adad_to_adad` | no |
| 64 | 10187 | Hopper Energy Tank Room `0xA15B` | `room_a15b_from_a130_to_a130` | no |
| 65 | 10187 | Spazer Room `0xA447` | `room_a447_from_a408_to_a408` | no |
| 66 | 10190 | Billy Mays' Room `0xA1D8` | `room_a1d8_from_a1ad_to_a1ad` | no |
| 67 | 10232 | Brinstar Reserve Tank Room `0x9C07` | `room_9c07_from_9bc8_to_9bc8` | no |
| 68 | 10237 | Gravity Suit Room `0xCE40` | `room_ce40_from_c98e_to_93fe` | no |
| 69 | 10247 | Wrecked Ship West Super Room `0xCDA8` | `room_cda8_from_caf6_to_caf6` | no |
| 70 | 10264 | Crateria Power Bomb Room `0x93AA` | `room_93aa_from_91f8_to_91f8` | no |
| 71 | 10575 | Hi-Jump Room `0xA9E5` | `room_a9e5_from_aa41_to_aa41` | no |
| 72 | 10624 | X-Ray Room `0xA2CE` | `room_a2ce_from_a293_to_a293` | no |
| 73 | 10673 | Alpha Power Bomb Room `0xA3AE` | `room_a3ae_from_a322_to_a322` | no |

Full ranked table: `docs/routes/ROOM_WORK_QUEUE.csv` · machine JSON: `maps/room_work_queue.json`.

_Generated 2026-07-30T22:47:49.383170+00:00_
