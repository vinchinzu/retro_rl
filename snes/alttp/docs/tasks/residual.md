# ALttP assisted power-on audit — 2026-10-01

User-requested independent audit, bead `rr-yglb`. Used the ALttP session gates
in `.grok/skills/alttp-session/SKILL.md`. Planner still owns `docs/STATUS.md`;
this audit does not promote a gate or change AP lane claims. The previous
residual is preserved in the frozen source copy.

## Actual reach

**HP-only assisted execution from a real core reset reaches castle B1 room
`0x81`. Zelda is not rescued; Sanctuary and dungeon approaches remain outside
the tested route.**

Start: stable-retro state `NONE`, reset, blank SRAM prepared before frame 1,
then title/name-entry/Link's House through controller inputs. No dungeon
entry pin, equipment grant, key grant, follower write, or capacity grant.
ROM SHA-256:
`66871d66be19ad2c34c927d6b14cd8eb6fc3181965b6e517cb361f7316009cfb`.

The development controller searched with seven restores and suppressed one
redundant startup SRAM setup. Its committed tape was then replayed in a
separate fresh environment. Every frame matched the full-RAM CRC. Search
restores are development work, not a claim that the controller is Clean.

| Measurement | Reached `0x81` | Attempted mapped Zelda edge |
|---|---:|---:|
| Committed frames from reset | 17,102 (~4:45 at 60 Hz) | 17,447 |
| Replay desync frames | 0 | 0 |
| Replay state loads | 0 | 0 |
| HP writes | 2 | 2 |
| Hearts restored | 4 | 4 |
| Progression / capacity writes | 0 / 0 | 0 / 0 |
| Death frames observed | 0 | 0 |
| Final room / xy | `0x81` / `(996,4485)` | `0x81` / `(776,4256)` |

Refills: frame 13,850 in room `0x61`, and frame 16,585 in room `0x82`;
each writes current HP `8→24`, retaining the existing three-heart capacity.
Assist enabled from frame 1. Final glance: module/submodule `7/0`, sword
`1`, lamp `1`, follower `$F3CC=0`, small keys `0`, floor `0xFF`, layer `1`.

The tested path is House → castle grounds → secret hole/uncle → main hall
`0x61` → `0x60` → `0x50` → `0x01` → `0x72` → `0x82` → `0x81`.
`room_81:west_to_0x80` then spends 345 frames and fails **before reaching
the door**, with `stuck before zelda_west`. This does not prove that the
gold door rejected a key. The actual predecessor has no key or big key.
AP-075's big-key pickup is isolated and is not composed into this path.
The built-in `--through zelda` remains fail-closed.

The ordinary shipped `run_spine --through room_81` command, without HP
assist and using its default `YazeSlot000` start, died in `0x61` during
`castle_dungeon_prefix`. Reported frames: 11,001. It did not reproduce the
historical Clean `0x50` result in this audit. This is a different start from
the `NONE` assisted test, so it is not a controlled measurement of HP's
effect alone.

The local assist contract calls Survival development-only and does not
accept it as continuous product evidence. The zero-load replay above proves
physical reach under the disclosed assist; it does not promote Clean,
Sanctuary, C1, or any later rung.

## Workflow and integrity findings

Workflow `alttp-port-parallel-13` completed during this audit after about
24.6 minutes. It edited isolated boss/tablet work, not the opening chain.
Tracker headings remained **87 open, 94
done, 17 blocked, 2 claimed**. AP-068 is still the lowest open number and
depends on blocked AP-022. AP-113/AP-168 are claimed; AP-120 is done;
AP-165 is blocked. Counts were read independently without changing ticks.

Its final journal reports one south-window sword hit followed by death and
no completed Desert tablet approach. An independent replay of the final
Kholdstare source snapshot (`source_wave13/`) observed **four sword hits**
and eight fire hits, but still died at frame **1,419**. It used **350
restores and 5,464 RAM assignments** (four Link-coordinate assignments plus
5,460 camera assignments). This is isolated progress, not a boss clear or
any improvement to the power-on reach. The result differs from the lane's
reported one-hit attempt; both remain failed attempts.

1. **Write counters have an escape.** Stock `LiveRun` forwards
   `env.unwrapped` to the raw emulator. Helpers using
   `env.unwrapped.data.memory.assign` evade its RAM-write counter.
   A frozen-source Kholdstare replay reports **0 writes** through stock
   `LiveRun`; the same fight through a boundary that preserves auditing
   across `unwrapped` reports **5,868 assignments**. Both runs have 379
   `set_state` calls, eight fire hits, zero sword hits, and die at frame
   1,472. Filed **`rr-0by1`**. Zero writes from the current wrapper are not
   sufficient proof for helpers taking this path.
2. **Kholdstare performs more than camera correction.** `_place` writes
   Link x/y (`$20–$23`, four byte assignments) before the fight. Camera
   `$E2/$E3/$E8/$E9` accounts for 5,864 more byte assignments. These are
   isolated development pokes beyond the HP-only Survival contract.
   The code's “BG2 writes ... only” wording does not describe all writes.
   This result is the frozen version from the start of this audit; Grok
   changed `kholdstare.py` afterward while workflow 13 continued.
3. **Moldorm's success includes restore lookahead.** Independently reran
   offset 0: heart taken in 982 frames, six strikes, no floor hit or HP
   loss, **114 state restores**, no RAM assignments in this offset.
   The source also has a conditional sprite-deflection write (`$0CAA +
   slot`), which did not fire in this test. This verifies one isolated
   offset, not the recorded 12-offset matrix or a power-on Hera route.
   ALttP lookahead remains unruled in `ASSIST_CONTRACT.md`.
4. **Done ticks are not completed route rungs.** AP-073/AP-074 describe
   “continuous” results despite AP-022's restore blocker. AP-106 is called
   “chain + rung C2” but cites a `Whatif_Eastern_armos` prize pickup and
   explicitly says it is isolated. The actual Clean ladder still has all
   15 rows open. The tip runner also undercounts total frames; use committed
   step/tape counts rather than its `frames` field.

## Reproduction and evidence

Evidence root: `recordings/audit_power_on_20261001/` (game-local, gitignored).
`identity.json` and `source/` preserve the dirty runtime and maps; imports
were forced to that frozen copy. ROM/integration files are linked; the ROM
hash is checked by the audit replay. No existing successful baseline, AP
pin, or Grok runtime module was overwritten.

- `power_on_controller.json` / `.tape.npz`: development search to `0x81`.
- `power_on_replay.json`: independent 17,102-frame replay to `0x81`.
- `first_blocker.json` / `.tape.npz`: same reset path plus failed west edge.
- `final_replay.json`: independent 17,447-frame replay through that failure.
- `shipped_room81.json`: ordinary unassisted CLI failure.
- `moldorm_audit.json`, `kholdstare_audit.json`,
  `kholdstare_stock_audit.json`: isolated ROM audit evidence.
- `kholdstare_wave13_audit.json`: independent final-wave Kholdstare test;
  `workflow13_final_state.json` / `workflow13_final_journal.jsonl`: final
  workflow report and lane outputs.
- `pytest.log`: **297 passed, 8 deselected**, offline ALttP gate. These
  tests are separate from the ROM runs and do not establish route reach.

From the repo root, reproduce on the current tree:

```bash
export QT_QPA_PLATFORM=offscreen SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
uv run python snes/alttp/scripts/audit_power_on.py controller \
  --through room_81 --edge room_81:west_to_0x80 \
  --out snes/alttp/recordings/candidates/audit_controller.json
uv run python snes/alttp/scripts/audit_power_on.py replay \
  --tape snes/alttp/recordings/candidates/audit_controller.tape.npz \
  --out snes/alttp/recordings/candidates/audit_replay.json
```

For the exact audited runtime, prepend the absolute paths
`recordings/audit_power_on_20261001/source` and `source/snes` to `PYTHONPATH`.
Search-controller `ok` and replay `ok` have different meanings: the latter
means exact reproduction, and `route_goal_ok` says whether the source
reached its requested endpoint. A replay of a route failure can therefore
pass synchronization.

Next: fix the write-counter escape (`rr-0by1`), then build the missing
natural path/key acquisition before the `0x81` Zelda door (existing
`rr-ccxt.20/.21`, AP-076). Keep Kholdstare and Desert results isolated until
the Sanctuary and item dependencies are actually chained.
