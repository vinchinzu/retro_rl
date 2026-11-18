# Architecture: TMNT IV

Language: [CONTEXT.md](../CONTEXT.md). Commands: [AGENTS.md](../AGENTS.md).

## Why lookahead

Until 2026-11 the run was a reactive behavior tree tuned stage by stage on
save-state pins. It plateaued: heuristic dodging cannot see 4-24 HP chip
hits coming, pins did not match the RNG of a continuous run, and each
retune moved deaths to another stage. Measured 2026-11-14, the bare policy
died in Big Apple from power-on with every turtle.

The emulator answers "does this get me hit?" exactly and cheaply:

| Call | Cost (Ryzen 9, stable-retro, dummy SDL) |
|------|------|
| `em.step` | ~700 us (~23x real time) |
| `em.get_state` / `set_state` | ~30 us / ~80 us (430 KB state) |
| `parse_game_state` + policy tick | ~60 us |
| policy `deepcopy` | ~90 us |

Restore + 300 replayed frames is bit-exact. So the guard asks the ROM
instead of guessing.

## Layout

| Path | Owns |
|------|------|
| `ram.py` `stages.py` | RAM layout, `GameState` adapter, stage predicates |
| `menus.py` | Power-on boot: Hard + turtle choice |
| `policy.py` + `tactics/` | Base policy (progress); `tactics/knobs.py` constants |
| `lookahead.py` | `GuardedPolicy`: verified plan, rescue, macros |
| `tape.py` | Tape save/load, CRC checkpoints |
| `run/trial.py` | The one live loop: contract, audit, tape record/replay |
| `run/metrics.py` `run/report.py` `run/video.py` `run/freeze.py` | Credits tracking, report JSON, video overlay, stall abort |
| `assist.py` | Old assisted contract writes (not Clean) |
| `scripts/run.py` | The one CLI |

## The guard

```
live frame 0 ──── verified plan (64+ frames, each with expected RAM CRC) ──── frontier
    │ commit plan[0]                         ▲ extend by chunk (8) from frontier
    ▼                                        │ hit? → rescue
  env.step                     nodes every 8 frames: (state, policy copy, frame#)
```

1. **Extend**: restore the frontier node, play the base policy for one chunk,
   snapshot, restore live. Cost ≈ 1x emulation when nothing threatens.
2. **Rescue**: a predicted HP drop or KO at offset `h` tries every `MACROS`
   prefix (then the base policy) from the latest node at `h - lead` for leads
   16/32/48/64, until a lead yields a hit-free branch through `h + margin`.
   The best branch by score (life ≫ HP lost ≫ pizza, enemy damage) replaces
   the plan after that node.
   `MACROS` include d-pad mashes (the only grab escape) and 32-48 frame holds: after a short dodge the base policy
   often walks straight back into the same hit (Alleycat `LiveHardStage2`
   went from one forced 4 HP hit to 0 damage when they were added).
3. **Pairs**: if no single macro is hit-free, the 6 best first macros at
   the earliest node are each followed by every macro (depth 2). Skipped
   within 240 frames of the last forced hit; a lead stops after 3 hit-free
   branches, trying macros in order of past rescues.
4. **Forced**: still nothing hit-free → least-bad branch. The log line names
   `hit_at`, the plan length, and the node offsets, so a late-seen threat
   and a truly unavoidable one look different.
5. **Stall budget**: Clean forbids lost lives, not damage. *Progress* is
   HP lost in any occupied enemy slot (off-screen bosses included: the
   Technodrome tank is slot 0, char `0xFE`, HP 90, and counting only
   on-screen enemies hid a 300,000-frame tank loop), a stage change, or
   every slot empty. After `stall_frames` (1800) without progress, a
   predicted policy hit is **accepted** if it leaves HP at or above
   `hp_floor` (half the bar). The winning Technodrome pin run took 4 such
   trades (38 HP). At 600 frames it fired inside a winnable Baxter fight
   and derailed it. Counted as `accepted` / `accepted_hp`.
6. **Unstick**: the base policy also deadlocks on its own: tactics that
   align toward an enemy standing just off the walkable band (Technodrome
   Foot at y=144 over Leo's y=160 floor, a Prehistoric dino at y=128 over
   y=136) press UP or poke forever while nothing threatens. While stalled,
   every 300 frames the guard rolls every macro for 120 frames from the
   live point and splices the branch that damages an enemy (life kept, HP
   over the floor). Counted as `searches` / `unstuck`.
7. **Commit**: the live loop plays `plan[0]`. If live RAM CRC differs from
   the plan's expectation, the plan is dropped (`desyncs`).

Branch points are nodes at offset ≥ 0 only: every restore lands on the live
frame or its future, so the tape holds no rewinds. Each `tick` checks the
live RAM CRC is unchanged after its rollouts.

## Audit

`run_trial` wraps the env in `_AuditEnv`: `set_value` writes and
`em.set_state` loads after live are contract violations under Clean. The
guard is bound to the raw core, so its restores are reported as
`rollout_restores`, not loads. Every trial returns its tape; a replay
(`TrialEntry.tape`) drives every frame from the tape and stops `desync` on
the first CRC miss.

An interrupted ending tape can be recovered with `--replay <tape>
--finish-credits`. The runner replays every saved input from power-on and
checks its CRCs. Only after the ending has started may an exhausted tape
continue with neutral input. It records a complete replacement tape,
including CRCs through the credits, and records the source tape's SHA-256.
An exhausted gameplay tape still fails. The replacement is then replayed
normally with `--video`, without recovery or lookahead. Replay input is
audited too: A-special and post-boot START fail before being committed.

The ending latches when the stage byte reaches 10. Its cast demonstrations
reuse stages 0-9 and player HP/lives; those changes are excluded from
gameplay damage, healing, deaths, and stage splits. Write/load/input/CRC
audits remain active throughout the roll. Completion still requires the
Hard credits event, both stage-9 cast demonstrations, the final Splinter
transition, 1,200 settle frames, and a further 600-frame capture hold.

## Artifacts

JSON + tape (+ mp4 on replay) under `recordings/` (gitignored). Dev runs land
in `recordings/dev/`. Natural pins land in `custom_integrations/TMNTIV-Snes/`
(gitignored).
