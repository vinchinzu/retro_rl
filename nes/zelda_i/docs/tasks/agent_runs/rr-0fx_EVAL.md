# prime-agent deepseek-v4-flash-0731 eval — rr-0fx

**Date:** 2026-08-07
**Model:** `openrouter/deepseek/deepseek-v4-flash-0731` via **prime-agent** CLI
**Bead:** `rr-0fx` (tip leaf; left **in_progress**, not closed)
**Result:** **STOPPED** after 3 runs — gates **not** met. Recon partial only.

## Verdict

prime-agent + flash-0731 is a **fast scout / scaffold worker**, not a **tip-closer** for multi-hop live OW geometry under dual-track purity rules.

| Capability | Grade | Notes |
|------------|-------|-------|
| Harness boot / RAM read | A | Correct `make_env`, settle, raft check, no poke |
| Isolated probe scripts | B+ | Many `_tmp_*.py`; noisy but directional |
| Live geometric discovery | B− | Found settle `0x74`, hops to `0x63`, exit map UP→`0x53` |
| Promote to controller + 2/2 gate | D | Never wrote `OverworldToLevel4` or evidence JSON |
| Tool discipline | D | attach_image thrash; nonlocal SyntaxError loops |
| Token efficiency | D | 150–400k tok per run before durable code |
| Autonomy vs gates | C− | Needs human kill/steer; gates unused |

## Hygiene done (before dispatch)

- Ready queue: tip **`rr-0fx`**, parallel **`rr-38p`**
- Spawned **`rr-5lu`** blocked on tip; deferred residuals P4
- Updated `QUEUE.md` / `AGENTS.md` / `PROCESS.md`

## Runs (stopped)

| Run | Tokens | Outcome |
|-----|--------|---------|
| r1 | ~186k | Settle **0x74 raft=1**; **0x74→0x73→0x63** live; fail east from y=221 |
| r2 fork | ~418k | Image-tool thrash; no product code |
| r3 no-skills | ~167k | y=221→140 via UP; **0x63 RIGHT not free** (UP→**0x53**); still no controller |

**Live facts kept for humans / stronger agent:**

- `Level3Complete` → OW **0x74** (128,125) raft=1 tf=0x04
- Working: LEFT→0x73 @y141, UP→0x63 (arrive y=221)
- On 0x63: UP into screen → y≈140; **east exit not free** (try alternate route to dock 0x55)
- Do not trust fixture `OW_55` raft=0 for entry claims

## What to give prime-agent (flash)

**Good fit (bounded, checkable):**

- One-file pure refactors / renames with pytest gate
- Extend an existing runner flag or stop predicate with golden pattern nearby
- Doc/bead hygiene, QUEUE/STATUS table updates from evidence already in hand
- “Read X, print RAM snapshot from state Y” probes (single script, &lt;50 lines)
- Catalog work: door_graph seed rows, puzzle catalog entries from known room IDs
- Mechanical dual-track plumbing: wire `--infinite-life` flag onto existing controller
- Generate first-draft hop table **after** human/live IDs known

**Bad fit (avoid or pair with stronger model):**

- Open tip leaves that need multi-screen live geometry discovery + 2/2 natural
- Anything requiring vision / screenshot interpretation (tool thrash)
- Boss policies, combat polish, Clean STATUS promote
- Multi-module “implement OverworldToLevelNController end-to-end” without seed hop list
- Long autonomous runs without mid-gate checkpoints (token burn + drift)

## How to dispatch (tight)

```bash
# Prefer small goals + file gates + low thinking + no skills for flash
prime-agent \
  --provider openrouter \
  --model deepseek/deepseek-v4-flash-0731 \
  --thinking low --no-skills \
  --cwd "$PWD" \
  --goal "ONE concrete deliverable only" \
  --autonomous \
  --autonomous-gate 'test -f <one-evidence-file>' \
  --autonomous-max-tokens 80000 \
  --autonomous-max-turns 12 \
  --autonomous-timeout-ms 900000 \
  "Constraints: no docs crawl; no images; write product code early."
```

**Monitor:** `~/.prime/agent/sessions/*.jsonl` (TUI `list` often empty for headless).

## Gates (failed)

- [ ] `Level4Entrance.state`
- [ ] `recordings/l4_entry_recon.json`
- [ ] live ids in `level4_overworld.py`
- [ ] `bd close rr-0fx`
