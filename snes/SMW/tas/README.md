# SMW TAS Oracle

This directory contains tracked parsers and verification tooling. ROMs, source
movies, generated BK2 files, states, logs, and extracted skills are local,
ignored artifacts.

## Source Roles

| Source | Role | Current result |
| --- | --- | --- |
| 2022 native Snes9x full-clear BK2 | Current native compatibility | GREEN Yoshi's Island 2; later retries/game-over make it unsuitable for clean multi-level skills |
| 2016 native BizHawk warps BK2 | Optimized movement | Deterministic GREEN through Yoshi's Island 2 and 3 after metadata-only BSNESv115+ retarget; input log is unchanged |
| 2025 all-96-exit ACE BK2 | BizHawk 2.11 compatibility and level enumeration | Upstream sync-verified; forced exits are not movement skills |
| 2011 Snes9x 1.43 SMV | Negative control | RED current-core desync |
| bsnes v085 LSMV | Negative control | RED current-core desync |

The online catalog does not currently provide a newer optimized vanilla
movement BK2 that is more useful than the native warps port. Newer is not
automatically better: the verifier, not the publication date, decides whether
an input window can become a skill.

## Commands

Replay an explicit native BK2 without changing its core metadata:

```bash
uv run python -m SMW.tas.oracle_runner modern_probe \
  --source snes/SMW/tas/ref/smw_2022_full_clear.bk2 \
  --target-levels 1 --max-frames 20000
```

Retarget the native optimized warps input log to BizHawk's current bsnes core:

```bash
uv run python -m SMW.tas.oracle_runner warps_probe \
  --source snes/SMW/tas/ref/warps_3019_bizhawk.bk2 \
  --core-profile v115 --target-levels 1 --max-frames 10000
```

Each GREEN segment writes an entry state, exit state, RAM fingerprints, event
log, and an exact RLE input artifact under
`snes/SMW/recordings/tas_oracle/<source>/<run>/`. Skills are labeled
`clean_single_attempt` or `replay_with_retries`; RED and aborted segments do
not produce skills.

Compare independent GREEN proofs before promoting their segment boundaries:

```bash
uv run python -m SMW.tas.compare_proofs \
  snes/SMW/recordings/tas_oracle/<source>/run_a/proof.json \
  snes/SMW/recordings/tas_oracle/<source>/run_b/proof.json
```
