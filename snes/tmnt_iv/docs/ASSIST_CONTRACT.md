# Assist contract: TMNT IV

Runtime observation: **Bronze** (game-specific read-only RAM). Play traps
live in [`AGENTS.md`](../AGENTS.md).

## Clean (published M8 result)

A Clean run, enforced by `run_trial` under `CLEAN_CONTRACT`:

- zero RAM writes of any kind (HP, iframes, stage, lives, inventory)
- zero state loads after power-on other than lookahead restores
- zero life losses (the first lost life fails the run), no continue
- no A-special (the trial stops on it); START is stripped after boot
- Hard difficulty confirmed in WRAM, staff/cast credits observed

Natural pizza (`char 0x30`) picked up with controller input is play, not an
assist.

### ROM lookahead (disclosed)

The live policy plays candidate futures on the emulator and restores
(`tmnt_iv.lookahead`). Adopted 2026-11-14 at the owner's direction,
following the Zelda I ruling of 2026-09-28 (lookahead allowed; the recording
is the tape replayed with none). Rules:

- Restores land only on the live frame or its future, never earlier.
- Reports disclose them as `rollout_restores`, separate from `state_loads`
  (which must be 0).
- The published video is `scripts/run.py --replay <tape> --video`: power-on,
  the committed buttons, no policy, no lookahead, RAM CRC checked every 600
  frames.

An interrupted tape may be extended with neutral input through an already
started ending (`--finish-credits`). Recovery replays from power-on, checks
all original CRCs, and records the source tape hash and the added frames.
It cannot supply missing gameplay. The complete tape must then pass a
normal replay through the final Splinter scene and settle period.

## Assisted (previous M8 baseline)

| Assist | Trigger | Write | Notes |
|--------|---------|-------|-------|
| Emergency HP | HP ≤ 16 | restore HP to 80 | counted per intervention |
| Form-2 iframe hold | Super Shredder form 2 (stage 9, event `0x0A`) | hold iframe timer at 1 | counted per frame |

`--assisted` selects this contract. Its reports carry both counters and are
never labelled Clean. Assisted artifacts keep the `tmnt_iv_full_hard_credits*`
names; Clean artifacts use `*_clean` / `clean_poweron_*` and never overwrite
them.

## Forbidden under every contract

Stage / progress / boss flag writes, lives grants, inventory or character
unlocks, mid-run save-state loads.
