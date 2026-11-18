# Agent instructions: tmnt_iv

SNES TMNT IV: Turtles in Time. The target is a **Clean** Hard clear from
power-on through the staff credits: zero life losses, zero RAM writes, zero
state loads, no A-special. Words: [CONTEXT.md](CONTEXT.md). How the guard
works and what each file owns: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
Verified results: [docs/STATUS.md](docs/STATUS.md). Next work:
[docs/plan.md](docs/plan.md). What counts as Clean:
[docs/ASSIST_CONTRACT.md](docs/ASSIST_CONTRACT.md).

## Method

The ROM is the enemy model. `GuardedPolicy` (`lookahead.py`) commits a frame
only after the emulator has played the next 64 frames hit-free. On a
predicted hit it branches macros from snapshots inside that uncommitted
window and splices in the best hit-free future. `Stage1Policy` and
`tactics/` exist to make **progress**; **safety** is the guard's job.

Work a death as a **first-failure loop**: find the guard's `forced` lines
(stdout and the report's `lookahead`), reproduce from the natural pin
before it (`--from Clean<Turtle>Stage<N>`), fix by widening the search
(`MACROS`, `LookaheadConfig`) or by giving the base policy a better approach
to that spot, then re-run from power-on.

## Commands

Run from the repo root. Prefix `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy`
when headless.

```bash
uv run python -m tmnt_iv.scripts.setup_rom          # relink the ROM (after a clone or repo move)
uv run python -m tmnt_iv.scripts.run --save-points CleanLeo   # power-on -> credits; JSON + tape; pins <prefix>Stage<N>
uv run python -m tmnt_iv.scripts.run --from CleanLeoStage5 --until-stage 6   # one stage from a natural pin
uv run python -m tmnt_iv.scripts.run --replay snes/tmnt_iv/recordings/<tag>.tape.npz --video   # the publishable mp4
uv run python -m tmnt_iv.scripts.run --no-lookahead --from <pin> --until-stage N   # bare-policy A/B arm
QT_QPA_PLATFORM=offscreen uv run pytest snes/tmnt_iv/tests -q   # no ROM needed
```

## Traps

- `scripts/run.py` is the only CLI; `run/trial.py` `run_trial` is the only
  live loop. New behavior is a tactic, a macro row, or a `LookaheadConfig`
  knob.
- The guard reaches the core only through `GuardedPolicy.bind`, and every
  restore lands on the live frame or its future. A restore to an earlier
  frame is a rewind and voids the run.
- Rollouts pass the real frame number to `parse_game_state`: Baxter, pizza,
  and Raph tactics key off `state.frame % N`.
- A run that stops progressing: read the report's `lookahead` (`accepted`,
  `searches`, `unstuck`) and the enemy list at the stall. Guard vetoes are
  solved by the stall budget, policy deadlocks by unstick; fix the tactic
  only when neither moves it.
- A grab (turtle frozen beside a Foot, hit every ~33 frames) breaks only to
  a d-pad mash: UP/DOWN alternating every frame frees in 7 frames, 0 HP.
  Held directions, jumps, and pokes never do. Keep the `mash_*` macros.
- Nine enemy slots (`0x08D0` + i*0x70, i=0..8), not the 7 the PAR codes
  list. A frozen "empty" screen means an enemy the parser cannot see; dump
  the raw slots before tuning a stall tactic.
- `0x003A` is a frame clock (+1 per frame), not camera progress.
- A stage clear zeroes the player entity (HP 0, X 0). The guard treats HP 0
  as a KO only with the turtle on screen outside a fade event.
- Default `--turtle leo`: the base policy's reach was tuned on Leo. Mike
  has 96 HP to Leo's 80 (hit sizes match) but trades hits at Metalhead.
- Publish a replayed tape (`--replay ... --video`); the replay checks RAM
  CRC every 600 frames. Live-search runs write JSON + tape only.
- After boot, START (pause) is stripped. A (special) drains HP; the trial
  stops on it.
- Clean artifacts use `*_clean` / `clean_poweron_*` names and leave the
  assisted baseline `tmnt_iv_full_hard_credits*` in place.
- Prefer `Clean<Turtle>Stage<N>` pins. The 400+ legacy pins in
  `custom_integrations/` mix turtles and assisted HP.
