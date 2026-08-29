# Coding standards

How a **Gut** sitting and the `/code-review` Standards axis judge a diff.
Language: [docs/GLOSSARY.md](docs/GLOSSARY.md) (**Composer**, **Tactic**,
**Gut**, **A/B loop**, **Speed**, **Skill**). Game words stay in that game’s
`CONTEXT.md`. Doc layout: [docs/REPO_HYGIENE.md](docs/REPO_HYGIENE.md).

This file **supersedes** per-game “split before 500” / “extract a sibling”
bars. Hop work still uses the game session skill; it does not claim a **Tip**
or rung from a Gut sitting.

## Composer

Each game has **one** production tick that dispatches **Skills** from a table
of rows. New behavior is a **row** or a **Tactic** behind that tick.

| Tree | Composer |
|------|----------|
| Super Metroid | `tips.play_hops` / `TipSpec` |
| Harvest | `DayPlanTask` / skill table |
| TMNT IV | `Stage1Policy` / `StageSpec` |
| Zelda I | `SpineHop` rows |
| Super Mario Bros. | `tas.stages.StageSpec` |

A sitting that needs a second dispatcher has not finished. Merge the behavior
into the table, or delete it.

## A/B loop (protected)

Three interfaces, not today’s filenames:

1. Load a pin.
2. Play two **Skills** or **input tapes** through the **Composer**.
3. Compare RAM (and video only when a human is watching).

**Gut** may fold a fat loader until it is under ~1000 LOC. **Gut** does not
remove the interface or mint a replacement CLI. **Speed** (button-press / RTA)
goes through this loop only.

**Tape** trees (`human_tape`, `tas`, Harvest CrossMap movies) are *inputs*.
Delete a tape file once the **Skill** exists. Keep the player.

## Delete

Default is delete. Agents add; this file is the prune.

Delete on sight: clone runners (`start_to_*.py`, `*_route.py` beside a
Composer, `run_stageN_segment.py`), mixin clusters (`crop_*.py` as the split
of one FSM), probe CLIs that are not the **A/B loop**, leftover `utils`
whose README says they are not required, stacked residuals / `_vN` reports,
archive trees, dual `CLAUDE.md`.

Before deleting a **Skill** on the living **Tip** or rung: grep callers,
including spec tables and JSON catalogs. If the **A/B loop** would lose
load-pin, play, or compare, fold; do not delete.

Git is the restore path. Prefer a missing file over a second copy.

## Size

Soft max **~1000 LOC** per source file. Crossing 1k means merge into the
**Composer** (or the module that already owns that **Skill**) or delete.
A sibling extract to beat the bar (`foo_2.py`, a 13-file mixin) is a
violation even when every file is under 1k.

Tests for the **Skill** still pass after the fold. A Gut sitting does not
add a package, a new tick, or a probe CLI.

## Sitting

- **Campaign:** one package (`routes/kpdr/`, `harvest/tasks/`, `combat`).
- **Agent:** one source file, or one named cluster.
- **Done:** that file is gone or lives in the owner under ~1000 LOC; no new
  sibling; no second **Composer**; tests for the touched **Skill** pass.
- **Non-claims:** did not edit `STATUS.md`; did not change the living **Tip**
  / rung; did not overwrite a published recording on a red run.

Parallel agents in the same package do not each add a dispatcher.

## Promote

A helper stays game-local until a **second in-game consumer** exists, then a
**second game**, then `retro_harness`. One adapter is a hypothetical seam.

## Review

`/code-review` Standards axis: cite this file (rule + path). Judgement-call
smells stay the Fowler baseline in the skill; a rule here overrides them.
Skip anything a test or `compileall` already enforces.

Implementer load: [`.grok/skills/gut-package/SKILL.md`](.grok/skills/gut-package/SKILL.md).
Root `AGENTS.md` is a pointer, not a copy of these rules.
