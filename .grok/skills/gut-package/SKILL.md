---
name: gut-package
description: "Gut one package: merge a file into the Composer or delete it. Use when the user says gut, fold, prune, coding standards, delete unused, or runs /gut-package."
disable-model-invocation: true
---

# Gut a package

Read [CODING_STANDARDS.md](../../../CODING_STANDARDS.md) first. Language:
[docs/GLOSSARY.md](../../../docs/GLOSSARY.md). This sitting is **Gut**, not
a **Tip** / rung. Session skills (`sm-session`, `harvest-session`) stay
product work.

## Loop

1. Name the **campaign package** (one directory). Name the **Composer** for
   this game from the standards table. If the package already has two ticks,
   the sitting is “merge onto one,” not “add a third.”
2. Take **one source file** or one named cluster (`crop_*`, one probe CLI).
   Campaign is the package; the agent is the file.
3. Classify: **A/B loop** interface (load pin / Composer play / compare) →
   fold under ~1000 LOC. Live **Skill** on the tip/rung → grep callers
   (including spec tables and JSON), then merge into the owner. Anything
   else → delete.
4. Apply [CODING_STANDARDS.md](../../../CODING_STANDARDS.md) (every rule).
   Crossing 1k: merge or delete. A new sibling file is a failed sitting.
5. Run the narrowest tests for the touched **Skill**. `compileall` on the
   package if there is no test.
6. Overwrite one residual if the package has one. Do not mint `_vN`.
7. Stop. Do not start the next file in this sitting.

## Done

The file is gone, or it lives in the owner under ~1000 LOC. No new
**Composer**. Tests pass. `STATUS.md` untouched.

## Non-claims

Did not STATUS-promote. Did not change the living **Tip** / rung. Did not
rewrite the **A/B loop**. Did not extract `foo_2.py` to beat the line bar.
Did not claim product progress from a delete.
