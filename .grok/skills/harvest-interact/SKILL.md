---
name: harvest-interact
description: >
  Classify a Harvest Moon pick/talk/keep-menu from an existing tape or a live
  pin before writing A. Do not record a new walk to "feel" a forage. Use when
  the user says "pick that", "don't talk", "grape", "forage", "Gotz", "item
  box", "Don't eat", "scan the tape", or runs /harvest-interact.
---

# Harvest interact (scan, don't record)

Read [snes/harvest/docs/INTERACT.md](../../../snes/harvest/docs/INTERACT.md)
and run the scan commands there before any new `tasks/*.json` recording.
Session gates: `harvest-session`.

## This turn

1. `interact_scan tape <existing>` if a recording exists. First `held_item`
   change is the pick; the next Down/A window is the keep-menu.
2. `interact_scan search <item>` and believe UnlinkedText. Eat/Don't eat is
   keep, not Gotz.
3. If you already have a stand pin: `interact_scan tap --state <pin>`.
   Classify from the table in INTERACT.md.
4. Implement the class you measured. Face-walk is movement. Fail closed on
   mountain/town dialogue with `held=0`.
5. Probe greens only on **kept** (held + lock=1), never on "reached" or the
   first held tick.

Do not record house→item. Record a corridor only after live BFS from the
land tile has no gap — then use [harvest-route](../harvest-route/SKILL.md).
Shop doors: [harvest-shop](../harvest-shop/SKILL.md).
