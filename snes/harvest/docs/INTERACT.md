# Interact without a new tape

Record **only** when the walk corridor is unknown **and** live BFS has no gap.
Picks, talks, and keep-menus are RAM + decomp. The grape mess was: wrong
object class → Gotz talk → ban A → ignore a tape that already had `held=0x03`.

## Order (do not skip)

1. **Scan an existing tape**, do not re-record it.
   ```bash
   uv run python -m harvest.scripts.interact_scan tape mountain_grape_stand
   # no trace? replay once:
   HEADLESS=1 uv run python -m harvest.scripts.mountain_berry_probe \
     --mode replay --task get_berry
   ```
   First `held_item` change **is** the pick. Read the ±80 frames of buttons
   (A / Down). That is the protocol.
2. **Read the box** in decomp before writing A logic.
   ```bash
   uv run python -m harvest.scripts.interact_scan search grape
   ```
   Eat / Don't eat means **Down then A to keep**. Mash-A eats. Gotz text
   means you faced an NPC.
3. **Live stand scan** (screenshot + objects + one A) from a pin you already
   reached. `0x02xx` sprites are NPC candidates. Ground forage is not a 2×2
   plant tile and is not `face_toward_bush`.
   ```bash
   HEADLESS=1 uv run python -m harvest.scripts.interact_scan tap --state <pin>
   ```

## Classify (held / lock / text)

| held | lock | text / face | Class | Do |
|------|------|-------------|-------|----|
| 0→forage | 2 | Eat / Don't eat | `forage_keep_menu` | Down, A. Wait lock=1 |
| 0→forage | 1 | (anim) | pickup anim | wait for the box; do not succeed yet |
| stays 0 | 2 | NPC `0x02xx` in face tile | `npc_talk` | fail closed. Do not mash-learn |
| stays 0 | 1 | — | miss | you walked off the item (face-walk is movement) |

Green = item still held **and** `input_lock=1` **and** the ground sprite is
gone (or keep-menu closed). "Reached stand" is not a pick. First held tick
during A is not "kept."

## Do not

- Cargo-cult farm-bush A onto a mountain ground spawn.
- Ban A after one Gotz talk.
- Call a box "untrusted" without `dialog_text_id` + UnlinkedText.
- Record house→item to "feel" a pick you can scan.
- Hold a face direction to "face" — that **walks**. On-tile A only.

Nav corridor (cliff, carpenter gap) is the one thing worth a short tape, and
only after MultNav BFS from the live land tile fails.
