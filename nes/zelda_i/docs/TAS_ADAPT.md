# TAS adaptation — The Legend of Zelda (NES)

Mirror of SMB / Super Metroid button-movie import: pull published FCEUX
``.fm2`` streams, map to NES-9, then **segment-adapt** against our graph
nav / dungeon controllers.

**Policy: prefer non-glitch movies.** TASVideos has **no formal glitchless
any%** for LoZ. Use publications that are **not** tagged
[Heavy glitch abuse](https://tasvideos.org/Movies-bugs). Soft reset, damage
boost, and item-drop manip still appear; route-breaking glitches (status-bar
scroll travel, game-end ACE, etc.) are out of scope for Clean STATUS work.

## Open sources (button movies)

### Non-glitch (default fetch)

| Source | Time | Format | Path / URL | Notes |
|--------|------|--------|------------|-------|
| **chatterbox all-items** (published) | **31:52.07** / 114 913f | `.fm2` | `tas/ref/chatterbox_allitems_4767M.fm2` · [4767M](https://tasvideos.org/4767M) | **Primary.** No heavy-glitch tag. Console-verified. **PRG1** matches our ROM. Soft reset + intentional damage only. Author skipped recorder-warp routes. |
| TASeditor all-items (obsolete) | 32:16.98 | `.fm2` | `tas/ref/taseditor_allitems_2508M.fm2` · [2508M](https://tasvideos.org/2508M) | Prior all-items; also no heavy-glitch tag. **PRG0**. |

### Glitched (opt-in: `--include-glitched`)

| Source | Time | Why excluded by default |
|--------|------|-------------------------|
| Lord Tom any% [3232M](https://tasvideos.org/3232M) | 22:17.53 | Heavy glitch abuse; PRG0 |
| Lord Tom swordless [3289M](https://tasvideos.org/3289M) | 24:39.71 | Heavy glitch abuse |
| chatterbox 2nd quest [4715M](https://tasvideos.org/4715M) | 24:48.75 | Heavy glitch abuse |
| FDS game-end glitch [2868M](https://tasvideos.org/2868M) | 03:06 | Explicit ACE / game-end glitch |
| chatterbox 2nd all-items [5187M](https://tasvideos.org/5187M) | 37:55 | Heavy glitch abuse |

Game hub: [tasvideos.org/12G](https://tasvideos.org/12G). Userfiles: [UserFiles/Game/12](https://tasvideos.org/UserFiles/Game/12) (casual WIPs, subframe ACE — not default).

### Route notes (all-items)

From [submission #7565](https://tasvideos.org/7565S):

- Dungeon order roughly **3 → 4 → 1 → 8 → 2 → 5 → 7 → 6 → 9** (with OW shops / gambling / heart containers).
- Soft resets used for routing (empty hearts after recorder, revisit rooms).
- Secret entrances save stairs time (same idea as any%, lighter use).
- Recorder warping considered and **not** fully used (key/route cost).
- Item definition: best version of each upgrade (magical sword replaces white, etc.).

Our STATUS path is **sword → L1 TF → sequential later levels**. All-items is a
**room/combat + item-routing oracle**, not a drop-in full seed.

## ROM

| | SHA-1 | Movie match |
|--|-------|-------------|
| **Our** `rom.nes` | `3701381A…` = USA **PRG1** | all-items #4767M ✓ |
| TASeditor all-items #2508M | USA **PRG0** | secondary |

## Adaptation pipeline

```text
1. fetch_refs          →  tas/ref/*allitems*.fm2  (no --include-glitched)
2. import_fm2 summary  →  frame count / first input
3. export nes9_rle     →  models/*_raw.json (optional continuous seed)
4. Split at mode/screen transitions (title, OW, dungeon rooms)
5. Replay segments from our natural predecessor states
6. Promote only when Clean natural-entry specs pass (STATUS gate)
```

### Commands

```bash
uv run python -m zelda_i.tas.fetch_refs
uv run python -m zelda_i.tas.import_fm2 --summary-only

uv run python -m zelda_i.tas.import_fm2 \
  nes/zelda_i/tas/ref/chatterbox_allitems_4767M.fm2 \
  --out nes/zelda_i/models/zelda_allitems_raw.json \
  --route-id zelda_chatterbox_allitems

# Only if you need glitched any% / swordless comparison:
# uv run python -m zelda_i.tas.fetch_refs --include-glitched
```

## Layout

```text
nes/zelda_i/tas/
  ref/           # vendored .fm2 (gitignored)
  fm2.py         # parse + nes9_rle (reuses smb.tas.fm2)
  fetch_refs.py  # default = non-glitch only
  import_fm2.py
  README.md
```
