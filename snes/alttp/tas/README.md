# ALttP TAS import

HappyLee/Sniq-style **button-press movies** for *A Link to the Past*,
vendored under `ref/` and sliced into `snes12_rle` JSON under `slices/`.
Catalog: [`catalog.py`](catalog.py) (TASVideos
[game 191](https://tasvideos.org/191G) +
[userfiles](https://tasvideos.org/UserFiles/Game/191)).

This tree only vendors inputs. Castle / sewers / sanctuary cuts are
**stubs** — listed so a later sitting can measure windows. They do not
play, and exporting them is refused.

GitHub had no extra machine-readable any% `.bk2` / `.lsmv` / `.smv`
beyond TASVideos.

## Ref movies

Unwrapped SHA-256 is in `catalog.py`. Files are gitignored; re-fetch.

| File | Source | Frames | Format | On disk |
|------|--------|-------:|--------|---------|
| `ref/fmp_geg_3898M.lsmv` | [TASVideos #3898M](https://tasvideos.org/3898M) fmp/total/Yuzuhara_3 GEG | 3 277 | lsnes LSMV | vendored (~4.6 KB) |
| `ref/m_riss_100_wip_10k.bk2` | [Userfile](https://tasvideos.org/UserFiles/Info/639202560055329291) m_riss 100% WIP | 10 000 | BizHawk BK2 | vendored (~3.8 KB) |
| `ref/taseditor_glitched.lsmv` | [Userfile](https://tasvideos.org/UserFiles/Info/8710388382096206) TASeditor glitched any% | 8 127 | LSMV | vendored (~2.6 KB) |
| `ref/fmp_nmg.bk2` | [Userfile](https://tasvideos.org/UserFiles/Info/67832544397586850) fmp NMG | 113 135 | BK2 | fetch (~69 KB) |
| `ref/tompa_1269M.smv` | [TASVideos #1269M](https://tasvideos.org/1269M) Tompa USA no-major-glitch | 274 264 | Snes9x SMV | fetch (~549 KB) |
| `ref/fmp_yuzuhara_fullinv_3874M.bk2` | [TASVideos #3874M](https://tasvideos.org/3874M) full inventory | 190 660 | BK2 | fetch (~45 KB) |

GEG / glitched / NMG / full-inventory are **JP 1.0**
(`SHA1 E7E852F0…`). WIP and Tompa are **USA**
(`SHA1 6D4F10A8…`, workspace ROM). This game’s opening spine is USA —
do not treat JP glitch movies as sanctuary-route oracles.

Skipped (not movies): RAM watch `.wch`, `minimap.lua`, `maptracker.lua`.

## Formats + button order

Env SNES-12 (stable-retro / `retro_harness.controls.SNES_BUTTON_NAMES`):

`[B, Y, Select, Start, Up, Down, Left, Right, A, X, L, R]`

| Format | Spec | P1 field |
|--------|------|----------|
| LSMV | [lsnes LSMV](https://tasvideos.org/EmulatorResources/Lsnes/LSMV) | `F.\|BYsSudlrAXLR` — same order as env. #3898M uses `ygamepad16` (16-char P1 + extra ports); parser keeps the first 12. |
| BK2 | [BizHawk BK2](https://tasvideos.org/Bizhawk/BK2Format) | `Input Log.txt` + `LogKey`. Observed ALttP LogKey: `#Reset\|Power\|#P1 Up\|Down\|Left\|Right\|Select\|Start\|Y\|B\|X\|A\|L\|R\|`. Parser maps LogKey → env. |
| SMV | Snes9x binary `SMV\x1a` | 12-bit words in BizHawk SmvImport order `Right Left Down Up Start Select Y B R L X A`, then mapped to env. |

`snes12_rle` JSON (same as Super Metroid):

```json
{
  "format": "snes12_rle",
  "game_name": "Zelda3-Snes",
  "route_id": "m_riss_100_wip_menu",
  "num_frames": 1200,
  "segments": [{"n": 958, "b": []}, {"n": 1, "b": ["START"]}]
}
```

Parsers are thin wraps of `retro_harness.bk2` and `super_metroid.tas.{lsmv,smv,rle}`.

## Commands

```bash
uv run python -m alttp.tas.fetch_refs --list --skipped
uv run python -m alttp.tas.fetch_refs
uv run python -m alttp.tas.export_slices --list
uv run python -m alttp.tas.export_slices --verified
uv run pytest alttp/tests/test_tas.py -q
```

## Intended slices (stubs)

| Id | Intent | Status |
|----|--------|--------|
| `castle` | house → courtyard → secret entrance → main hall | unverified window; not exported |
| `sewers` | B1 lamp / keys / Zelda cell | unverified window; not exported |
| `sanctuary` | escort 0x50 → Sanctuary | unverified window; not exported |

Verified exports (when refs are present): `fmp_geg_full`,
`m_riss_100_wip_full`, `m_riss_100_wip_menu`, `taseditor_glitched_full`.

## Layout

```
tas/
  catalog.py        # TASVideos game 191 fetch list
  fetch_refs.py
  lsmv.py / bk2.py / smv.py
  rle.py            # snes12_rle, game_name=Zelda3-Snes
  slice.py          # named windows + stubs
  export_slices.py
  ref/              # gitignored movies
  slices/           # generated snes12_rle JSON
```
