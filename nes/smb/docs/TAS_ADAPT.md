# TAS / RTA-rules adaptation — Super Mario Bros.

**Strategy shift (2026-08-07):** stop treating isolated window hill-climb as
the primary claw-back path. Public WR / TAS button streams already encode
frame-perfect route structure (FPG, BBG, fast accel, 4-2 top-clip, framerule
edges). Import → verify → segment-adapt those movies; use local search only to
heal emulator desync or residual 8-4 frames.

Keep prior work (natural_82 21,559f, 8-1 polish −42f) as baselines. Do not
delete seeds.

## Gap (why hill-climb is the wrong main loop)

| Contract | Ours (natural_82) | HappyLee #1715 | Δ |
|----------|-------------------:|---------------:|--:|
| Power-on → axe | **21,925f** (~06:04.8) | **17,868f** (04:57.31) | **~+4,057f / ~67s** |
| RTA control → axe | **21,559f** (~05:58.7) | **~04:54.032** | **~+64s** |

8-1 isolated polish (−42f) is real but noise against multi-framerule /
glitch-route debt. A single saved **framerule** is 21 frames of *level exit
phase*; missing FPG/BBG/fast-accel routes costs whole rules, not single
holds.

## Open sources to pull

### Button movies (importable)

| Source | Time | Format | Path / URL | Notes |
|--------|------|--------|------------|-------|
| **HappyLee warps** (published) | 04:57.31 power-on / 04:54.032 RTA | `.fm2` | vendored `tas/ref/happylee_warps_1715M.fm2` · [tasvideos.org/1715M](https://tasvideos.org/1715M) | **Primary** console-verified warps TAS. 17,868 frames. **85× L+R**. |
| flamexx warps userfile | claims ~4:54.099 RTA | `.fm2` | vendored `tas/ref/flamexx_warps_rta_4_54_099.fm2` · [UserFiles](https://tasvideos.org/UserFiles/Info/638803295557458382) | Newer community strategies (8-2 rule, 8-4 polish). Longer file (~18,392f) — different pad/timing; compare carefully. |
| HappyLee #1715 encode | same | video | Archive.org / TASVideos YT | Visual route reference only. |
| RTA-rules / IL practice | — | practice ROM | [github.com/pellsson/smb](https://github.com/pellsson/smb) | Frame-rule start, level select, PB tracking — not a movie, but gold for segment targets. |

### Trick / rule catalogs (route structure)

| Resource | What to take |
|----------|----------------|
| [TASVideos GameResources SMB](https://tasvideos.org/GameResources/NES/SuperMarioBros) | Walljump, **flagpole glitch**, enemy FPG / BBG, vine teleport, alternate pipes, brick clips, **L+R accel**, 21-frame rule, fireworks, lip jumps |
| [Wikipedia — SMB speedrunning](https://en.wikipedia.org/wiki/Super_Mario_Bros._speedrunning) | RTA history, fast acceleration, BBG, human limit vs TAS |
| [negative-seven smb1explained](https://negative-seven.github.io/smb1explained/) | Engine / timing depth |
| [simplistic6502 smbpedia movement](https://simplistic6502.github.io/smb1_tll/smbpedia_movement.html) | Movement physics |
| [speedrun.com smb1 guides](https://www.speedrun.com/smb1) | Human any% route notes (pl8-1, KosmicZ BBG, 8-3 FPG) |
| [8-4 frame spreadsheet](https://docs.google.com/spreadsheets/d/1hv47h27sQtboqzvjNwhB0nDgFcihpWRei-aA8lTiejs) | Named 8-4 exit frames (community) |

### 21-frame rule (critical)

Level blackouts wait so progress rounds up to a **21-frame** boundary. Saving
1–20 frames mid-level often **does nothing** to total time; saving enough to
cross a rule boundary saves **21f** (or multiple). Exception: **8-4** (axe
ends timing; rules do not apply the same way). Optimization objective for
1-1…8-3 should be **framerule class**, not raw frame count alone.

## Any% warps route structure (what HappyLee / modern RTA encode)

High-level exits (not our coarser RAM splits):

1. **1-1** — full speed; optional FPG / top-step variants in modern RTA
2. **1-2** — underground → **warp zone World 4** (not W5/W8 wrong-pipe)
3. **4-1** — overworld; FPG variants on modern RTA
4. **4-2** — vine / **top-clip / alternate pipe** into **World 8** warp
5. **8-1** — **fast acceleration** (backwards accel) + clean run; modern
   **pl8-1** FPG saves a framerule vs casual full-speed
6. **8-2** — **Bullet Bill Glitch (BBG / KosmicZ)** multi-rule save when
   enemies line up; else slower flag
7. **8-3** — FPG / fireworks control (timer digit 1/3/6 → fireworks)
8. **8-4** — maze + Bowser + axe; **frame-level** optimization free of
   framerule bus (last true continuous claw-back)

Our current seed clears the same **world path** (W4→W8 warps) but is not
framerule-optimal and does not systematically encode FPG/BBG/fast-accel.

## Adaptation pipeline (new main loop)

```text
1. Import FM2  →  nes9_rle  (smb.scripts.import_fm2)
2. Power-on verify under stable-retro (no L+R sanitize)
3. If desync: phase-align boot (Select vs Start, RAM init) or
   segment-split at natural control and retime like reactive_warp
4. Split movie at level exits → control-relative bodies
5. Replace our natural_82 bodies level-by-level when TAS segment wins
6. Only then: residual polish (8-4, desync heal) — not global hill-climb
```

### Commands

```bash
# Summary of vendored HappyLee movie
uv run python -m smb.scripts.import_fm2 --summary-only

# Write raw continuous seed (includes title/boot frames)
uv run python -m smb.scripts.import_fm2 \
  nes/smb/tas/ref/happylee_warps_1715M.fm2 \
  --out nes/smb/models/smb_happylee_warps_raw.json \
  --route-id smb_happylee_warps

# Verify playback (expect ending ≈ 17868 if fully syncs)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.import_fm2 \
  nes/smb/tas/ref/happylee_warps_1715M.fm2 --verify
```

### Hard rules when adapting

1. **Preserve Left+Right.** `sanitize_action` zeroes L+R; TAS uses it for
   instant accel / brake. Replay paths must skip directional sanitize.
2. **Boot inputs come from the movie** (HappyLee uses **Select** ~frame 41),
   not necessarily our START spam boot (350+16). Compare both.
3. **Framerule objective** on 1-1…8-3; raw frames on 8-4.
4. **Do not absolute-stitch** TAS mid-movie into natural_82 without
   control-relative retime (same trap as 1-2 polish).
5. **Keep old seeds** (`smb_1_1_to_ending_natural_82.json`,
   `smb_8_1_control_best.json`, unpromoted 8-1 polish).

## Verified results (2026-08-07)

### Power-on full movie

**Does not sync** on `SuperMarioBros-Nes-v0` / fceumm. Blackout after Start is
longer than FCEUX; TAS inputs from ~196 land on load screen; death ~f4082 at
x≈844. Constant `--skip-movie` / `--pad-before` cannot recover full power-on.

### Isolated Level1_1 slice (works)

| Seed | Isolated clear | vs `smb_1_1_tas_best` |
|------|---------------:|----------------------:|
| Our hill-climb `smb_1_1_tas_best.json` | ~1903 leave / 1924 len | baseline |
| **HappyLee slice** `smb_1_1_happylee_slice.json` | **1733f** | **≈ −170–190f** |

Recipe: `Level1_1` + settle **2** + FM2 body from index **190** (even indices
176–196 also clear; **odd indices die** — hitbox parity). Artifact written by
import/search session.

```bash
# 1-1 only (verified path)
uv run python -m smb.scripts.tas_1_1 verify \
  --seed nes/smb/models/smb_1_1_happylee_slice.json
```

### Natural-entry 1-1 (works)

| Settle after boot ready | Outcome | Seed frames to clear |
|------------------------:|---------|---------------------:|
| even (0,2,4,…) | death @ first pit | — |
| **odd (1,3,5,…)** | **success** | **1749** (settle=1) … 1739 (settle=11) |

Default `NATURAL_SETTLE_FRAMES=1` matches our other natural seeds. Evidence:
`recordings/tas_import/happylee_1_1_natural_settle_search.json`.

```bash
uv run python smb/scripts/run_1_1.py --natural-entry --settle 1 \
  --seed nes/smb/models/smb_1_1_happylee_slice.json
```

**Do not** absolute-stitch this 1-1 into the old continuous 1-2 body — phase
mismatch sends the route to 1-3. Use control-relative 1-2 (below).

### Control-relative 1-2 W4 slice (works)

Predecessor: Level1_1 + HL 1-1 (settle 2) + idle until `is_surface_control`
(~165f). Then FM2 body:

| Field | Value |
|-------|------:|
| FM2 start index | **2109** (odd; matches odd ctrl_wait parity) |
| W4 frames | **1657** |
| UG enter | ~334 |
| Seed | `models/smb_1_2_happylee_slice.json` |

| Path to W4 | Frames | Δ vs natural_82 |
|------------|-------:|----------------:|
| natural_82 (1-1 1911 + 1-2 1973) | **3884** | — |
| HL 1-1 + ctrl wait + HL 1-2 (1733+165+1657) | **≈3555** | **≈ −329f** |
| HL 1-2 body alone vs our 1973 | **1657** | **≈ −316f** |

```bash
# Verify natural chain HL 1-1 → surface → FM2 → W4
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.import_fm2 --verify-1-2-slice
# Re-export / re-search if 1-1 body changes
uv run python -m smb.scripts.import_fm2 --export-1-2-slice
# uv run python -m smb.scripts.import_fm2 --search-1-2 --1-2-start-min 2080 --1-2-start-max 2140
```

Helpers: `smb.tas.slice` (`reach_surface_after_hl_1_1`, `search_1_2_offsets`,
`verify_1_2_natural_chain`). Evidence:
`recordings/tas_import/happylee_1_2_slice_verify.json`.

Save-state probes can drift one frame vs a full rebuild — always re-verify
with a fresh Level1_1 chain before promoting indices.

### Control-relative 4-1 + 4-2 → W8 (works)

Predecessor: HL chain to W4, then idle to ``is_4_1_control`` (wait **214**,
even). Then FM2 bodies:

| Field | 4-1 | 4-2 |
|-------|----:|----:|
| FM2 start index | **3968** (even) | **6207** (odd) |
| Body frames | **2062** → 4-2 load | **1516** → W8 |
| Gate wait after prior | 214 (even) | 165 (odd; timer often **0**) |
| Seed | `models/smb_4_1_happylee_slice.json` | `models/smb_4_2_happylee_slice.json` |

| Path | Frames | Δ vs natural_82 |
|------|-------:|----------------:|
| natural_82 to 4-1 exit | **6198** | — |
| HL chain to 4-2 load | **≈5831** | **≈ −367f** |
| natural_82 to 8-1 entry | **12628** | — |
| HL chain to W8 | **≈7512** | **≈ −5116f** |
| 4-1 body alone vs 2314 | **2062** | **≈ −252f** |
| 4-2 (wait+body) vs 2764 | **165+1516=1681** | **≈ −1083f** |

```bash
# Verify HL 1-1 → 1-2 W4 → 4-1 → 4-2 → W8 (fresh rebuild)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.import_fm2 --verify-4-1-4-2-slice
uv run python -m smb.scripts.import_fm2 --export-4-1-slice --export-4-2-slice

# MP4 of the same verified chain (HUD + audio; same writer as any%)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.record_happylee --to w8
# → recordings/tas_import/happylee_w8.mp4  (~2:05 chain clock; sub-5 min)
# uv run python -m smb.scripts.record_happylee --to w4   # stop at W4
# uv run python -m smb.scripts.record_happylee --to 1-1  # isolated 1-1
```

Helpers: ``is_4_1_control`` / ``is_4_2_control`` / ``verify_4_1_4_2_natural_chain``
in ``smb.tas.slice``; recorder ``smb.scripts.record_happylee``. Evidence:
``recordings/tas_import/happylee_4_1_4_2_slice_verify.json``,
``recordings/tas_import/happylee_w8.mp4`` (when recorded).

**4-2 gate:** do **not** require ``timer > 0`` — first control matches natural
fingerprints with timer 0. Absolute continuous FM2 from 4-1 control also
clears W8 @3755f (includes score/load); prefer split + control gates.

**4-2 path note:** HL 4-2 is **not** our natural vine path. Probe exits:
surface → area ``(3,2)`` @**334f** → World 8 @**1516f** (x≈810). That is the
warps top-clip / alternate-pipe structure. Before promoting a full-run fold,
validate against TASVideos encode **and** RAM (world/level/area_pointer,
player_x through the clip) — different geometry than natural_82 4-2.

### Stage frame board (exit-detect, Level1_1-relative)

Same contract as STATUS RTA splits: cum = first frame post-exit world/level
RAM (or W4/W8 entry / stage load). Seg times @ NTSC 60.0988….

| Exit / gate | natural_82 cum | Seg | HL cum (probe) | HL body / wait | Δ cum vs n82 | Status |
|-------------|----------------:|----:|---------------:|---------------:|-------------:|--------|
| 1-1 leave | 1911 | 1911 | **1733** | 1733 body | **−178** | **done** seed |
| 1-2 → W4 | 3884 | 1973 | **3555** | wait165 + 1657 | **−329** | **done** seed |
| 4-1 leave | 6198 | 2314 | **5831** | wait214 + 2062 | **−367** | **done** seed |
| 4-2 → W8 | 8962 | 2764 | **7512** | wait165 + 1516 | **−1450** | **done** seed |
| 8-1 leave | 12628 | 3666 | **≈10602** | wait209 + **2881** | **≈−2026** | probe only |
| 8-2 leave | 15779 | 3151 | **≈12976** | wait165 + **2209** | **≈−2803** | probe only |
| 8-3 leave | 17985 | 2206 | — | nat bridge | — | **bridge** (pure HL open) |
| 8-4 axe | **21559** | 3574 | **≈18031** hybrid v2 | FX 2661 | **≈−3528** | **FX 8-4 done** |

HL body-only vs natural_82 segment (control-relative, excluding wait):

| Stage | n82 seg | HL body | Δ body | FM2 start | Notes |
|-------|--------:|--------:|-------:|----------:|-------|
| 1-1 | 1911 | 1733 | −178 | 190 | settle=2 isolated |
| 1-2 | 1973 | 1657 | −316 | 2109 odd | W4 warp |
| 4-1 | 2314 | 2062 | −252 | 3968 even | |
| 4-2 | 2764 | 1516 | −1248 | 6207 odd | glitch path; +wait165 vs n82 |
| 8-1 | 3666 | **2881** | **−785** | **7930 even** | wait81=209 **odd** but **even** FM2 |
| 8-2 | 3151 | **2209** | **−942** | **10910** | BBG-class?; 8-3 phase TBD |
| 8-3 | 2206 | — / nat 2062 leave | — | — | pure HL blocked; nat bridge |
| 8-4 | 3574 | **2661 FX** | **−913** | **15210 FX** | after nat 8-3 control |

### Ours vs WR (full-run contracts — still natural_82 until fold)

| Contract | Ours (natural_82) | HappyLee #1715 | Δ |
|----------|------------------:|---------------:|--:|
| RTA control → axe | **05:58.726** (21,559f) | **04:54.032** (17,671f) | **+01:04.693** (+3,888f) |
| Power-on → axe | **06:04.816** (21,925f) | **04:57.31** (17,868f) | **+01:07.505** (+4,057f) |

Partial HL chain (no 8-3/8-4 yet), same exit-detect clock:

| Milestone | HL frames | NTSC | vs n82 same gate |
|-----------|----------:|------|-----------------:|
| to W4 | 3555 | 00:59.153 | −329 |
| to W8 | 7512 | 02:04.994 | −1450 (vs 8962) / −5116 (vs 8-1@12628) |
| to 8-2 load | ≈10602 | 02:56.409 | ≈−2026 |
| to 8-3 load | ≈12976 | 03:35.911 | ≈−2803 |
| ending | — | — | need 8-3 + 8-4 |

If 8-3/8-4 later match HL movie length after 8-2, projected RTA would still
need a fresh full-chain verify (do not invent an ending from partial sums).

### World 8 + hybrid ending (2026-08-07)

Predecessor: verified HL chain to W8 @**7512**. Idle **209** → 8-1 control.

| Field | 8-1 | 8-2 |
|-------|----:|----:|
| FM2 start | **7930** (even) | **10910** |
| Leave frames | **2881** → 8-2 load | **2209** → 8-3 load |
| Gate wait | 209 (odd) | 165 |
| Seed | `smb_8_1_happylee_slice.json` | `smb_8_2_happylee_slice.json` |

```bash
uv run python -m smb.scripts.import_fm2 --verify-8-1-8-2-slice
uv run python -m smb.scripts.import_fm2 --export-8-1-slice --export-8-2-slice
```

Evidence: ``happylee_8_1_8_2_slice_verify.json`` (to 8-3 load **12976**).

**8-1 parity:** wait81 **odd** but **even** FM2 starts clear — search both.

**8-3 pure continuous FM2:** still phase-blocked (gated raw max_x≈1030;
continuous dies early). **Stitchless leave (skills path, 2026-08-07):**
HL 8-2 control → progress-healed body prefix (max_x 3390) → land-pin skill
(cut1478 + hop jh=44/gap=6/hops=3) + idle fold → **8-4 control 2374f 2/2**.
No natural_82 mid-splice. Seed: `smb_8_3_stitchless_skills_leave.json`.
Evidence: `happylee_8_3_skills_leave.json`. CLI: `smb.scripts.stitchless_8_3`.

**8-4 TAS after natural 8-3 (works 3/3):** after HL→8-2 + wait83 + natural
bridge to ``is_8_4_control`` (nat@15933 for **2227f**), **flamexx** FM2
start **15210** clears axe in **2661f** (HL alt 15034/2833). Replaces
natural 8-4 (~3400f).

| Contract | Frames | NTSC | vs n82 | vs sub-5 (18030f) |
|----------|-------:|------|-------:|------------------:|
| Hybrid v1 (nat 8-3/8-4) | 18769 | 5:12.302 | −2790 | +739 |
| **Hybrid v2 (nat 8-3 + FX 8-4)** | **18031** | **5:00.023** | **−3528** | **+1** |
| Seed v2 | `smb_happylee_hybrid_v2_fx84.json` | | | |
| 8-4 slice | `smb_8_4_flamexx_slice.json` | | | |
| 8-3 bridge | `smb_8_3_natural_for_hl_hybrid.json` | | | |
| Evidence | `happylee_hybrid_v2_fx84_verify.json` (3/3) | | | |

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m smb.scripts.record_happylee --to ending
```

**Sub-5 path:** pure HL 8-3 (or −1f on v2) for ≤18030; full HL 8-3/8-4
projects ~**17.9k / ~4:57** if phase matches. v2 is the full-clear showcase
(not Clean power-on; not WR).

## Expected outcomes (updated)

| Outcome | Meaning | Next |
|---------|---------|------|
| Isolated / natural 1-1 faster | **Done** (−170f class) | Keep; fold later |
| Control-relative 1-2 → W4 | **Done** (−316f body / −329f to W4) | Keep |
| Control-relative 4-1 / 4-2 → W8 | **Done** (−252f 4-1 / −1083f 4-2 / −5116f to W8) | Validate 4-2 path video+RAM |
| 8-1 / 8-2 leave probe | **Done** (2881 / 2209 exported) | Keep |
| 8-3 stitchless leave | **Done** 2374f 2/2 (skills) | Fold full chain + Clean |
| 8-3 pure continuous FM2 | **Open** (phase-blocked) | Secondary |
| 8-4 TAS body | **Done** (FX 2661 after nat 8-3) | Attach after skills 8-3 |
| Power-on full movie dies early | Core lag/blackout ≠ FCEUX | Don't force full FM2 |
| Absolute stitch → 1-3 | Warp pipe phase miss | Always gate on control |
| Parity (even/odd starts) | Hitbox / enemy phase | Prefer match wait; **search both** if miss |

## Mapping notes (FM2 → nes9)

| FM2 char | Meaning | NES index |
|----------|---------|-----------|
| R | Right | 7 |
| L | Left | 6 |
| D | Down | 5 |
| U | Up | 4 |
| **T** | **Start** (sTart) | 3 |
| **S** | **Select** | 2 |
| B | B | 0 |
| A | A | 8 |

HappyLee’s first non-idle input is **Start** (~frame 41), not Select.

Index **1** is the stable-retro NES hole (always 0).

## Relation to beads / prior work

- Epic **rr-asa**: TAS adapt HappyLee per-level (primary claw-back).
  - **rr-nwl**: natural-entry 1-1 HappyLee — **done** (settle odd / default 1).
  - **rr-9m9**: slice FM2 per-level from control — **1-2 W4 done**.
  - **rr-zzw**: slice 4-1 / 4-2 from control after HL W4 — **done** (→ W8 @7512f).
  - **rr-b8k**: slice 8-1…8-4 — **in progress** (8-1/8-2 exported; **8-3
    stitchless leave 2374f** via skills; 8-4 FX body exists; full fold open).
  - **rr-34v (stitchless 8-3):** leave **done 2/2** at **2374f**
    (`smb_8_3_stitchless_skills_leave.json`) after HL 8-2 control — progress
    prefix + pure hop/flagpole skill + idle fold; **no natural_82 splice**.
    Rich handoff FP + `stitchless_8_3` / `skills_8_3`. Grounded/pit-jump
    grids paused. **Next:** fold continuous HL…8-2 + skills 8-3 + FX 8-4;
    Clean power-on 3/3; optional 21f/FPG polish. Hybrid v2 18031 showcase only.
  - **rr-k96**: FPG/BBG/fast-accel named macros — open (encode after more slices).
- **rr-9dg** (8-1 polish −42f): keep artifacts; secondary to TAS structure.
- 8-2/8-3/8-4 hill-climb (rr-7n0, rr-yqb, rr-6m9) remains secondary until
  HappyLee bodies cover W8 legs.
- STATUS promote only after Clean power-on 3/3 on a full adapted seed.

## References

- TASVideos #1715: https://tasvideos.org/1715M
- Game resources: https://tasvideos.org/GameResources/NES/SuperMarioBros
- Submission notes: https://tasvideos.org/2964S
- Practice ROM: https://github.com/pellsson/smb
- Our timing contracts: `smb/timing.py` (`happylee_warps_tasvideos`, RTA note)
