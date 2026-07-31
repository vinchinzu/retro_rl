# SM-TIGHTEN-06 Report — Bomb Torizo fight split dwell analysis

**Date:** 2026-07-31
**Source:** `recordings/start_to_kraid.json` (outcome=`kraid_entry`, total_frames=97,139)
**Model:** Flash (report-only; no controller/continuous/STATUS edits)

---

## 1. Commands

```bash
uv run python super_metroid/scripts/export/split_dwell.py \
  super_metroid/recordings/start_to_kraid.json --top 15
uv run python super_metroid/scripts/export/split_dwell.py \
  super_metroid/recordings/start_to_kraid.json --reasons --top 40
```

## 2. Split data

| Split | Room | Frame | Dwell | % of total |
|-------|------|------:|------:|-----------:|
| `bombs` | 0x9804 (Bomb Torizo) | 41,502 | — | — |
| `bomb_torizo_defeated` | 0x9804 | 43,561 | 2,059f | 2.1% |
| `bomb_torizo_exit` | 0x9879 | 45,279 | 1,718f | 1.8% |
| **Total bomb torizo window** | | | **3,777f** | 3.9% |

Dwell rank: `bombs` is #2 overall (11,812f, includes the entire pit→torizo→post-torizo segment), `bomb_torizo_defeated` is #11 (2,059f), `bomb_torizo_exit` is #13 (1,718f).

**From `start_to_bomb_torizo.json` (standalone recording):** `bombs` @42,672 → `defeated` @44,524 = 1,852f, → `exit` @46,449 = 1,925f. Total window: 3,777f (same).

## 3. Controller ownership

The bomb torizo fight is **entirely inside a monolithic recorded policy**:

| Segment | File | Frames | Type |
|---------|------|-------:|------|
| `pit_to_post_torizo` | `policies/early_game/pit_to_post_torizo.json` | 13,143 | raw_buttons (PolicySegment) |
| Pre-fight (Pit 0x975C → bomb tunnel → 0x9804 entry) | inside same policy | 8,683 | raw_buttons |
| Fight (bombs collected → defeat) | inside same policy | 2,059 | raw_buttons |
| Exit (defeat → 0x9879) | inside same policy | 1,718 | raw_buttons |

The structured strategy `play_bomb_torizo_fight` in `combat/bomb_torizo.py` uses `fight_bomb_torizo_action` with `fight_bomb_torizo` / `fight_bomb_torizo_idle` reason labels — but **is not used in the continuous route**. The continuous route (`play_start_to_bombs` in `routes/continuous.py:565`) calls `play_policy(session, _PIT_TO_POST_TORIZO)` — a single monolithic recording.

The policy metadata shows this is a **legacy manual replay** (`source_slice: "[20:] after Pit Room grounded settle"`, `provenance: "legacy manual replay; acceptance requires natural-entry replay"`). A climb splice was already applied (cut 1,170f of repeated left-wall fall loops) reducing the climb dwell from 4,339f→3,339f.

## 4. Policy idle analysis

### Overall policy (13,143 frames)

| Metric | Value |
|--------|------:|
| Total frames | 13,143 |
| Zero-button (idle) frames | 3,960 (30.1%) |
| Max idle run | 221f at frame 12,922 (near end of segment) |
| Max identical navigation frames | 399 (segment metadata) |

### Per-phase breakdown

| Phase | Policy frames | Duration | Idle frames | Idle % | Max idle run |
|-------|:------------:|---------:|:----------:|:------:|:------------:|
| Pre-fight (0 → 8,683) | 8,683 | 0→8,683 | 2,828 | 32.6% | — |
| **Fight (8,683 → 10,742)** | **2,059** | **bombs→defeated** | **419** | **20.3%** | **80f** |
| Exit (10,742 → 12,460) | 1,718 | defeated→exit 0x9879 | 334 | 19.4% | — |
| Remainder (12,460 → 13,143) | 683 | post-exit settle | — | — | — |

### Fight button distribution (2,059f)

| Button | Count | % of frames |
|--------|------:|:-----------:|
| R (run) | 902 | 43.8% |
| X (fire) | 588 | 28.6% |
| RIGHT | 360 | 17.5% |
| LEFT | 350 | 17.0% |
| DOWN | 339 | 16.5% |
| A (jump) | 215 | 10.4% |
| UP | 112 | 5.4% |
| B (jump) | 91 | 4.4% |
| SELECT | 22 | 1.1% |

X is pressed every ~3.5 frames on average. The structured strategy's `fire_period=2` fires every 2 frames — **~75% more often** — which would defeat the torizo faster.

## 5. Natural entry state

A natural capture state exists at `scratch/natural_bomb_torizo_active.state` (provenance: `scratch/natural_bomb_torizo_active.provenance.json`):

- Captured at frame 42,242 (continuous prefix)
- Room 0x9804, x=131, y=192, pose 39 (morph)
- Enemy HP: 800/800, spritemap 0xAA12 (active combat)
- `developmentOnly: true`

This is a valid natural-entry start point for the structured strategy.

## 6. Structured strategy vs. raw policy

The structured `play_bomb_torizo_fight` (`combat/bomb_torizo.py`) uses:
- `BombTorizoStrategy`: `min_range=70, max_range=120, jump_range=100, fire_period=2, max_fight_frames=8,000`
- Range-kiting: move toward/away based on distance
- Jump every 50 frames for 18f
- Fire every 2 frames
- `max_fight_frames=8,000` (upper bound; actual fight is typically 1,500-2,000f)

The raw policy has **20.3% idle** during the fight phase (419/2,059f), including an 80f max idle run. The structured strategy would have zero idle frames (every frame is either a movement or fire action) and would fire at 2× the rate.

## 7. Top waste analysis

### Waste band 1 — Fight idles (highest value, medium risk)

- **419 idle frames (20.3%)** in the 2,059f fight window
- 80f max idle run suggests Samus standing still while torizo is active
- **Root cause:** raw_buttons policy was recorded from a human replay; idle frames are natural human hesitation / waiting for torizo to land
- **Fix:** Replace the fight portion of the monolithic policy with the structured `play_bomb_torizo_fight` strategy
- **Speculative band:** 150-400f saved (structured strategy has no idle frames; may also defeat faster due to higher fire rate)
- **Risk:** Medium — requires splitting the monolithic policy at the exact bombs-collection frame, inserting the strategy call, then continuing with the exit portion. The exit portion must still be a raw policy (post-torizo navigation).

### Waste band 2 — Exit phase idles (medium value, low risk)

- **334 idle frames (19.4%)** in the 1,718f exit window (defeat → room 0x9879)
- Includes post-fight settle, item fanfare, door transition
- Some of the idle is unavoidable (fanfare, door transition), but the raw policy has **19.4% idle** beyond what's expected
- **Fix:** Tighten the exit portion by re-recording the exit from the natural defeat state
- **Speculative band:** 50-150f
- **Risk:** Low — exit is navigation-only (no combat), so a tighter recording or structured approach is safe

### Waste band 3 — Pre-fight idles (lowest value, high risk)

- **2,828 idle frames (32.6%)** in the 8,683f pre-fight window (Pit room → bomb tunnel → 0x9804)
- However, this includes the bomb tunnel bomb-jump wait, the Pit room navigation, etc.
- Some idle is unavoidable (bomb jumps, door transitions)
- A climb splice already removed 1,170f from this section
- **Fix:** Split the monolithic policy into separate segments (Pit→bomb tunnel, bomb tunnel→torizo entry, fight, exit)
- **Speculative band:** 200-500f beyond what the climb splice already saved
- **Risk:** High — splitting the monolithic policy is a multi-step refactor

## 8. Patch recipes (future implement cards)

### P1: Replace fight portion with structured strategy (highest value)

- **Files:** `policies/early_game/pit_to_post_torizo.json` (split), `routes/continuous.py` (insert `play_bomb_torizo_fight` call), `combat/bomb_torizo.py` (already exists)
- **Approach:**
  1. Split `pit_to_post_torizo.json` into two policies: `pit_to_bomb_torizo_entry.json` (Pit→bomb acquisition) and `post_torizo_exit.json` (defeat→0x92FD)
  2. Replace the middle section with `play_bomb_torizo_fight(session)` using the structured strategy
  3. Wire in `continuous.py:play_start_to_bombs` — call policy1, then `play_bomb_torizo_fight`, then policy2
- **Risk:** Medium — the split must be at the exact frame where bombs are collected (not mid-bomb-jump or mid-door). The natural entry state at frame 42,242 (x=131, y=192, morph pose) is a valid structural join point, but the policy's entry into 0x9804 is at a different position (the policy starts from the Pit room).
- **Speculative band:** 150-400f on the fight split alone
- **Acceptance:** `uv run python super_metroid/scripts/record/continuous.py --to bombs --no-video` integrity green; then `--to kraid --no-video` integrity green

### P2: Trim exit phase (low risk, medium value)

- **Files:** `policies/early_game/pit_to_post_torizo.json` (extract/post-process), or re-record
- **Approach:** Re-record the exit portion (0x9804 defeat → 0x9879 → 0x92FD) using a natural-entry state after defeat. The current exit is 1,718f with 19.4% idle; a tighter recording could cut 50-150f.
- **Risk:** Low (navigation-only, no combat)
- **Speculative band:** 50-150f
- **Acceptance:** As above

### P3: Full policy decomposition (highest risk, largest potential)

- **Approach:** Decompose the 13,143f monolithic policy into 3-4 controller segments:
  1. `play_pit_to_bomb_tunnel` (Pit room 0x975C → bomb tunnel entrance)
  2. `play_bomb_tunnel_to_torizo` (bomb tunnel → 0x9804 entry)
  3. `play_bomb_torizo_fight` (structured strategy)
  4. `play_post_torizo_exit` (0x9804 exit → 0x92FD)
- **Risk:** High — the pit-to-bomb-tunnel navigation is geometry-sensitive (morph ball, bomb jumps, screw attack blocks). The climb splice already shows the pre-fight portion had issues (repeated fall loops).
- **Speculative band:** 400-1,000f (from eliminating 30%+ idle across the whole segment)
- **Acceptance:** Full `--to bombs --no-video` and `--to kraid --no-video` integrity green

### Combined P1+P2 (recommended next card)

~200-550f savings on a 3,777f window (5-15% of the bomb torizo window). Implement P1 first (structured fight), then P2 (trim exit). Leave P3 (full decomposition) for a future card after the route is tightened elsewhere.

## 9. Non-claims

- No frame savings claimed without re-record.
- The `policy_pit_to_post_torizo` reason (13,143f) is **not** all "bomb torizo fight" — it includes the entire Pit room navigation, bomb tunnel, fight, and post-torizo exit. Only **2,059f** is the fight itself.
- The fight idle (419f, 20.3%) is **not all** tightenable — some idle is due to natural torizo behavior (torizo jumps, invulnerability phases, landing animation).
- The structured strategy's performance has not been verified on a full continuous recording from Pit room natural entry. The `natural_bomb_torizo_active.state` is a development-only scratch state.
- Did not edit `continuous.py`, `STATUS.md`, `combat/bomb_torizo.py`, or any policy JSON.

## 10. Acceptance

| Criterion | Status |
|-----------|--------|
| Report complete | Pass |
| Report-only diff | Pass — no code changes |
| Phase table with dwell data | Pass |
| 2-3 implement recipes with risks | Pass (P1, P2, P3) |
| No code edits outside report | Pass |

## 11. Residual risks / planner next

- **The structured strategy has never been tested on a continuous `--to bombs` or `--to kraid` run.** The `natural_bomb_torizo_active.state` uses a `play_start_to_bombs` prefix but the capture is development-only. A P1 implement card must verify the strategy defeats the torizo reliably from the natural entry position.
- **The policy split point (bombs collection frame) is inside the policy.** The split must be exact: after the bombs item-fanfare ends, before the defeat animation starts. The `progress_events` in the recording could identify the exact frame.
- **The `post_torizo_parlor_alignment` (20f) in the action_reasons shows the post-fight exit exists as a separate reason** — but it's only 20f, not the full exit. The exit is still inside the monolithic policy.

**Planner next:** Dispatch a P1 implement card (SM-K4-06-P1 or similar) that:
1. Splits `pit_to_post_torizo.json` into pre-fight + post-fight policies
2. Inserts `play_bomb_torizo_fight` call in `continuous.py` between them
3. Verifies with `--to bombs --no-video` and `--to kraid --no-video`
4. Compares the `bomb_torizo_defeated` split dwell diff vs. 2,059f