# Clean playbook — TMNT IV (do not relearn)

Permanent lessons from Stage 1 pizza-only Clean (2026-07-27). Apply these
when porting Clean (zero emergency HP, zero form-2 iframe writes) to later
stages. **Do not re-discover these via thrash.**

## Definition of Clean

| Allowed | Forbidden |
|---------|-----------|
| Natural pizza pickup (`char 0x30`, controller Y) | Emergency HP RAM writes |
| Better spacing / standoff / jump-slash | Form-2 iframe RAM hold |
| Path-flexible policy (live state) | A-special (HP drain) |
| Checkpoint **and** power-on / mid-entry proofs | Labeling assisted runs as Clean |

Proof tool pattern: `heal=none` probe + multi-entry suite (checkpoint +
natural/power-on entry). Stage 1 reference:

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_stage1_clean --suite
```

## Verified Stage 1 results (2× suite)

| Entry | Frames | Damage | Min HP |
|-------|--------|--------|--------|
| `Stage1` | 15,237 | 108 | 30 |
| power-on | 15,046 | 138 | 10 |
| Baxter | 5,323 | 40 | 44 |

## Hard rules (apply every stage)

1. **Pizza is the only heal.** Prefer aggressive seek when HP is missing
   **only** on stages where boxes are reachable. Stage 0 is safe;
   **global** pizza seek soft-locked Skull & Crossbones (unreachable slot).
   Scope seek by `state.stage` (or a allowlist).

2. **No empty-screen attack spam.** `pickup_every=0` on walk. PizzaSeek
   owns real boxes; blind `RIGHT+Y` is visual junk and can desync timing.

3. **Do not re-enable Stage 1 wrecking-ball jump-dodge without A/B.**
   Jump-through (`B+RIGHT` into column) caused Clean mid-wave deaths.
   Offline dodge + pizza survived path RNG better. `HazardAvoid` exists
   for tests only until a **phase-safe** dodge beats offline in suite.

4. **Bosses: force a safe flank, never walk into the body.**
   Baxter: left standoff, HP-adaptive width/cadence, jump-slash only when
   elevated and HP not critical. Duo bosses already use left-flank poke —
   keep stall-suppress while they are live.

5. **Path RNG = different entries, same policy.** Always prove:
   - fight-ready checkpoint
   - at least one natural/power-on or continuous-faithful entry  
   Mid-wave historical locks (`*_Clear_w*`) can use different spawn tables;
   optional, not required for Clean gate.

6. **Checkpoint gains often do not transfer.** Stage 2/3 damage cuts
   needed full dry-runs. Any mid-route timing change → full dry-run before
   production claim. Prefer `RaphFullHard*` for continuous work (char 8).

7. **Jump-slash is situational.** Use for elevated/true-air, stacked Foot,
   dinos, hover Foot. **Do not** jump-slash: Slash shell, Rat King long
   poke, Super Shredder form 2, Mode-7 depth (Y is depth not height).

8. **Never use A-special.** Y attack, B jump only.

## Stage Clean rollout (order)

Roll Clean **one stage at a time** with pizza-only probes. Keep production
emergency assist until that stage’s multi-entry suite is green, then remove
assists for that stage byte only (or whole-run when all green).

| Order | Stage byte | Name | Clean focus (reuse Stage 1 lessons) |
|-------|------------|------|-------------------------------------|
| 0 | 0 | Big Apple | **Done** — pizza seek, Baxter standoff, hazard dodge offline |
| 1 | 1 | Alleycat Blues | **In progress** — left flank + between-wave pizza; Metalhead Clean OK; early waves still die |
| 2 | 2 | Sewer Surfin' | **In progress** — stall thrash offline; 0x1C spike jump; LiveHard entry (lives=2); Rat King HP≥1 boss_active |
| 3 | 3 | Technodrome | Duo left-flank; tank throws; blocker Foot; no stall override on duo |
| 4 | 4 | Prehistoric | Slash hybrid (spin 52 production); dino B+Y; no jump-slash on Slash |
| 5 | 5 | Skull & Crossbones | **No global pizza seek**; duo left-flank |
| 6 | 6 | Wounded Knee | Stacked `0xb0` jump-slash; Raph cadence; stall Y-quantize |
| 7 | 7 | Neon Night Riders | Near-band only `y≥140`; Krang left-flank; Mode-7 props filtered |
| 8 | 8–9 | Starbase / Shredder | Hover jump-slash; form-2 wall dodge **without** iframe write |

## Probe recipe (copy per stage)

1. Build `heal=none` segment probe from fight-ready state (like
   `probe_stage1_clean.py`).
2. Suite: checkpoint + continuous-faithful or power-on/natural entry.
3. Metrics: frames, damage, min HP, pizza heal count, emergency heals
   (must be 0), life losses (must be 0).
4. If a “smart” dodge/timing helps checkpoint but kills suite → **reject**.
5. Update `STATUS.md` with suite table; only then touch whole-run assist.

## Anti-patterns (already burned)

| Anti-pattern | What happened |
|--------------|----------------|
| Jump-through wrecking ball always | Stage1 Clean mid-wave death |
| Aggressive hazard LEFT thrash | Dumpster soft-lock forever |
| Always jump-slash Baxter | Faster boss, higher damage, continuous death |
| Global pizza seek all stages | Skull & Crossbones soft-lock |
| Alleycat **mid-wave** pizza chase | Stage2 emergency 190→479 dmg; still Clean death |
| Generic elev≥44 jump on Alleycat | Stage2 emergency ~443 dmg (false air reads) |
| Alleycat pack jump-hop thrash | Stage2 emergency 183→483 dmg |
| Sewer dumpster / WalkProgress thrash | Auto-scroll freezes X; UP/DOWN → 4×16 spike hits |
| Sewer spike LEFT thrash | 4 residual −16 hits; jump-right is better |
| Stage3/Boss3 last-life Clean gate | Dies on 0x0B fade after kill; use LiveHard (lives=2) |
| Rat King boss_active HP ≥ 4 only | Abandons finishers at HP 1–3 |
| Blind walk `RIGHT+Y` every N frames | Stutter spam; no real pizza benefit |
| Port Slash spin=40 from probe | Continuous +807 total damage |
| Mid-run knob without full dry-run | Checkpoint win, route loss |

## Code map

| Concern | Where |
|---------|--------|
| Pizza seek (stage 0 full; stage 1 underfoot + between-wave) | `policy.py` `PizzaSeek` |
| Alleycat left flank / standoff 36 | `policy.py` `_ALLEY_*` + `PreferredFlank.LEFT` |
| Baxter Clean standoff | `policy.py` `BaxterTactics` |
| Hazard helper (tests only) | `policy.py` `HazardAvoid` (not in production tick) |
| Elevated jump (stage 0 only) | `policy.py` `_elevated_jump_slash` / `_suppress_elevated_jump` |
| Stage 1 Clean suite | `scripts/probe_stage1_clean.py` |
| Stage 2 Clean suite | `scripts/probe_stage2_clean.py` |
| Stage 3 Clean suite | `scripts/probe_stage3_clean.py` (prefer `LiveHardStage3`) |
| Sewer spike dodge | `policy.py` `SewerSpikeAvoid` |
| Continuous assist | `scripts/record_full_hard_run.py` (emergency + form-2) |
| Assist contract | `docs/ASSIST_CONTRACT.md` |
