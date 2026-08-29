# Slash vulnerability map (char `0x50`)

Research-only notes for improving `SlashTactics` in `tmnt_iv/policy.py`.
**Do not treat this as a policy rewrite** — feed these rules into the next
implementer pass.

| Item | Value |
|------|--------|
| Probe script | `tmnt_iv/lab/slash_vuln.py` |
| State | `FullHardBoss5` (spawn HP **160**, stage 4, event `0x0A`) |
| Method | Production thrash (`Stage1Policy` → `SlashTactics`) + constant player HP top-up |
| Sample | 40k frames, **28** boss HP drops (160 → 32), + 20k/25k player-damage passes |
| Status byte | `EnemyState.animation` = enemy entity **base+0** |
| Player iframes | `PLAYER_BASE + OFF_IFRAMES` (`0x046E`) |

Re-run:

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.lab.slash_vuln \
  --state FullHardBoss5 --max-frames 40000 --pre-hit 20 --mode thrash
# JSON → tmnt_iv/recordings/slash_vuln_probe/FullHardBoss5_thrash.json
```

---

## Status lexicon (observed)

| Status | Role | Typical run length | Notes |
|--------|------|--------------------|--------|
| `0xEE` | Shell **spin** | short bursts ~6–8f; long spins up to ~160f | Dominant “busy” state. **Does not** deal player damage on the drop frame itself. Often precedes both punish windows and the claw string. |
| `0x40` | Post-spin / walk | ~6f | Mildly hittable (1 opener landed from `0x40`). |
| `0x3E` | **Grounded punish idle** | ~13–20f | Primary **pre-hit vulnerable** state for openers (~3–4 dmg). |
| `0x17` | **Hitstun** (single) | ~4f | Almost always the **at-hit** status for openers. |
| `0x2E` | **Multi-hitstun / juggle** | ~33f mean | At-hit for follow-ups and big hits; also pre-hit for chain Y. Keep pressing. |
| `0xB7` | **Big punish / mutual** | ~5–6f | Best DPS pre-hit (**always 8 dmg** in sample). Also chips Leo for 6 if `adx ≲ 28`. Entered via `0xEE → 0x23 → 0xB7`. |
| `0x23` | Windup into `0xB7` | exactly ~6f | Not a hit target; telegraph for the 8-dmg window. |
| `0x83` | Claw **windup** | ~6f | Always precedes damaging `0x09`. |
| `0x09` | Claw **active** | ~6f | **Main player damage** (~8/hit). `adx` 31–79 (p50 ~40). |
| `0x9F` / `0x74` / `0x42` | Post-hitstun twitches | ~4f each | Appear between combo hits; `0x74→0x2E` often precedes a follow-up connect. |
| `0xB8` / `0xB2` / `0x97` / `0xB6` | Other attack / move cycles | varies | Mostly non-punish or rare. `0xB6` can precede `0xEE→0x23→0xB7`. |
| `0x00` | Spawn / intro | long once | Ignore. |

### Pre-hit vulnerable vs post-hit stun

**Pre-hit (status on the frame *before* boss HP drops)** — 28 hits:

| Pre-status | n | Mean dmg | Role |
|------------|---|----------|------|
| `0x2E` | 10 | 3.9 | Already in multi-hitstun — keep Y |
| `0x3E` | 9 | 3.2 | Classic opener after spin settles |
| `0xB7` | 6 | **8.0** | Highest single hit |
| `0xEE` | 2 | 3.5 | Rare spin-chip (not reliable) |
| `0x40` | 1 | 5.0 | Rare post-spin |

**At-hit (status on the drop frame)** — always stun:

| At-status | n | Meaning |
|-----------|---|---------|
| `0x2E` | 19 | Multi-hitstun (follow-ups + big hits land here) |
| `0x17` | 9 | Single hitstun (typical first connect from `0x3E`) |

**Policy takeaway:** treat `{0x3E, 0x2E, 0xB7, 0x17}` as “press toward+Y” (matches current `_PUNISH_STATUS` plus `0xB7` which is **missing** today). `0xEE` is **not** a punish target.

---

## Combo / cycle shape

### Micro-combo (when a window is found)

Reliable 2–3 hit string once the first connect lands:

1. **Opener** — pre `0x3E` (or rare `0xEE`/`0x40`) → at `0x17`, dmg **3–4**, `adx ≈ 48–52`
2. **+70–80f** — pre `0x2E` → at `0x2E`, dmg **3–5** (sometimes 10), `adx ≈ 16–18`
3. **+40f** (when `0xB7` appears) — pre `0xB7` → at `0x2E`, dmg **8**, `adx ≈ 16`

Full string damage ≈ **14–17**. Partial strings (opener + one follow-up only) ≈ **6–7**.

Canonical trajectory examples:

```
# opener from settled spin
… 0xEE × N → 0x3E × ~8 → HIT (at 0x17)

# follow-up chain
… 0x17 → 0x9F → 0x74 → 0x2E → HIT (at 0x2E)

# big punish
… 0xEE → 0x23 × 6 → 0xB7 → HIT 8 dmg (at 0x2E)
```

### Macro cycle length

- Inter-hit gap **p50 ≈ 462f** (matches the historical “~497f thrash cycle”).
- Many gaps are **long dry spells** (1.2k–7k f) while Slash is off-lane / spinning / clawing and thrash whiffs.
- Successful **cluster** starts (first hit → next cluster first hit) often land near **~500–1400f**, with outliers >5k when the policy loses the lane.
- Thrash spent **~54%** of frames on `slash_approach` and only landed 128 damage in 40k f — **window detection + lane stickiness**, not raw mash rate, is the bottleneck.

---

## Geometry: attack dx / side

All successful hits were **Y-aligned** (`player_y ≈ slash_y ≈ 170`).

| | Value |
|---|--------|
| Preferred side | **Player on Slash’s left** (`side=left`, `dx > 0`) — 21/28 hits |
| Opener `adx` | **48–52** (p75 of all hits ~50; first connects sit here) |
| Follow-up / `0xB7` `adx` | **16–18** (tight, inside current “too close” back-off of 10) |
| `dx` sign | Positive when attacking from the left; mirror if forced right |
| Iframes at boss hit | Always **0** in thrash sample — connects do **not** require player iframes |

Current `SlashTactics` already jump-crosses then toward+Y; the probe confirms **behind/left at mid range, then glue at adx≈16 during stun**.

---

## Danger map (player damage)

Player HP drops while thrashing (heal-after-measure, 20–25k f):

| Slash status at drop | Share | Dmg/hit | `adx` |
|----------------------|-------|---------|-------|
| **`0x09`** (claw active) | **~90%+** | 8 | mean ~44, range **31–79**, p50 ~40 |
| `0xB7` | ~5–8% | 6 | **15–28** (point blank) |
| Others (`0x42`, `0xB6`, …) | rare | 8–16 | varies |

**Shell spin `0xEE` never appeared as the status on the player-damage frame.**  
Typical claw string:

```
(0xEE optional) → 0x83 × ~6f (windup) → 0x09 × ~6f (active, 8 dmg)
```

~60% of player hits had `0xEE` somewhere in the prior 20f (spin often arms the claw), but **dodging only `0xEE` leaves the real hitbox (`0x09`) live**.

### Safe distance rules

| Situation | Rule |
|-----------|------|
| During **`0x09`** or **`0x83`** | Leave to **`adx ≥ 80`** (beyond observed claw reach 79). Prefer hop **away** + don’t walk in. |
| During **`0xEE`** | Not directly damaging, but treat as **pre-claw**: start spacing out if `adx < 64`, complete dodge before `0x83`. Current dodge `adx < 48` is **too tight** vs claw reach. |
| During **`0xB7`** / **`0x23`** | Attack from **`adx ≈ 16–24`** is optimal for 8 dmg, but **`adx < 16` risks 6 chip** — hold spacing, don’t overlap. |
| During **`0x3E` / `0x2E` / `0x17`** | Commit toward+Y; these are our windows. |

---

## Recommended attack / dodge rules (for implementer)

1. **Add `0xB7` (and optionally `0x23` as telegraph) to punish set** — highest dmg window; current `_PUNISH_STATUS` is only `{0x3E, 0x2E, 0x17}`.
2. **Opener:** when status ∈ `{0x3E, 0x40}` and `|dy| ≤ 10`, approach to **`adx ∈ [40, 56]`** on Slash’s **left**, then toward+Y (or short cross if on the wrong side).
3. **On first connect (`0x17`/`0x2E`):** glue to **`adx ∈ [14, 22]`**, mash toward+Y for ~80–120f to catch follow-ups and possible `0xB7`.
4. **Watch `0xEE → 0x23`:** immediately micro-space to adx~16–20 and Y-attack through `0xB7` (8 dmg); do not flee this window.
5. **Dodge claw, not only spin:** on **`0x83` or `0x09`**, hop away until **`adx ≥ 80`** (or off the attack lane). `0xEE` alone with adx 48 still dies to the following `0x09`.
6. **Widen spin pre-dodge:** if `0xEE` and `adx < 64` and no iframes, start retreat (today’s 48 threshold is under ranged claw).
7. **Prefer left flank** (`player_x < slash_x`); if Slash corners left, accept right flank with mirrored dx rather than idle mid.
8. **Ignore `0xEE` as a hit target**; rare spin-chips are not worth walking into claw range.
9. **Cycle expectation:** plan offense in **~450–700f** bursts after each spin settle; if no `0x3E`/`0xB7` connect for >900f, re-acquire lane (don’t infinite `slash_approach` at x>256 parking).
10. **No A special**; survival assist is orthogonal — better claw dodge cuts the emergency-heal count more than raw DPS.

---

## Gaps / next measurements

- Hard-mode only (`FullHardBoss5`); confirm same status IDs on `Boss5` / `Boss5_mid`.
- Jump-attack (B+Y) vs grounded Y damage was not isolated — thrash already mixes cross jumps.
- Exact claw hitbox width vs Y-delta not swept; all natural thrash damage was on equal Y.
- KO tail (HP 32→0) not reached in 40k thrash; low-HP rage patterns may differ.

Raw hit log: `tmnt_iv/recordings/slash_vuln_probe/FullHardBoss5_thrash.json`.
