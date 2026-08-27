## Residual — rr-iprz.3 Form-2 vertical offset

**Status:** KEEP on emergency. Play extracted to `tactics/shredder_f2.py`.
Do not edit `STATUS.md` / `BASELINE_METRICS.md` from this pin.
Assists in `record_full_hard_run.py` were **not** turned off.

### Verified this session

| Pin | Heal | Before | After |
|-----|------|--------|-------|
| `Boss9_phase2` (char 2, 0xAE spawn) | emergency | **life_loss** 485f / 15 dmg / boss 190→186 | **cleared 2×** 9,420f (2:36.7) / 152 dmg / 3 heals / min HP 16 / lives 4→4 / 190→0 |
| `Boss9_phase2_mid` | emergency | (not baselined) | cleared 7,020f / 120 / 2 |
| `Boss9_phase2_low` | emergency | (not baselined) | cleared 2,280f / 40 / 2 |
| `RaphFullHardBoss9` (char 8, form 1) | emergency | cleared 6,300f / 144 / 2 | **unchanged** 6,300f / 144 / 2 (probe stops at form-1 `event=0x19`) |

Drop window is leaving anim `0xEE`/`0xFE`. Aura-up holds a 16–28px vertical
offset; green fireball no longer life-losses the emergency pin.

### Exact next action

1. **heal=none stretch (failed):** `Boss9_phase2` killed the boss (190→0)
   then **life_loss at 4,779f** (min HP 8, dmg 72, `continue` 120f). Fade
   after HP 0 still needs the ≤16 emergency top-up. Do not claim Clean
   form-2. Tighten post-kill idle / don't walk into leftover flame.
2. **Raph form-2 pin:** no `Raph*phase2` state. `RaphFullHardBoss9` is
   form 1 only (probe `boss_down` on stage 8→9). Capture a Raph form-2
   state or extend the probe past `event=0x19` so the continuous-faithful
   Raphael pin actually fights 0xAE (Raph HP 48 vs Leo 80).
3. Production iframe hold is still on (4,635f whole-run). Emergency
   form-2 clear is the play prerequisite, not assist removal.

### Non-claims

- No STATUS / BASELINE_METRICS promotion
- No whole-run dry-run
- No iframe assist off
- No heal=none KEEP
- No Raphael-char-8 form-2 clear (Leo `Boss9_phase2` only)
