# Raphael Hard speedrun / wiki strategies (for policy)

Sources, not STATUS. Map these onto `policy.py` / `tmnt_iv/tactics/` without
A-special. Clean-compatible tools: **Y, B, dash+Y, jump+Y, screen throw,
multislam**. Forbidden: Power Attack (Y+B / A) because it spends HP.

## Sources

| Source | What it owns |
|--------|----------------|
| [SRC damage guide](https://www.speedrun.com/teenage_mutant_ninja_turtles_iv_turtles_in_time/guides/o9l5p) (Nitsuja via Spiriax) | Per-hit damage table; turtle ranking |
| [RetroMaggedon Hard walkthrough](https://retromaggedon.com/index.php/teenage-mutant-ninja-turtles-iv-turtles-in-time-walkthrough-snes/) | Stage/boss patterns on Hard |
| [SRC Any% 1P1C Hard](https://www.speedrun.com/Teenage_Mutant_Ninja_Turtles_IV_Turtles_in_Time) | Human WR-class ~19:33 Raphael |
| Mike Uyama 22:31 Hard Raph (2006) | Why Raph: one-hit purple Foot, best special/dash |
| ESA17 / Sunday Sequence Break commentary | Skip purple-Foot-only screens; standing special = iframes (we cannot use that) |

## Raphael kit (Nitsuja numbers, Hard Super Shredder tests)

Raph is **best** at upward jump attacks, specials, dash attacks, and speed.
Raph is **worst** at regular grounded combos and defense.

| Move | Damage | Notes | Clean-ok? |
|------|--------|-------|-----------|
| Ground combo 1–4 | −5/−4/−3or5/−8 | Weak; current policy lives here | yes |
| Jump+Y 1 (upward kick) | −4, often 2-hit | Raph's best grounded-to-air tool | yes |
| Jump+Y 2–4 | −4/−6/−8 | Floating downward slash can multi-hit | yes |
| Dash+Y (shoulder ram) | −10 | Breaks block; Raph-best | yes |
| Post-ram standing kick | −12 | After ram connects | yes |
| Dash+jump+Y 1–2 | −14/−20 | Fast travel + huge hit | yes |
| Special / dash-special | −16 | **HP cost** | **no** |

Policy implication: Raphael should **open with B+Y or dash+Y**, not mash
standing Y. Standing Y is the turtle's worst tool and is what
`fight_nearest_action` currently emits.

Dash is **hold a direction long enough**, then Y — already used in
`TechnodromeTactics._blocker_action` (retreat → charge ≥34f → toward+Y).

## Stage / boss recipes (Hard)

### Technodrome tank (Shredder machine) — 1,022 dmg bucket

- Damage Shredder **only** by throwing stunned Foot at the screen.
- Hard: **only Tonfa / pink Foot** spawn. They block standing Y.
- Wiki options: (1) **jump behind** + quick Y, then throw immediately;
  (2) shoulder check then throw **on the next frames** or Shredder guns
  you. Current code uses a long retreat+charge (40f + 34f) which is
  safer but slow.
- Candidate: jump-behind opener for `0x6C` when already close (`adx` small),
  keep the long ram only when the Foot is facing and blocking at range.

### Slash (Prehistoric) — 861 dmg / 11,386f boss

- Hard Slash **blocks shoulder slams**. Do not ram him.
- Wiki (Normal, still useful): jump-kick **over**, land just behind, pivot
  combo, hop away. Repeat. Never face-tank.
- Wiki (Hard): stay away, **bait the jump**, jump-kick as he lands, a few
  grounded hits, then jump away. He can break combo anytime. At ~3 bars
  he shells-spins (`0xEE`) — hop away (already in `SlashTactics`). At 2
  bars he becomes very fast — patience.
- Current hybrid whiplash (approach@48 → B-cross → toward+Y, spin dodge
  adx 52) works but is slow. Jump-over + behind-combo is the wiki kill
  that should cut both frames and chip.

### Form-2 Super Shredder — iframe assist 4,635f

- Aura color = next attack. **Green fireball from the front = life loss.**
- Stand **just above or below** (or behind) while aura is up; step in and
  combo when it drops; follow the teleport. Never jump in from the front.
- Current `SuperShredderForm2Tactics` left-standoffs and hops every cycle.
  Wiki wants **vertical offset**, not only horizontal, and attack only on
  the drop. That is the Clean path off the iframe write.

### Other stages (lower ROI this sitting)

- **Rat King:** aerial during torpedoes; bottom-right during mines (already
  long-poke from water lane).
- **Bebop/Rocksteady:** kill **one** and they finish each other. Jump-kick
  in; do not walk into the rapier.
- **Leatherhead:** 3 hits then he charges — jump away. Hard: falling
  barrels on the charge (watch shadows).
- **Krang (Neon / Starbase):** jump toward and hammer; teleport more on Hard.
- **Purple-Foot-only screens:** speedrunners skip. Bot currently fights
  everything. A later skip policy is a big time cut but needs scroll-lock
  RAM (not this sitting unless a probe proves a skip does not freeze).

## Anti-patterns already burned (do not relearn)

See `CLEAN_PLAYBOOK.md`. Speed-specific extras:

- Blind `slash_spin_dodge_adx=40` from a Raph boss probe **raised**
  whole-run damage via Skull/WK RNG.
- Leo `blocker_hit_frames=8` **raised** continuous Technodrome 1,022→1,131.
- A-special iframes are a speedrun staple; they are **out of contract**.

## How to land a KEEP

1. Unit test the new branch (char 8, named reason strings).
2. Live probe on a `RaphFullHard*` state, heal=emergency, 2×.
3. Compare frames **and** damage to the table in `BASELINE_METRICS.md`.
4. If the probe is a big win on a boss that changes later-stage RNG
   (Slash especially), do **not** STATUS — park for a full dry-run.
5. Extract if `policy.py` would grow.
