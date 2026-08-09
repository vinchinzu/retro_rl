# Next session prompt — SM TAS BSNES Oracle (long path)

Copy-paste starter for the next agent/human session.

---

## Prompt

```text
You are continuing Super Metroid TAS work in retro_rl at the monorepo root.

## Mandate
Execute the **long path** (native-core oracle), not more snes9x open-loop thrash.

Authoritative plan:
  snes/super_metroid/docs/TAS_BSNES_ORACLE.md
Sibling (snes9x hybrid / product re-anchor — still valid, not the focus):
  snes/super_metroid/docs/TAS_ADAPT.md

Beads epic: rr-0lz6
  bd show rr-0lz6
  bd ready -l super_metroid
Claim one child before coding. Prefer Phase 1 (rr-wbsr) then Phase 2 (rr-l8zj).

## Non-negotiables
1. Do NOT spend the session grid-searching movie_start under stable-retro/snes9x for full 100% or Climb.
2. Do NOT STATUS-promote movie frames or oracle dumps.
3. Do NOT sanitize L+R on TAS bodies.
4. Product pure-first remains the continuous tip; oracle is reference only.
5. Full BK2 power-on under snes9x is a documented dead-end (Ceres thrash only).

## Why (context)
- sniq_100p.bk2: BizHawk Core=BSNES Compatibility; harness is snes9x only.
- Inputs parse fine; physics desync after early Ceres under harness.
- Landing→Parlor movie splice works; Climb under open-loop movie does not.
- extract_hops thrash gate already cuts post-desync (usable ~6 on sniq_100_full).

## This session goals (in order)
1. Read tas/ref/ORACLE_ENV.md (Phase 1 partial already done 2026-08-08).
2. Claim rr-wbsr residual: clear Linux BizHawk 2.11 Mono **SIGSEGV** mid-Ceres
   (Wine Windows BizHawk, alternate 2.x build, or isolate crash). Re-run
   ./snes/super_metroid/tas/oracle/run_verify_100.sh until GREEN
   (morph + Zebes room, or full movie end). Prior run already proved Ceres Ridley
   0xE0B5 @~10k under libsnes BSNES — not a ROM/hash problem.
3. If Phase 1 green: claim rr-l8zj — BizHawk Lua dumper under tas/oracle/
   writing recordings/tas_oracle/sniq_100_bsnes/{pins,summary,room_timeline,...}
   matching harness TraceEvent / probe_pin shapes (plan schema).
4. When dump exists: claim rr-gwd0 — offline extract_hops/import so oracle dir
   produces a truth board (not thrash sniq_100_full).
5. Update plan checklist + close beads with evidence paths.
6. bd sync; commit code + .beads/issues.jsonl together. Push only if human asked.

## Out of scope this session
- Porting bsnes into stable-retro (Phase 6 stretch only).
- Replacing product morph spine with movie bodies.
- Pure Ice Snake (rr-5if) unless blocked and human redirects.
- Closing rr-d7mq from Ceres thrash skill tags.

## Done when
- SEGV cleared and verify_proof.json status=GREEN (morph + Zebes), OR new env notes in ORACLE_ENV.md.
- Ideally: first non-empty oracle pins.json with rooms beyond Ceres six.
- Plan checklist Phase 1 boxes updated; next bead left ready.

## Commands / paths
  Movie: snes/super_metroid/tas/ref/sniq_100p.bk2
  Env:   snes/super_metroid/tas/ref/ORACLE_ENV.md
  Tool:  snes/super_metroid/tas/oracle/run_verify_100.sh
  BizHawk: which bizhawk  (~/.local/bin/bizhawk → 2.11 Mono)
  Plan:    snes/super_metroid/docs/TAS_BSNES_ORACLE.md
  Out:     snes/super_metroid/recordings/tas_oracle/sniq_100_bsnes/
  Tests:   uv run pytest snes/super_metroid/tests/test_tas_*.py -q
  Beads:   bd update <id> --status in_progress
           bd close <id> --reason "…"
           bd sync
```

---

## One-line next action

Clear libsnes `PPU::render_line` SEGV (Wine Windows BizHawk preferred) → `LUA_SCRIPT=long_count.lua MAX_FRAMES=60000` verify past intro (~8–12k elev) to Landing/morph GREEN → claim `rr-l8zj`.
