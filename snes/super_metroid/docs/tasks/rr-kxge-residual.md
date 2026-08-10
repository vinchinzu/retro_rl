## Residual — rr-kxge Dual continuous `--to ice` stabilize

### Result
**CLOSED — dual continuous GREEN** power-on → Ice Beam.

| Report | Result | Frames | Room | Beams |
|--------|--------|-------:|------|-------|
| `ice_dual_d` → `ice.json` | **GREEN** | **148167** | `0xA890` | `0x1007` |
| `ice_dual_e` → `ice_dual.json` | **GREEN** | **148167** | `0xA890` | `0x1007` |

Integrity both runs: 0 loads / prog / capacity / deaths; `ice_collected`;
`natural_ice_room_entry` + `post_ice_ordinary`. Exact frame match.

Default CLI tip promoted to **`ice`**. STATUS program gate updated 2026-08-10.

### Climb harden (landed)
| Change | Detail |
|--------|--------|
| Cont-tuned ladder | Charge loadout: runup 18/20/22 + pos_1339 90 first; pure 8→14/84 ladder for pre-Charge pins |
| Classic warehouse setup | `bound_floor_left=False` keeps open-loop LEFT/LEFT/RIGHT (spine frame-lock) |
| Minimal door lip | Classic setup only RIGHT-biases at floor x≤40 (no re-center desync) |
| Safe setup | `bound_floor_left=True` re-center + soft bound for Ice retries only |
| Prejump band | Cont path nudges prejump x into ~78–90 after walk-to-pos |
| RIGHT-biased recover | Floor recover without HJ `0xAA41` kiss |
| HJ soft return | Pressure RIGHT back from shaft when setup exits |

### Verify paste
```bash
uv run pytest snes/super_metroid/tests/test_continuous_tips.py \
  snes/super_metroid/tests/test_k4_speed_branches.py \
  snes/super_metroid/tests/test_k4_ice_scaffold.py \
  snes/super_metroid/tests/test_k4_wave_return_scaffold.py \
  snes/super_metroid/tests/test_source_states_and_ram_cache.py \
  snes/super_metroid/tests/test_segment_contracts.py -q
# → 55+ passed (default tip ice)

uv run python snes/super_metroid/scripts/probe/kpdr.py pure business-to-ice-gate \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_frog_save_to_business_pure.state \
  --expect-room 0xA7DE
# → GREEN room=0xA815 xy=(1752,651) frames=3255 (×2)

uv run python snes/super_metroid/scripts/record/continuous.py --to ice --no-video \
  --report snes/super_metroid/recordings/ice.json
# → GREEN 148167f ice_collected 0xA890 beams 0x1007
```

### Acceptance
- [x] Ice tip hops include Wave→Business return + Ice pure stack (11 hops)
- [x] Pure floor Business→Gate dual green (3255f×2)
- [x] Elev Business→Gate still green (891f)
- [x] **Dual continuous green** power-on → Ice (148167f ×2)
- [x] Default CLI tip **`ice`** (STATUS-promoted)
- [x] Tests + residual + NIGHT_WATCH_LOG

### Return chain (success path, ice_dual_d/e)
wave_to_double → … → frog_save_to_business f141473 → business_to_ice_gate f145230
→ ice_gate_to_acid → ice_acid_to_snake → ice_snake_to_ice f147799.

### Residual risks (non-blocking)
1. Offline continuous floor dump can still RED ~2/5 under enemy noise; dual
   continuous integrity is the contract and is green.
2. Over-aggressive safe setup (re-center always) desynced warehouse spine
   (+457f → bubble Super miss) — **must not** change classic `bound=False` path.
3. Optional demo video still open (no-video dual only).

### Non-claims
- Did not record ice demo mp4 this session
- Did not claim Clean-track Ice
- Did not start Alpha PB pure stack
