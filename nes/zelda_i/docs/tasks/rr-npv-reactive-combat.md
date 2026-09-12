# Reactive combat tools — tracking / threat / postmortem

Cross-lane tooling sitting. **No STATUS claim.** `route_eligible=false`.
No pokes, no assists, no recording overwrites. The Clean tip is still
`l1_tf` (M5, L1 only).

## The gap

Every blocked Clean room failed the same way, and the residuals could not
see it because each lane described its own symptom:

| Lane | Room | Symptom in the residual |
|------|------|-------------------------|
| `rr-npv.4` | L8 `0x1E` | 3/3 deaths oscillating `128↔112` on `STAND_Y=181` |
| `rr-d6v` | L6 `0x78` | 3/3 deaths sliding along the `y=141` waist |
| `rr-npv.2` | L5 `0x77` | 3rd death on the same hold cell `(120,173)` |

All three policies saw **positions only**. Motion was re-derived ad hoc in
five places (`level5/path.py`, `level5/dungeon.py`, `level6/wizzrobe.py`,
`level6/gohma.py`, `level8/magic_key.py`), each with its own `prev_xy`, and
evasion was a per-room position table. `level6/wizzrobe._off_band_dir`
literally took the beams as an argument and `del`-ed them.

A position table cannot answer *when*, so it tunes forever.

## Landed

Three modules under `dungeon/`, each importable by any controller:

- **`tracking.py`** — `ObjectTracker` turns slots into tracks with velocity,
  age and hazard class. Identity is `(slot, type_id)`; a type change or a
  >32 px teleport restarts the track so a respawn is not a missile.
  Classification is **motion-first**: the L6 beam is type `0x59`, in no
  `dungeon/ids.py` table, and a HP-0 slot moving ≥1.6 px/frame is a shot
  whatever its type byte says. Observation is idempotent per snapshot, so
  a subclass calling `super().step` does not sample velocity twice.
- **`threat.py`** — `contact_frames` / `assess` give time-to-contact for a
  candidate button; `ReactiveEvader.decide` returns the button that buys the
  most frames, or `None` when standing is already safe. Plus the pre-emptive
  half: `firing_axis` / `in_firing_line` / `off_line_step`.
- **`postmortem.py`** — `DamageLog` names the hazard that took each heart,
  using the *previous* frame's tracks (a shot despawns on contact). Wired
  into `GenericDungeonRoomController` and `HopController`, so every room and
  every dest hop now reports a cause, not only a death tile.

## The measurement that closes three lanes

Link walks 1 px/frame. A sidestep only clears a hitbox once he has walked
its full pad:

```text
MIN_DODGE_BODY = 16 frames   (Link 8 + body 8)
MIN_DODGE_SHOT = 12 frames   (Link 8 + shot 4)
```

`threat.dodgeable(impact)` is `False` below that. At the L8 `0x1E` stand
line the fireball arrives in ~2 frames, so **no peel tuning could ever have
worked** — h13/h14/h15 were tuning an impossible move. The same arithmetic
explains L5 `0x77`: the peel started well inside 16 frames.

Consequences now encoded in the evader:

- A step that gains fewer than `MIN_ESCAPE_GAIN` (4) frames is refused as
  `evade_no_gain` — delay-only shuffles *are* the oscillation.
- A committed escape is held for `commit_frames`, and the reverse of the
  last step is only taken when it clearly beats every other option.
- Already in contact (`ttc ≤ 1`): `evade_peel` breaks the axis Link shares
  with the source (the nearly-zero one), toward the larger free span.
- Nothing in flight: `off_line_step` leaves the shooter's axis while there
  is still time to walk.

## ROM evidence (L6 `0x78`, Clean, no assist)

`run_level6_entrance_tf.py --from-state Level6Entrance --no-infinite-life
--no-video --trials 1`.

- `l6_threat_v1`: still red at `level6_west_clear_0x78`, but the leftover
  **moved** off the v4–v9 pose `(144,141)` to `(189,149)` — and the census
  showed the room is a **crossfire**: 5 × `0x24` plus 4 × `0x59` live at
  death, not the single beam the residual modelled. Notes carried
  `threat_off_firing_line`, `threat_evade_commit`, `threat_evade_no_gain`,
  `threat_evade_peel`, so the layer engaged.
- The new pose was the finding: `off_line_step` cleared one wizzrobe row by
  stepping **east into the corner** where the other beams converge. It
  counted lines cleared and ignored where it landed. Fixed by breaking the
  tie toward the room interior (`_interior`), with a unit at `(176,141)`.

- `l6_threat_v2` (interior tie-break in): red at the same stage, 2452f,
  leftover `0x78 (168,141)` mode 17. First trial in this lane to report a
  **cause**:

  ```text
  death_cause: f308 body 0x59 (unknown) from E v=(-3.0,+0.0) d=11
               while action=wizzrobe_beam_peel phase=FIGHT
  ```

  And the per-stage hit census, which is the actual finding:

  | stage | ok | frames | hits by cause |
  |-------|----|--------|----------------|
  | `level6_right_0x7a` | yes | 374 | — |
  | `level6_east_key_0x7a` | **yes** | 1084 | `0x24_E` ×4, `0x59_E`, `0x59_W` |
  | `level6_return_0x79` | yes | 305 | — |
  | `level6_west_key_0x78` | yes | 381 | — |
  | `level6_west_clear_0x78` | no | 308 | `0x59_E` |

  **Six of the seven hits land in stages that pass.** `0x78` is where Link
  runs out, not where he loses the health. Six sittings tuned `0x78`; the
  leak is `level6_east_key_0x7a` taking four contact hits from `0x24`
  bodies on the east side. That is invisible to a green/red stage table
  and obvious in one damage census.

- The same report also said `body 0x59` — the live beams carry `hp=128`, so
  the HP-0 test mis-filed them as bodies (pad 16, body dodge threshold).
  Classification is now: an **unrecognised** type at shot speed is a shot,
  whatever its HP byte says. Known enemy kinds stay bodies however fast.

- `l6_threat_v3` (shot classification fixed): red at the same stage, 2449f,
  leftover `0x78 (176,144)`. The cause line is now correct and the verdict
  is explicit:

  ```text
  death_cause: f305 shot 0x59 (unknown) from E v=(-3.0,+0.0) d=8
               while action=wizzrobe_evade_no_gain phase=FIGHT
  ```

  `evade_no_gain` is the model saying the frame was already lost — at
  `d=8` and 3 px/frame there were under 3 frames, against `MIN_DODGE_SHOT`
  of 12. `level6_east_key_0x7a` again spent 6 hits (`0x24_E` ×3,
  `0x59_E` ×2, `0x59_W`). Link reaches `0x78` on his last heart.

**Next lever for `rr-d6v` is `level6_east_key_0x7a`, not `0x78`.** Do not
re-run the v4–v9 east-waist chase; that class stays blocked.

## Clean tip, as a table

`spine/clean_tip.py` + `scripts/clean_tip.py` replace the hand-carried
"W1 open, W2/W3/W4 blocked" note. Rows carry rung, blocker **class**, room,
pose and the residual they came from; `--json` for tooling.

```bash
uv run python nes/zelda_i/scripts/clean_tip.py
```

The blocker class is the point: `shot_undodgeable`, `body_undodgeable` and
`firing_line` each name the tool to reach for, so the next sitting does not
write a fourth position table. `tests/test_clean_tip.py` fails if a row
cites a residual that is not in the tree, or if a blocked row claims
spine-green.

## Not claimed

Clean tip unchanged (`l1_tf`). L5 / L6 / L8 beads stay open and
`route_eligible=false`. `engine.py` folded back under the ~1000 LOC bar by
dropping the speculative generic evade layer (no room spec consumed it) —
controllers configure `self.evader` themselves, as `level6/wizzrobe.py`
does.
