# Super Metroid practice ROM + repertoire

Community practice hack ([tewtal/sm_practice_hack](https://github.com/tewtal/sm_practice_hack),
patcher [smpractice.speedga.me](https://smpractice.speedga.me/)) plus this
repo’s **repertoire** — the full preset menu tree used as a shared spine for
**human demos, room policies, path stitching, and autopilot recovery**.

## ROMs (gitignored under `roms/`)

| File | Role |
|------|------|
| `SuperMetroid.sfc` | **Product** vanilla NTSC (continuous / pure / STATUS) |
| `SuperMetroid_Practice.sfc` | Practice hack (InfoHUD + category presets) |
| `SuperMetroid_Practice_tinystates.sfc` | Same + in-ROM savestates (emulator-friendly) |

Vanilla SHA1 (unheadered 3 MiB): `da957f0d63d14cb441d215462904c4fa8519c613`.

```bash
# Requires vanilla already at roms/SuperMetroid.sfc
uv run python snes/super_metroid/scripts/setup_practice_rom.py
uv run python snes/super_metroid/scripts/setup_practice_rom.py --status
```

Product evidence **never** loads the practice ROM. Practice ROM WRAM / tinystates
are not drop-in replacements for stable-retro vanilla `.state` pins. The
repertoire **names and order** are shared; binaries are harness pins.

## Repertoire = practice-hack preset menus

The practice hack’s category menus (KPDR, PRKD, 100%, GT, RBO, …) are the
standardized **session list** (ordered route spine). Catalog:

- `maps/practice_repertoire.json` — all categories, areas, leaf presets + room /
  items / beams fingerprints from preset data
- `practice_repertoire.py` — multi-role API + CLI
- Product default category: **`kpdr25`** (Any% KPDR – Early Ice)

### Roles (every session)

| Role | Use |
|------|-----|
| `human_practice` | `./play` / guided_human / multi-take demos |
| `policy_tune` | Room-local reactive policy compile from hop body + entry pin |
| `policy_graduate` | draft → candidate → `verified_live_anchor` → product_spine |
| `path_stitch` | Ordered seam: pin → hop_key → next pin (compose / hop-replay) |
| `autopilot_recovery` | Live room+items → nearest reseed pin when AP has no skill |

Graduation ladder:

```text
none → draft → candidate → verified_live_anchor → product_spine
```

### CLI

```bash
# Categories (★ = product)
uv run python -m super_metroid.practice_repertoire --list-categories

# Ordered route (stitch spine)
uv run python -m super_metroid.practice_repertoire --route --category kpdr25

# One session: human pin + hop_key + stitch + policy workspace + recovery
uv run python -m super_metroid.practice_repertoire --session kpdr25/crateria/morph

# Stitch seam morph → next preset
uv run python -m super_metroid.practice_repertoire --stitch kpdr25/crateria/morph

# Full stitch board (all seams)
uv run python -m super_metroid.practice_repertoire --stitch-board --category kpdr25

# Policy tune/graduate board (grade + existing policies + tune command)
uv run python -m super_metroid.practice_repertoire --policy-board --category kpdr25

# Autopilot / thrash recovery for a live room
uv run python -m super_metroid.practice_repertoire --recovery 0x9E9F --items 0x0004

# Product map + coverage
uv run python -m super_metroid.practice_repertoire --mapped
uv run python -m super_metroid.practice_repertoire --gaps
```

### Canonical artifacts

Every repertoire session id `cat/area/slug` owns:

| Artifact | Path |
|----------|------|
| Canonical state | `custom_integrations/SuperMetroid-Snes/practice_repertoire/<id>.state` |
| Canonical demo | `recordings/practice_repertoire/<id>.{json,mp4}` |
| Policy plan | `policies/reactive_rooms/plans/<id_with__>.json` |
| Hop identity | `hop_key` via `skill_bank.make_hop_key` (room:from→to:items) |

Product map (`PRODUCT_SESSION_MAP`) points high-value KPDR25 sessions at living
`full_start_v1_*` / pure pins so **demos, policies, and stitch** share one
continuous-like lineage (`./play morph`, `./play bomb`, …).

### Room policy tune → graduate

```text
repertoire session (entry pin + inventory)
  → hop body from human tape / bank
  → optimize_room_policy.py (reactive room skill)
  → status candidate → verified_live_anchor (dual-green + takeover sweep)
  → RoomAutopilot loads skill; missing rooms fall back + recovery hint
```

See [REACTIVE_ROOM_POLICIES.md](REACTIVE_ROOM_POLICIES.md). Policy board cards
print a starter `optimize_room_policy.py` command per session.

### Product route edges

Practice-hack menu order **is** the product route order for a category:

```text
… → session_i (entry pin) --hop_key--> session_{i+1} (leave pin) → …
```

API: `route_edge` / `product_route_edges` (CLI still accepts `--stitch` /
`--stitch-board`). Use with `human_tape.compose` / hop-replay / skill bank —
not multi-minute button concat.

### Autopilot recovery

When `RoomAutopilot` has no compiled skill for the live room, status detail
cites a repertoire recovery pin (`recover→kpdr25/... grade=… hop=…`). Callers
can also query:

```python
from super_metroid.practice_repertoire import recovery_hint_for_state
hint = recovery_hint_for_state(state)  # room+items → pin + hop_key + next
# or: autopilot.recovery_hint()
```

Reseed the PlaySession from `hint.state_path`, then re-enable AP or hop-replay
the seam to `next_session_id`.

Regenerate catalog from upstream menus:

```bash
uv run python snes/super_metroid/scripts/export/practice_repertoire.py
```

## Controls (in practice ROM)

- Open menu: **Select** (InfoHUD / practice menu — see upstream Help)
- Category presets: teleport with correct inventory along a route
- Prefer **emulator** build for BizHawk/stable-retro; **tinystates** when you
  want the hack’s own save/load slots

## Relation to other layers

| Layer | Role |
|-------|------|
| Practice repertoire | Shared **names + route order + fingerprints** for human + bot |
| `start_presets.py` | `./play` short names → living pins |
| `source_states.py` | Pure entry fingerprints for executor cards |
| `skill_bank.py` | Hop PB bank keyed by `hop_key` |
| `reactive_policy` / `autopilot` | Room skills + mid-room attach; recovery via repertoire |
| `full_start_v1` seams | Human-recorded product item-to-item pins |

Align new pins to repertoire ids when possible
(`kpdr25/upper_norfair/bat_cave` ↔ `bat-cave` / `full_start_v1_bat`).
