# Harvest Save-State Editing Inventory

This inventory covers RAM-mode edits backed by `HarvestStateDocument` and
`harvest/core/ram_catalog.py`. These edits patch the `RAM` block inside an Snes9x `.state`;
they do not rebuild VRAM, sprite objects, or active map graphics.

## Exposed Edits

- Calendar and clock: `set_calendar_date(game_year=..., season=..., day=...)`,
  `set_clock(hour=..., minute=..., second=..., time_running=...)`, plus cached
  season/weekday/day-suffix glyph buffers.
- Generic scalar fields from `SCALAR_FIELDS`: date/weather, player/runtime
  fields, inventory/resources, tool slots, feed counts/flags, animal counts,
  romance hearts, child ages, wife pregnancy, and the known event flag banks.
- Farm tiles: persistent farm map bytes and the current visible map buffer for
  a selected `(x, y)` tile. `fill_farm_ground_with_grass()` converts every
  non-structure farm tile (`< 0xA0`) to mature grass `0x79`, updates planted
  grass/development scalars, and can leave the active map buffer untouched for
  indoor save states.
- Cows: raw status, care/age/pregnancy byte, home map, happiness, position,
  four glyph name bytes, add/clear helpers, and animal count recounting.
- Chickens: raw status bytes, position, add helper, and animal count recounting.
- Family/kids: house level flags, kid 1/kid 2 existence bits, ages, coarse
  stages (`newborn`, `baby`, `child`, `grown`), and four glyph name bytes.
  Birth/growth event trigger bits are cleared so edited kids are already there
  rather than pending a cutscene.
- Horse: ownership flag, street pickup/name prerequisite and completion flags,
  adult saddlebag scene completion flag, farm map assignment, adult age byte,
  and four glyph name bytes.
- Ending evaluation: global happiness, development rate, persistent power
  berries, shipped crop totals, dog hugs/pickups (`0x7F1F52`), ranch mastery,
  ranch development, and the hidden ending scene iterators
  `ending_scene_index`/`ending_aux`.
  `harvest/tools/ending_probe.py` drives a save through the final sleep and records these
  fields to JSON for branch testing.
- Live scalar patching: `LiveRamEditor` can apply catalog scalar fields to a
  running emulator session.

## Ending Probe Notes

- `latest` was captured with `harvest/tools/ending_probe.py` at
  `debug_alignment/ending_probe_latest.json`. It reached ending indices
  `0x04,0x05,0x09,0x22,0x0d,0x12,0x13,0x15,0x16,0x17,0x18,0x19,0x1a,0x1b,0x20,0x25`
  and finished with `ranch_mastery=348`, `ranch_development=78`.
- `latest_y3_ending_all_unmarried.state` exercises the crop, unmarried
  happiness, girl-heart, child, best-ending, and mastery branches. Its capture
  is `debug_alignment/ending_probe_latest_y3_ending_all_unmarried.json` and
  finished with `ranch_mastery=822`.
- `latest_y3_ending_all_married.state` sets all five marriage bits to exercise
  wife-specific branches. Its capture is
  `debug_alignment/ending_probe_latest_y3_ending_all_married.json` and finished
  with `ranch_mastery=742`, `ranch_development=100` after converting all
  non-structure farm tiles to mature grass.
- `latest_y3_ending_ranch_master_999.state` is built from the user's refreshed
  `latest.state` and reaches `ranch_mastery=999`, `ranch_development=100` in
  `debug_alignment/ending_probe_latest_y3_ending_ranch_master_999.json`.
  It uses 12 adult cows, 12 chickens, full cow affection, full development,
  and `511` shipped totals for all four crops. Girl-heart fields are set to
  `509` rather than `511` because the final sleep from this state raises
  Maria by 2 before the bugged `value & 0x01FF` Ranch Master calculation.
  The reproducible preset is `state_presets/ranch_master_999.json` and can be
  applied with `uv run python state_preset_builder.py state_presets/ranch_master_999.json`.
- Dog pickup/hug count is `dog_hugs` / `dog_pickups` at `0x7F1F52`. The
  decomp increments it when the dog interaction succeeds, and the best-ending
  branch checks for at least `100` (`0x0064`).
- Some ending branches are mutually exclusive in one realistic save. A married
  save skips the unmarried happiness/girl-heart checks; an unmarried save
  cannot naturally show wife-specific scenes. The two showcase saves cover both
  sides.

## Still Missing Or Partial

- Full decoding of event flag banks `0x7F1F5A..0x7F1F70`; many bits are still
  provisional HM-Decomp notes.
- Season/map visual rebuilds. Changing the season byte does not regenerate
  loaded VRAM, palettes, tilemap ID, or active map actors.
- High-level crop plot construction. Individual tile IDs are editable, but
  crop type/stage layout builders and all seasonal variants are not complete.
- Full inventory and shed bitfield APIs beyond known scalar item counts and a
  few livestock-builder tool bits.
- Human text encoding for names. The helpers write four glyph bytes; they do
  not yet convert ASCII names into the game's glyph table.
- Dynamic object/NPC table editing. NPC schedules should still be discovered
  via `WorldSnapshot.entities`/`harvest/core/npc_catalog.py`.
- Building upgrade timers and all house construction side effects. The helper
  sets known persistent house-level bits and the runtime size byte only.
- Animal state semantics beyond known cow/chicken raw fields. Sickness,
  cranky, pregnancy, milked-today, and age flags need named wrappers.
- Emulator-backed validation for every structure. Current coverage verifies
  selected scalars and livestock/family bytes, but not every event or visual
  side effect.
