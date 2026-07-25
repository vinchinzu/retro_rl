# ALTTP — RAM map

stable-retro `get_ram()`:

- index `0..8191`: WRAM `$7E0000-$7E1FFF`
- index `16384+offset`: full WRAM for offsets `>= 0x2000`

| Field | WRAM | Index | Notes |
|-------|------|-------|-------|
| module | `$10` | 16 | `0x01` title, `0x02` file select, `0x07`/`0x09` play |
| submodule | `$11` | 17 | `0` when controllable |
| indoors | `$1B` | 27 | `0` outdoor |
| link Y/X | `$20`/`$22` | 32/34 | u16 |
| screen id | `$8A` | 138 | overworld screen |
| room id | `$A0` | 160 | u16 dungeon room |
| dark world | `$0FFF` | 4095 | nonzero in DW |
| sword | `$F359` | 78713 via WRAM_IDX | high WRAM |
