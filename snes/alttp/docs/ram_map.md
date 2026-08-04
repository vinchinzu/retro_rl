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
| sword | `$F359` | 78713 via WRAM_IDX | high WRAM (0=none, 1=fighter) |
| dungeon keys | `$F36F` | 78735 via WRAM_IDX | `0xFF` = blank HUD / no dungeon keys yet |
| follower | `$F3CC` | 78828 via WRAM_IDX | tagalong; `1` = Zelda |
| secret passage room | `$A0` base | 160 | opening hole drop-in `0x55` |
| secret hole approach | world X/Y | `$22`/`$20` | near Yaze `0x7D` ~(2432,1696) on screen `0x1B` |
| link direction | `$2F` | 47 | 0=up, 2=down, 4=left, 6=right |
| link action | `$5D` | 93 | `21` = hold-up-item after sword get |
