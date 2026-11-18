# RAM and ROM layout

Target: headerless Chrono Trigger US v1.0, SHA-1
`de5822f4f2f7a55acb8926d4c0eaa63d5d989312` (4 MiB).
Setup strips a 512-byte copier header, verifies the hash, and keeps the
normalized ROM game-local. Other revisions fail before extraction/play.

WRAM offsets below start at bus `$7E0000`. Field direct-page offsets in
the disassembly need `$0100` added. Read the actual 128 KiB WRAM memory
block; high `get_ram()` aliases are unreliable on this core. Numbers in
offset expressions are hexadecimal.

| Block offset | Meaning | Observations / limits |
|---|---|---|
| `$0100` word | Current location | Power-on through 1000/600 AD, the 2300 AD Factory and the End of Time |
| `$0117`, `$0119` | Field scene flags / brightness | Transition settling |
| `$011F`, `$0129` | Field control / message state | Message state includes dialogue and choices; it is not a general mode enum |
| `$0197` word | Leader object table index | Object ID times two; changes by scene |
| `$1800+slot`, `$1880+slot` words | Field X/Y, divided by 16 for pixels | Directional movement; party object slots at `$0197/0199/019B` |
| `$1600+slot` | Facing 0/1/2/3 = up/down/left/right | NPC interaction |
| `$0283`, `$0285` words | Overworld pixel X/Y | Eight-pixel movement lattice |
| `$001D`, `$1BF6` | Overworld brightness / fade bits | Overworld settling |
| `$0500..0503` | NMI trampoline | Menu `5C 0E 84 C2`, field bank C0, world bank C2; battle `5C 7C E7 D1`, battle exit `5C C0 F4 D1` |
| `$0067`, `$0F0C` | Menu mode / naming destination | Mode 0D plus a valid character-name pointer recognizes naming; START retains default names |
| `$2603/2605`, `$2607/2609` words | Persistent Crono HP/max HP, MP/max MP | Updated after combat; stale during battles. Tech MP spending and inn/Enertron restoration observed |
| `$2600+char*50+27/28/29/2A` | Helmet / armor / weapon / accessory | Outfit checks offered inventory and verifies equipped IDs |
| `$2980..2982` | Active party IDs, 80 = empty | Crono 0, Marle 1, Lucca 2, Robo 3, Frog 4 observed |
| `$2400..24F1`, `$2500..25F1` | Inventory type / quantity slots | Equipment, consumables, Pendant D6, Gate Key D7, Seed D4 and Bike Key D5 |
| `$2C53` three bytes | Gold | Purchases and battle rewards observed |
| `$10000` | Story counter, bus `$7F0000` | Opening 03..0C; queen rescue 1B; future arrival 33; Arris descent 35; recording 36; Bike Key 37; Robo joins 3C; shutdown 3F; R-Series won 42; End of Time arrival 48 |
| `$10055` bits 7 / 3 | Exhibit open / candy event pending | Announcement opens exhibit; recruitment starts candy wait; release movement during it |
| `$10056` bit 1 | Telepod test complete | Set after Crono's teleport; story remains 0A |
| `$10146` bit 0 | First ambush cleared | False before first battle; set after victory. Event command bit 7 selects `$7F01xx` |
| `$100EC` bits 0 / 4 / 6 | Guardian triggered / corpse examined / rat password learned | Guardian bit is set before the fight; password bit requires catching the rat |
| `$100A4` bit 0; bits 5..7 | Seed obtained; three Arris password switches | Seed exchanged for Bike Key; switches require L+R held with A |
| `$10154` bits 0 / 3 | Johnny introduction / Lab 32 foot crossing enabled | Completing a race attempt enables walking even after a loss |
| `$10058` bits 0 / 2 | Factory security lasers / entry access unlocked | Robo operates the switches; security flag can precede its guard battle |
| `$1005C` bit 5 | Auxiliary console door unlocked | Opens the ladder route to Factory security |
| `$101D0` bits 0 / 1 | Zabie password accepted / shutdown escape completed | X-A-B-Y opens the power core; both bits present after the Factory |
| `$10212/10214/10216` | Zabie local script variables, bus `$7F0212/0214/0216` | X/A/B set the bytes to 1; held button rows stop after acknowledgement, Y requires the persistent password flag |
| `$007E` | Battle NMI mode | Mode 0 is setup; read initialized battler records in modes 1/2 |
| `$AF0A..AF14` | Initial battler identities, FF = unused | Party slots 0..2, enemy slots 3..10; identities survive KOs |
| `$96F5..96FF` | Battler presence | 1 present, 0 removed; field overwrites these, so require battle mode |
| `$5E30+slot*80` | Live HP/max HP/MP/max MP, four words | Damage, healing, MP spending, enemy KOs and party defeat |
| `$95D5`, `$95DB`, `$95DC+pc` | Active actor, submenu, command row | Main submenu 0; Attack/Tech/Item rows 0/1/2 |
| `$93EE+pc*7` bit 7 | PC waiting for command | Require a living, present active actor |
| `$1A80+pc*8C` | Twenty seven-byte tech entries per PC slot | ID +0; MP costs for PC slots 0/1/2 at +3/+5/+6; 80+ excludes that slot |
| `$9EE3`, `$9EE5`, `$A099` | Selected tech ID / usability / grading active | Selected-entry copy lags cursor changes; wait before checking unavailable bit 7 |
| `$1580+index*5`, `$95E5`, `$95E6` | Battle item records / scroll and cursor | Tonics BD/BE/BF, Ethers C0/C1/C2, Revive C7; scroll plus cursor selects inventory record |
| `$9609`, `$9614`, `$99C0+cursor` | Target selector active / cursor / candidate slot | Attack, tech and item targets |
| `$101EC` | Last battle result | 0 also precedes combat and can follow escape; 1 on observed defeat |
| `$0067`, `$0054`, `$0F00`, `$0FC6` | Field menu page / cursor / equipment offers / offer count | Normal menu page 0, outfit page 1; character then slot/list cursors |
| `$1040..104A`, `$1000`, `$6E00+item*2` | Shop cursor/count/total/list length, IDs, prices | Mode 0C; UP adds ten, RIGHT adds one; check total before purchase |
| `$0155`, `$1D1E/$1D1F` | Leader level / coordinate masks | Planner uses levels, half-tile shapes and one-way slides |
| `$0F00+slot`, `$1100+slot`, `$1B01+slot` | Object enabled / hidden / solid | Exclude party objects; other NPCs can block narrow passages |
| `$0BCB`, `$0BCD` | Field map width/height | Plane export dimensions |
| `$3000`, `$7000`, `$7040`, `$7080` | Metatiles, attributes A/B, exit indices | Row stride 256; shapes from `$30C0+tile` |
| `$4000` | Overworld layer-1 map | 96 by 64 bytes |
| `$1B38`, `$7E00` | Overworld entrance count / seven-byte records | Live entrance triggers |

Battle entry runs in C1, but the installed NMI runs in D1. Unknown modes
remain `unknown`; field coordinates are suppressed during combat. `hp` and
`mp` use live battler records in combat, persistent records elsewhere.
`battlers` retains original participants with current HP and presence.

An explicit opening `battle` row requires witnessed enemies, zero HP for
every enemy slot, result 0, a surviving party and settled field control.
The first ambush also requires its event flag. Routed rows can escape
ordinary encounters after enemy HP stalls; boss orders disable that
heuristic because healing/menu recovery can pause damage for longer than
the timer. Guardian requires entering the food locker with control
restored; its victory trace must also contain a zero-HP Guardian (D4).
Neither story 35 nor its pre-battle flag proves a win.
R-Series orders disable escape, prioritize middle enemy slots for area
attacks, and use an Ether when the acting member has fewer than four MP.
Story 42 is assigned after the win; acceptance also requires a victory
trace with all six R-Series (F8) at zero HP. Robo's return briefly displays
an overworld with half-scale coordinates and control disabled; confirm
through this scripted sequence rather than planning overworld movement.

The ROM suite exercises the canyon wins, a no-attack defeat, stale result
rejection, the full route through the Factory, and exact controller replay.
`battles.json` records changes with frame numbers. The scenario has no
ending detector; reports explicitly set `game_complete: false`.

ROM file offsets: location headers at `$360000`, 512 records of 14 bytes;
exit offsets and seven-byte records in bank `$250000`, 513 pointers. Export
validates pointer ordering, record alignment, and ROM identity.

Primary sources consulted (development references, no runtime dependency):

- [ChronoRET engine RAM and ROM structures](https://github.com/ThisIsAkill/ChronoRET/blob/1ac6f9043646129b74559490556f9f47c9c4f04b/asm/include/ram_engine.inc).
- [ChronoRET battle RAM structures](https://github.com/ThisIsAkill/ChronoRET/blob/1ac6f9043646129b74559490556f9f47c9c4f04b/asm/include/ram_battle.inc)
  and [battle NMI installation](https://github.com/ThisIsAkill/ChronoRET/blob/1ac6f9043646129b74559490556f9f47c9c4f04b/asm/bankD1/bankD1.asm).
- [ChronoRET menu/overworld structures](https://github.com/ThisIsAkill/ChronoRET/blob/1ac6f9043646129b74559490556f9f47c9c4f04b/asm/include/ram_menu.inc).
- [Temporal Redux character and flag definitions](https://github.com/OnemusCT/temporal-redux/blob/d8d4e38c0bf08b51a29441ef7ea2a489d63da58f/sourcefiles/practice/memorylocationdefs.py).
- [Temporal Redux event decoder](https://github.com/OnemusCT/temporal-redux/blob/d8d4e38c0bf08b51a29441ef7ea2a489d63da58f/sourcefiles/jetsoftime/ctevent.py),
  used locally to inspect the fair announcement, candy, telepod, pendant,
  first-gate, and canyon ambush events. The flags above were then observed
  in the ROM run.

Optional development clones live under ignored `refs/`. No third-party
source is vendored into the bot; map layouts are implemented from these
documented structures and checked against the local ROM.
