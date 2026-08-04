# Game Selection Notes

Condensed research notes for choosing games. The live board is
[GAME_MATRIX.md](GAME_MATRIX.md). Capability phases are
in [DEVELOPMENT_LADDER.md](DEVELOPMENT_LADDER.md).

## What makes a good early automation target

Prefer games with linear progression, few meaningful actions, stable cameras,
clear completion flags, low randomness, little inventory, readable RAM for
player/enemy pose, and fast recovery. Short human playtime alone is not enough.

## Candidate bands

### Easiest / pipeline fixtures

- Great Waldo Search — cursor/menus; verified continuous clear
- Simple fighting-game match states — harness validation
- Short educational titles — optional menu/state-machine practice

### Linear combat (near-term factory)

- TMNT IV — reference continuous clear
- Final Fight — primary generalization target
- Super Double Dragon, Rival Turf!
- Knights of the Round — later

### Continuous control

- F-Zero, Pilotwings, Star Fox
- Battle Clash — only after Super Scope injection exists

### Platforming trunk

- Magical Quest, Joe & Mac
- Super Mario World (`SMW/`), Donkey Kong Country
- Mega Man X — should be added
- Aladdin, Tiny Toon, Run Saber — later candidates from archive research

### Graph navigation

- Super Metroid — active
- A Link to the Past (`alttp/`) — title → castle grounds active
- Soul Blazer, Goof Troop — later

### Long campaigns

- Chrono Trigger, Final Fantasy IV, Super Mario RPG, EarthBound

### Planning-heavy

- Harvest Moon (`harvest/`) — first scheduling / long-horizon foothold
- Tactics Ogre, Ogre Battle, Uncharted Waters, Civilization

### Adaptive / procedural frontier

- Shiren the Wanderer
- Randomized or unseen scenarios
- Unseen-game generalization

## Hard games that are poor early targets

Familiarity is not ease: Super Castlevania IV, Contra III, Wild Guns, and dense
RNG-heavy titles belong later. Famous RPGs are long but often more structured
than procedural games — do not postpone them only because human playtime is
large.

## Unreleased / import notes

Import-only and unreleased titles can be useful for generalization research
once the shared trunks exist. Keep them out of the near-term completion board
unless a ROM and evaluation contract are actually available in-repo.
