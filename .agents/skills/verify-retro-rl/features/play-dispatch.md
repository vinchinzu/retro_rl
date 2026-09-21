# Dispatch play scripts

`./play` is the root dispatcher that lists and forwards to game-local play scripts.

## Sub-features

- `play-list` prints the four dispatched games.
- `play-help` prints usage when given `--help`, `-h`, or no arguments.
- `play-unknown` rejects an unknown game id on stderr with exit `1`.

## How to get to it (user POV)

- Run `./play --list` from the repo root.
- Run `./play --help` or `./play` with no arguments.
- Run `./play <game> …` to forward into `nes/smb/play`, `snes/super_metroid/play`, `snes/alttp_rando/play`, or `snes/sm_rando/play`.

## Driving it with verify-retro-rl

Preconditions:

- Doctor reports `ok=true`.
- `./play` is executable at the repo root.
- This run will not start a headed emulator session.

- **List games.** Run `.cursor/skills/verify-retro-rl/scripts/cli --out $RUN/play-list -- ./play --list`. `exit.txt` is `0`. `stdout.txt` contains these four lines (spacing may vary, slugs must match):
  - `smb` and `nes/smb/play`
  - `sm` and `snes/super_metroid/play`
  - `alttp-rando` and `snes/alttp_rando/play`
  - `sm-rando` and `snes/sm_rando/play`
- **Help.** Run `.cursor/skills/verify-retro-rl/scripts/cli --out $RUN/play-help -- ./play --help`. `exit.txt` is `0`. `stdout.txt` contains `Usage: ./play <game>` and the example `./play sm morph full_start_v1`.
- **Unknown game.** Run `.cursor/skills/verify-retro-rl/scripts/cli --out $RUN/play-unknown -- ./play not-a-game`. `exit.txt` is `1`. `stderr.txt` contains `Unknown game: not-a-game` and `Try: ./play --list`.
- **Proof.** Keep the three step directories. Listing and help did not boot an emulator (no new PID, no `No romfiles found` in those two stdout files).

## Gotchas

- `./play sm` and `./play smb` start real play sessions and need ROMs. A green `--list` is not a Super Metroid or SMB clear.
- Aliases (`sm` / `super_metroid`, `alttp-rando` / `alttp_rando`) only show on help and in the `case` dispatcher, not as extra `--list` rows.
- `./play` with no args exits `0` and prints usage; that is help, not an error.
- Harvest, Zelda, and SMW are not `./play` games. Do not expect them on this list.
