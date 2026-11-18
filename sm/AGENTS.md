# Agent Instructions — sm (lsnes Lua through Gravity)

**Parked 2026-11-16.** No run has written a Gravity leave here
(`recordings/gravity_leave.json` is a placeholder, `frame` 0). Super
Metroid work is in `snes/super_metroid/` on stable-retro, where the
emulator restores bit-exact for search; see its `AGENTS.md` and
`docs/plan.md`. Do not extend this tree unless the owner reopens it.

In-emulator Clean KPDR through Gravity Suit. Lua 5.1 loaded by lsnes
rr2-β25. Not the Python gym in `snes/super_metroid/`. Not `../sm_ceres`.
No Python files in this tree. Do not STATUS-promote.

## Launch

```bash
cd /tmp && TURBO=1 /home/v/01_projects/11_games/retro_rl/sm/run.sh
```

`run.sh` passes `--rom-type=snes --rom-a=`. Without the type, lsnes
loads a null core and `input.joyset` throws. Log is `./lsnes.log` in
the cwd, so run from `/tmp`. Headless is the default: Xvfb on a free
display and `GDK_BACKEND=x11`, because wxGTK ignores dummy SDL and
would map on the Hyprland session. `HEADED=1` is the only watch mode.

## Next

That command. The script is `spine.play_to_gravity`, starting at
`ceres.spine.play`. Stop at the first `error:` line in
`/tmp/lsnes.log`. Fix the controller that raised it. Run again from
power-on.

A Gravity leave is `gravity_leave.json` with nonzero `frame` and
`health` written by that process. `TIP=credits` waits on that leave.
Room changes print `room 0x....` from `run_gravity.lua`.

## Leave

RAM: room, game state, xy, pose, health, items, timer_type.
Ceres prefix: `0x91F8` gs=8 → `sm/recordings/ceres_leave.json`.
Gravity: `0xCE40` gs=8, `collected_items & 0x0020` →
`sm/recordings/gravity_leave.json`. No MP4. No STATUS.
