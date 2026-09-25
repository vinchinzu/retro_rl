# Plan: Mortal Kombat

Verified limits are in `docs/STATUS.md`. Do not call a save-state eval, a
Continue accept, or the Match 1-7 tape a credits clear.

## Next

1. Finish courtyard Endurance 1 from the Fight 7 pin with controller input
   only. The second fighter appears after two round wins (`match_counter`
   7 to 8), not after one KO. Do not treat throne-room Match 5 Kano, or
   leftover pin HUD (`hp=59/0`, rounds 2-0), as that fight.
2. Eval any promotion candidate at N>=20. Do not `--promote` earlier. Do not
   retarget v3 x/y off `0x00DA` without a new train.
3. Goro and Shang Tsung are still open after Endurance.
4. Credits only after one continuous power-on run that reaches them. Until
   then, do not raise the STATUS result.
