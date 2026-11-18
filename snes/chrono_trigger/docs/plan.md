# Campaign plan

Next action: continue at the End of Time from `RoboRecovery`: Gaspar,
Spekkio and magic, then the onward era gates. Add bounded rows to
`route.FUTURE`, prove them from the real predecessor, and replay from
power-on. Verified progress and recordings live in [STATUS.md](STATUS.md).

The route now includes Arris Dome: Guardian, Seed, reverse rat chase,
three L+R+A password switches, Lavos's recording, Seed exchange for the
Bike Key, and an Enertron rest before returning to the 2300 AD overworld.
The Guardian row requires the food locker with control restored; story
`35` and event bit `$7F00EC.0` are set before combat and cannot prove a win.
The extension completes a Lab 32 race attempt and crosses on foot,
recruits Robo with Lucca, opens the Factory consoles and X-A-B-Y lock,
shuts down the generator and defeats all six R-Series. The scripted
return repairs Robo and takes the four-person party to the End of Time.

## Next route work

1. End of Time: finish Gaspar's guidance, visit Spekkio, learn magic,
   verify party selection and the onward era gates. Current party is
   Crono / Marle / Lucca, with Robo in reserve after the arrival dialogue.
2. Build the remaining required campaign graph and Lavos route. Each
   segment needs resource costs, event prerequisites and a natural entry.
3. Implement the ending/credits predicate, then prove reset-to-ending
   without state loads or RAM writes before recording completion footage.

## Engine residuals

- Field-menu items, including Shelter use at save points, remain unbuilt.
- R-Series starts with low MP after the Factory guards. Its battle orders
  use Cyclone / Flame Toss, Ethers below four MP, and enemy slot priorities.
  The controller does not coordinate readiness for dual techs; allowing
  Fire Whirl lost the full power-on fight despite narrower practice wins.
  Keep resource checks when extending the natural predecessor.
- The verified endpoint has no Tonics or Mid Tonics, one Ether and one
  Revive. Verify a rest/healing option and restock before onward fights.
- Lab 32 uses the foot crossing after a race attempt. A direct race win
  remains unverified and is optional route optimization.
- The Factory's X-A-B-Y checks need held inputs until acknowledgement;
  two-frame pulses can miss the script's polling loop. Local variables
  `$7F0212/0214/0216` acknowledge X/A/B; `$7F01D0.0` proves the full code.
- After carrying Robo, an overworld appears with half-scale coordinates
  and control disabled. The recovery row only confirms through the whole
  scripted return and gate trip; trying to navigate there finds no path.
- Dialogue currently confirms A; a row can pulse a direction, but reliable
  indexed choice selection and cancel/retry semantics remain unverified.
- Lab 16 East has a refused step not explained by the collision model.
  Remembered edges and rerouting pass it; retain the uncertainty.
- The forward Arris rat has no verified chase. The route crosses south
  and catches the reverse rat (object 13). Chase rows replan every four
  frames, run on long legs, and confirm frequently nearby. A 24-frame
  target refresh failed at natural entry by stopping at stale positions.
- Solid NPCs can disconnect a route before an event moves them. Use a
  reachable `via` point to start the event; the planner then retries.
  After the Bike Key handoff, revisit the basement to reset the crowd's
  positions before resting and leaving Arris.
- Object dialogue can temporarily display an overworld during a
  cutscene. The Composer waits through it instead of planning toward a
  field NPC on the overworld.
- Mandatory boss rows disable automatic escape. The full-route ROM test
  caught Yakra's healing/menu recovery exceeding the enemy-HP stall timer;
  the narrow Yakra replay now passes and the full test guards that failure.
- Complete instrumentation of later menus, status effects, party changes
  and failure recovery as they become actual route requirements. Full-game
  completion remains unimplemented; reports say `game_complete: false`.

## Completion contract

Start from a fresh power-on/new game and finish after the chosen Lavos
clear reaches credits. Default names; controller input only; read-only
RAM and ROM geometry allowed (Bronze / Clean). No New Game+, injected
inventory, warps, stat refills, or mid-run state restores. Practice pin
loads and the explorer's state restores remain explicitly separate from
continuous evidence. Keep raw maps, screenshots, reports and reference
clones game-local and ignored.
