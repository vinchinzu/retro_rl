# Verified status

| Field | Value |
|---|---|
| Current maturity | M1 |
| Last verification | 2026-11-16 |
| Runtime / intervention | Bronze / Clean |
| Start / last goal | Fresh power-on -> Factory cleared, Robo repaired, End of Time arrival |
| Continuous route length | 218 rows, 445,242 frames (about 123m41s emulated) |
| State loads / RAM writes | 0 / 0 |

The route plays the opening and first gate, the 600 AD queen rescue and
Yakra, the trial/prison and Dragon Tank, and the 2300 AD chapter through
Arris Dome, Lab 32, Proto Dome and the Factory. It defeats Guardian,
solves the rat/password switches, watches Lavos's recording and obtains
the Bike Key. Lab 32 completes a race attempt and then crosses on foot;
no race win is claimed. Robo joins with Lucca while Marle stays behind.
The Factory route opens the left-side consoles and X-A-B-Y lock, shuts
down the generator, defeats all six R-Series, and carries Robo home.
Confirmation alone completes his repair and the four-person gate trip.
Every row passed continuously from its real predecessor.

Final RAM: location `1D0`, story `48`, party `00 01 02`, Crono HP `171/202`,
MP `8/28`, gold `3613`. Robo is repaired and in reserve after the default
party selection. Gate Key `D7` and Bike Key `D5` remain present; Seed `D4`
is removed. Control is live with no message, menu or battle active.
The run ends on the End of Time arrival platform, before Gaspar/Spekkio.
Tonics and Mid Tonics are exhausted; one Ether and one Revive remain.
Verify recovery and restocking before the next combat segment.

The original 190-row Arris prefix is unchanged: its report matches every
row and still ends at frame 326,577. Guardian's victory trace contains
zero HP for Guardian and both Bits, and its food-locker leave predicate
passes. R-Series has a victory trace with all six robots at zero HP;
story `42` follows the win. Zabie's persistent flag is `$7F01D0.0`, and
its shutdown escape bit is also set.

Verification: the 218-row power-on run passed. The complete ROM suite
and exact controller replay are being rechecked for this extension.
All 8 documentation tests passed before the final status update.
The R-Series/recovery segment also passed input offsets 1, 2, 3 and 7
from its saved natural entry. Mandatory bosses disable automatic escape;
R-Series uses Cyclone/Flame Toss, slot priorities and Ethers below four MP.
Allowing uncoordinated dual-tech attempts lost the first full-run fight,
so the final orders restrict attacks to those single-character techs.

Local evidence:

- [Continuous run report](../recordings/route-factory/report.json)
- [Controller tape](../recordings/route-factory/inputs.json)
- [Battle trace](../recordings/route-factory/battles.json)
- [Observed map viewer](../recordings/route-factory/atlas.html)
- [Final screen](../recordings/route-factory/final.png)
- [ROM test log](../recordings/rom-test-factory-final.log)
- [R-Series timing probes](../recordings/rseries-offsets.log)

Milestone pins and provenance live in
`custom_integrations/ChronoTrigger-Snes/`. Generated evidence is ignored;
the route and acceptance tests stay in source. ROM mapping decodes 512
location headers and 814 static exits; these remain geometry, not a
complete story graph.

The M2 instrumentation gate remains open: map and verify full-game
completion and the remaining modes/resources needed by the campaign.
The ending detector and credits route remain unimplemented; reports
explicitly set `game_complete: false`.

Next action: [Gaspar, Spekkio, magic and the onward era gates](plan.md).
