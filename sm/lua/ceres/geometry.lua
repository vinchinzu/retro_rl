-- Ceres geometry constants and room-chain tables.
--
-- Named elev / magnet bands and pose sets used by reactive arm-pump navigation.
-- Do not re-encode these thresholds inline in controllers.

local takeoff = require("takeoff")
local rooms = require("rooms")

local M = {}

-- Classic arm-pump period — owned by takeoff, aliased for Ceres callers.
M._CERES_ARM_PUMP_PERIOD = takeoff.DEFAULT_PUMP_PERIOD or 2
-- Elevator geometry (smaller y = higher on screen).
-- Falling→elev mid-transition can still show y≈139; gs=8 remaps to bottom ~651.
-- Outbound first room (wiki Ceres 1 / Sniq 100% lsnes): pad y=72, short
-- RIGHT hop, air-turn pose 25→26 at ~x142 y79, land y=75 (pose 229), spinning
-- moonfall. Weave idles past 171/267. Extra L at p17 x=205 is not TAS 8782
-- B+RIGHT at (206, p17); skip once. Last-floor L on p17 carries leftover
-- pose 17 at (39, 139). That pose-17 is held-aim; TAS leftover 17 + B+RIGHT
-- is pose 15 at x=44, ours is still pose 9.
M._CERES_FIRST_PAD_Y = 72
M._CERES_FIRST_TURN_X = 142
M._CERES_FIRST_TURN_Y = 76
M._CERES_FIRST_FLOOR_Y = 640
M._CERES_FIRST_DOOR_X = 230
-- First inverted pulse (p17 + tape L). Do not L-every-p17 after 214.
M._CERES_FIRST_INVERT_L_X = 205
M._CERES_FIRST_INVERT_L_X_END = 214
-- gs=9 through last gs=11 (Sniq 100% lsnes f8789–8949). B+RIGHT+R is the
-- last fade frame; an L-pulse on the lip triggers 1px early (237 not 238).
M._CERES_FIRST_DOOR_FADE = 161
M._CERES_ELEV_SHIP_Y = 80  -- grounded ship pad band (product leave ~x145 y75 pose 2/10)
M._CERES_ELEV_SHIP_X = 145  -- product pad center before gs=32 Ceres-success
M._CERES_ELEV_TOP_Y = 171  -- s10 land / right-wall KB band
M._CERES_ELEV_TOP_X = 211  -- shaft right wall; the entry wall jump kicks off it
M._CERES_ELEV_LEDGE_Y = 571  -- mid-shaft ledge; not a product recovery seat
M._CERES_ELEV_BOTTOM_Y = 640  -- bottom floor band after a missed door jump
-- Shaft climb from the Falling-door entry (measured off ceres_first_control
-- in scratch/ceres_elev_wj). The entry arrives four air frames into a spin
-- jump, so the entry rise alone tops out at y=608 and the y=475 ledge is only
-- reachable by riding the `_CERES_ELEV_TOP_X` right wall and kicking off it. Sniq's own
-- elev_wj tape replays to the right places from this entry but never latches:
-- it releases A for a single frame and stable-retro needs two.
--
-- Above 475 the rungs are ordinary ground spin jumps between ledges, not wall
-- jumps — a full ground spin jump rises 111px and the gaps are 112/96/96.
-- Each launch x is the middle of its measured band; outside the band the jump
-- clips a ledge lip and drops back down the shaft.
M._CERES_ELEV_ENTRY_RISE_FRAMES = 16  -- RIGHT+A up the wall before the kick
M._CERES_ELEV_WJ_RELEASE_FRAMES = 2  -- LEFT without A; 1f reads as a jump cut
M._CERES_ELEV_WJ_KICK_FRAMES = 8
M._CERES_ELEV_WJ_RIDE_FRAMES = 34  -- pose 132 / movement type 20 carries to 474
M._CERES_ELEV_475_LAUNCH_X = 137  -- band 130-144, RIGHT onto 363
M._CERES_ELEV_363_LAUNCH_X = 191  -- band 185-211, LEFT onto 267
M._CERES_ELEV_267_LAUNCH_X = 144  -- band 132-156, LEFT onto 171
M._CERES_ELEV_171_LAUNCH_X = 48  -- band 41-55, RIGHT onto the ship pad (gs 32)
-- Falling west door (TAS lsnes sniq_100 f12902–12908). Jump at x=46 y=139
-- mx=2 inv=42, not a shutter crouch: TAS never DOWN on this ledge. 4th air
-- is (26, 120) pose 25 vy=+4 inv=36 after a B+RIGHT+A air-turn at (31, 129).
-- Dest gs=8 is (216, 632) pose 25 mx=2 inv=36. Walking $E23F is pose-138 /
-- mt=21 (mx=0, y=108 ceiling). min_momentum is 1: ground mx caps at 2.75
-- and halves on air frame 2. Floor remap y=651 is still a missed WJ.
M._CERES_FALLING_DOOR_LEDGE_Y = 139
-- TAS jumps at 46 mx=2 and 4th-airs at (26, 120). snes9x shutter wait dumps
-- mx to 1: jump at 42 undershoots the door (x=33 at y=120), jump at 33
-- overshoots (x=23). x=36 puts 4th air on the door slot.
M._CERES_FALLING_DOOR_JUMP_X = 36
-- TAS never crouches; snes9x $E23F is still shut at the TAS x=46 jump.
M._CERES_FALLING_DOOR_SHUTTER_FRAMES = 10
M.CERES_FALLING_DOOR_HOP = takeoff.platform_hop(
  M._CERES_FALLING_DOOR_LEDGE_Y,
  16,
  70,
  takeoff.takeoff_window({30, M._CERES_FALLING_DOOR_JUMP_X}, "LEFT", {min_momentum = 1})
)
-- Reverse Falling (sniq_100_lsnes_rich f12792–12908): run off 139 onto 187,
-- hop x=357 onto 171 at x=334, run LEFT *past* the 294 jet, turn RIGHT at
-- x=250 leftover LEFT mx, 1f p38 B+A+X, LEFT+A into the jet at (236, 162)
-- p83 inv=95. Door leave is the 4th air frame above, not (19, 121) p26.
M._CERES_FALLING_REV_FLOOR_Y = 187
M._CERES_FALLING_REV_SHELF_Y = 171
M._CERES_FALLING_REV_TILE_X = 300
-- TAS f12844 B+RIGHT at (250, 171) p16. 314 is still a LEFT shelf run.
M._CERES_FALLING_REV_TURN_X = 250
M.CERES_FALLING_REV_FLOOR_HOP = takeoff.platform_hop(
  M._CERES_FALLING_REV_FLOOR_Y,
  330,
  380,
  takeoff.takeoff_window({342, 352}, "LEFT", {min_momentum = 1})
)
-- Outbound Falling Tile (wiki Ceres 2 / Sniq 100% lsnes): run off the y=139
-- entry, short-hop the y=187 floor at x≈155–168 with LEFT+RIGHT+A so magnet
-- feet plant y=171, L-pump the shelf, then jump at x≈330–345 into the right door.
--
-- Both takeoffs are late on purpose. Air carries 3.4 px/f, the y=171 shelf run
-- carries 5.2–5.8 px/f, so every frame spent airborne instead of running is
-- lost ground. The floor hop opens at 155 (not 145) because the shelf face is
-- a wall at x=172: hopping earlier only buys float in front of it. The exit
-- hop opens at 330 (not 320) because the shelf runs out at x≈344 — leaving at
-- 320 flies 10 px of runway that could have been run.
M._CERES_FALLING_OUT_ENTRY_Y = 139
M._CERES_FALLING_OUT_FLOOR_Y = 187
M._CERES_FALLING_OUT_PLAT_Y = 171
M._CERES_FALLING_OUT_DOOR_X = 470
-- Shelf face; walking the y=187 floor past it is a pose-207 wall stall.
M._CERES_FALLING_OUT_SHELF_FACE_X = 172
M._CERES_FALLING_OUT_HOP_X = 155
M._CERES_FALLING_OUT_TAKEOFF_X = 330
-- Sniq 8977 `B+RIGHT+X`: one shot on the 4th air frame of the magnet-feet hop
-- unspins pose 25 → 19 (movement type 3 → 2). The taller unspun box catches
-- the y=171 lip three frames sooner than the spin does. This is a shot for its
-- collision box, not for anything it hits.
M._CERES_FALLING_OUT_UNSPIN_AIR_FRAME = 4
M.CERES_FALLING_FLOOR_HOP = takeoff.platform_hop(
  M._CERES_FALLING_OUT_FLOOR_Y,
  140,
  180,
  takeoff.takeoff_window({M._CERES_FALLING_OUT_HOP_X, 168}, "RIGHT", {min_momentum = 1})
)
M.CERES_FALLING_EXIT_HOP = takeoff.platform_hop(
  M._CERES_FALLING_OUT_PLAT_Y,
  300,
  360,
  takeoff.takeoff_window({M._CERES_FALLING_OUT_TAKEOFF_X, 345}, "RIGHT", {min_momentum = 1})
)
-- Falling east door ($E23F, the same species as the west shutter). It is a
-- pure proximity trigger: nothing happens until Samus x reaches
-- `_CERES_FALLING_EAST_DOOR_TRIGGER_X`, and from that frame the shutter runs a
-- fixed 5-phase spritemap cycle (63839 → 63881 → 63923 → 63955 → 63987 →
-- 63839) before the x=467 face is passable. Shooting it does not shorten the
-- cycle — a 60f shot volley from x=400 never starts it at all.
--
-- Arriving early is the expensive mistake, not arriving slow: the closed face
-- is pose-137 / movement-type-21 contact that zeroes momentum, so a full-dash
-- approach loses the stall *and* the whole dash rebuild behind it. Dashing
-- from the trigger covers the 40 px to the face in ~13f, well inside the
-- cycle. Walking (RIGHT, no dash) for `_CERES_FALLING_EAST_DOOR_WALK_FRAMES`
-- first and dashing after puts Samus on the face the frame it opens, at full
-- speed, and never touches it.
M._CERES_FALLING_EAST_DOOR_TRIGGER_X = 427
M._CERES_FALLING_EAST_DOOR_FACE_X = 467
M._CERES_FALLING_EAST_DOOR_OPEN_FRAMES = 21
M._CERES_FALLING_EAST_DOOR_WALK_FRAMES = 16
-- Magnet escape: leave door height ~y139; outbound mid ~y395.
do
  local src = debug.getinfo(1, "S").source:gsub("^@", "")
  local dir = src:match("^(.*)/[^/]+$") or src:match("^(.*)\\[^\\]+$") or "."
  M.CERES_DATA_DIR = dir .. "/data"
end
-- Stair-top magnet-stop plants x≈85 y=347 mx=0. Jump window is HIGH_HOP (70, 78).
M._CERES_MAGNET_STOP_X = 92
-- East steam burns on this pin; wait before the door hop.
M._CERES_MAGNET_DOOR_STEAM_FRAMES = 6
-- Outbound Magnet Stairs (wiki Ceres 3 / Sniq 100% lsnes): from the west
-- door (39, 139) run RIGHT, jump the y=139 ledge at x≈132–140, idle-spin
-- past the lip, DOWN then 1f RIGHT then LEFT onto y=219 (~x192), run LEFT,
-- jump the slope at y≈260 x≈124, X-unspin air-frame 2, land y=347, RIGHT
-- to the east door ~(236, 395). Steam at x~177 is 1f L then B+LEFT+X —
-- LEFT+B+X from a RIGHT dash is pose 37 and zeroes momentum.
-- https://wiki.supermetroid.run/KPDR_Room_Strategies#Ceres_3
-- Reverse (Sniq 100% lsnes magnet_escape): no east-door jump. Run 395
-- stairs onto 347, jump x≈68–82 LEFT then air-turn RIGHT onto 267. Run
-- RIGHT, jump x≈112–128, steam d-boost to 219, jump to 139. Do not jump
-- x≤50 (west magnet-stop) or x≥85 from 347 (267 underside).
M._CERES_MAGNET_TOP_Y = 139
M._CERES_MAGNET_MID_Y = 219
M._CERES_MAGNET_SHELF_Y = 267
M._CERES_MAGNET_SLOPE_Y = 255
M._CERES_MAGNET_BOT_Y = 347
M._CERES_MAGNET_DOOR_Y = 395
M._CERES_MAGNET_OUT_DOOR_X = 230
-- Both takeoffs are late on purpose, same reason as Falling: air is 3.4 px/f
-- and the ledges run 5+ px/f. Jumping the 139 lip at 126 only buys float in
-- front of the drop; TAS leaves at 134. The slope jump opens at y=260
-- (not 255) so the 219 shelf is run rather than flown.
M._CERES_MAGNET_OUT_TOP_HOP_X = 132
M._CERES_MAGNET_OUT_MID_HOP_Y = 260
-- Sniq 9314 `B+LEFT+A+X`: one shot on the 2nd air frame of the slope hop
-- unspins pose 26 → 20. Same collision-box trick as Falling's 4th-air-frame
-- unspin, not a shot at anything it hits.
M._CERES_MAGNET_OUT_MID_UNSPIN_AIR_FRAME = 2
-- Hop-1: idle-spin until the 139 lip is gone, DOWN on the way down, 1f RIGHT
-- at y≈151 then LEFT so the land is x≈192 not the x=219 overshoot.
M._CERES_MAGNET_OUT_HOP1_DOWN_Y = 142
M._CERES_MAGNET_OUT_HOP1_TURN_Y = 151
-- Hop-2: A-hold until y≈268, DOWN+A, 1f LEFT then RIGHT onto 347.
M._CERES_MAGNET_OUT_HOP2_DOWN_Y = 268
M._CERES_MAGNET_OUT_HOP2_TURN_Y = 281
-- Dash-kill then LEFT+B+X. Sniq's L-then-pose-74 gun is 1f on bsnes; on
-- this leftover LEFT+B+X is pose 37 (12f turnaround) either way, so the
-- dash-kill is what keeps the jet from eating the run. Do not L-pump
-- during the turnaround — RIGHT+B+L while facing left is moonwalk.
M._CERES_MAGNET_OUT_STEAM_X_LO = 160
M._CERES_MAGNET_OUT_STEAM_X_HI = 190
M._CERES_MAGNET_OUT_STEAM_RECOVER = 8
M.CERES_MAGNET_TOP_HOP = takeoff.platform_hop(
  M._CERES_MAGNET_TOP_Y,
  40,
  180,
  takeoff.takeoff_window({M._CERES_MAGNET_OUT_TOP_HOP_X, 145}, "RIGHT", {min_momentum = 1})
)
M.CERES_MAGNET_MID_HOP = takeoff.platform_hop(
  M._CERES_MAGNET_SLOPE_Y,
  110,
  145,
  takeoff.takeoff_window({118, 128}, "LEFT", {min_momentum = 1})
)
M.CERES_MAGNET_HIGH_HOP = takeoff.platform_hop(
  M._CERES_MAGNET_BOT_Y,
  40,
  M._CERES_MAGNET_STOP_X,
  takeoff.takeoff_window({70, 78}, "LEFT", {min_momentum = 1})
)
M.CERES_MAGNET_STEAM_HOP = takeoff.platform_hop(
  M._CERES_MAGNET_SHELF_Y,
  60,
  140,
  takeoff.takeoff_window({112, 128}, "RIGHT", {min_momentum = 1})
)
M.CERES_MAGNET_MID_ESCAPE_HOP = takeoff.platform_hop(
  M._CERES_MAGNET_MID_Y,
  170,
  210,
  takeoff.takeoff_window({180, 200}, "RIGHT", {min_momentum = 0})
)

-- Dead Scientist Room 0xE021: raised door alcoves (y≈139) over a pit (y≈187).
-- Sniq 100% lsnes never jumps: RIGHT+B+L/R off the left lip, run y=187, run
-- the right stairs, east door ~(492,139). A on the alcove bonks the ceiling.
-- Old floor takeoff x 350–410 caught the east ledge at _CERES_SCI_EXIT_LEDGE_X.
M._CERES_SCI_DOOR_Y = 139
M._CERES_SCI_FLOOR_Y = 187
M._CERES_SCI_ENTRY_LEDGE_X = 90
M._CERES_SCI_EXIT_LEDGE_X = 467

-- Outbound room chain (rightward).
M._CERES_OUTBOUND_CHAIN = {
  rooms.ROOM_CERES_ELEVATOR,
  rooms.ROOM_CERES_FALLING,
  rooms.ROOM_CERES_MAGNET,
  rooms.ROOM_CERES_SCIENTIST,
  rooms.ROOM_CERES_FLAT,
  rooms.ROOM_CERES_RIDLEY,
}
-- Escape reverse chain (leftward) before elevator shaft.
M._CERES_ESCAPE_CHAIN = {
  rooms.ROOM_CERES_RIDLEY,
  rooms.ROOM_CERES_FLAT,
  rooms.ROOM_CERES_SCIENTIST,
  rooms.ROOM_CERES_MAGNET,
  rooms.ROOM_CERES_FALLING,
  rooms.ROOM_CERES_ELEVATOR,
}

return M
