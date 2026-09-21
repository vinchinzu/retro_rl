-- Ceres reverse: Magnet Stairs, Falling Tile, Elevator to ship.
--
-- Magnet: hop east steam, jump 347 at x≈74, air-turn RIGHT onto 267, run
-- RIGHT, jump 219 then 139. L-every-other while running. Wait on 267 when
-- the jet spritemap is idle.
--
-- Falling reverse: run off 139 onto 187, hop 357 onto 171, run LEFT past
-- 294, turn RIGHT at x≤250 leftover LEFT mx, 1f p38 X, LEFT+A into the
-- jet. Door: TAS jumps x=46 y=139, air-turns at (31, 129), leave (26, 120)
-- p25 inv=36.
--
-- Elevator: one TAS wall-jump climb. Fast entry is x216 y624–641 spin.
-- Missed wall jump is a hard fail. No checkpoint recovery. Shaft over
-- 2500f is a hang-cap fail, not a TAS pass.

local ram = require("ram")
local rooms = require("rooms")
local takeoff = require("takeoff")
local enemies = require("enemies")
local geom = require("ceres.geometry")
local skills_geom = require("skills.geometry")

local M = {}

local GS_ORDINARY = ram.GS_ORDINARY or 8
local POSE_WALL_LATCH = skills_geom.POSE_WALL_LATCH or 132
local _POSE_DBOOST = 80

-- Hang cap. TAS elev_to_landing is CERES_ELEV_BENCH_FRAMES. Do not grow this
-- for checkpoint recover (that run was 3349f).
M.CERES_ELEV_BENCH_FRAMES = 2246
M.CERES_ELEV_MAX_FRAMES = 2500

local function make_set(list)
  local s = {}
  for i = 1, #list do
    s[list[i]] = true
  end
  return s
end

local function as_set(t, fallback)
  if type(t) ~= "table" then
    return make_set(fallback)
  end
  if type(t[1]) == "number" then
    return make_set(t)
  end
  return t
end

local CROUCH_POSES = as_set(skills_geom.CROUCH_POSES, {37, 38, 39, 40, 41, 42})
local SPIN_POSES = as_set(skills_geom.SPIN_POSES, {25, 26, 27, 28})
local LAND_POSES = as_set(skills_geom.LAND_POSES, {163, 164, 165, 166, 167})
local STAND_LOCOMOTION_POSES = as_set(
  skills_geom.STAND_LOCOMOTION_POSES,
  {1, 2, 5, 6, 7, 8, 9, 10}
)

local ROOM_CERES_ELEVATOR = rooms.ROOM_CERES_ELEVATOR
local ROOM_CERES_FALLING = rooms.ROOM_CERES_FALLING
local ROOM_CERES_MAGNET = rooms.ROOM_CERES_MAGNET

local CERES_FALLING_DOOR_HOP = geom.CERES_FALLING_DOOR_HOP
local CERES_FALLING_REV_FLOOR_HOP = geom.CERES_FALLING_REV_FLOOR_HOP
local CERES_MAGNET_HIGH_HOP = geom.CERES_MAGNET_HIGH_HOP
local CERES_MAGNET_MID_ESCAPE_HOP = geom.CERES_MAGNET_MID_ESCAPE_HOP
local CERES_MAGNET_STEAM_HOP = geom.CERES_MAGNET_STEAM_HOP
local _CERES_ELEV_171_LAUNCH_X = geom._CERES_ELEV_171_LAUNCH_X
local _CERES_ELEV_267_LAUNCH_X = geom._CERES_ELEV_267_LAUNCH_X
local _CERES_ELEV_363_LAUNCH_X = geom._CERES_ELEV_363_LAUNCH_X
local _CERES_ELEV_475_LAUNCH_X = geom._CERES_ELEV_475_LAUNCH_X
local _CERES_ELEV_BOTTOM_Y = geom._CERES_ELEV_BOTTOM_Y
local _CERES_ELEV_ENTRY_RISE_FRAMES = geom._CERES_ELEV_ENTRY_RISE_FRAMES
local _CERES_ELEV_WJ_KICK_FRAMES = geom._CERES_ELEV_WJ_KICK_FRAMES
local _CERES_ELEV_WJ_RELEASE_FRAMES = geom._CERES_ELEV_WJ_RELEASE_FRAMES
local _CERES_ELEV_WJ_RIDE_FRAMES = geom._CERES_ELEV_WJ_RIDE_FRAMES
local _CERES_ELEV_SHIP_X = geom._CERES_ELEV_SHIP_X
local _CERES_ELEV_SHIP_Y = geom._CERES_ELEV_SHIP_Y
local _CERES_ELEV_TOP_Y = geom._CERES_ELEV_TOP_Y
local _CERES_FALLING_DOOR_LEDGE_Y = geom._CERES_FALLING_DOOR_LEDGE_Y
local _CERES_FALLING_DOOR_SHUTTER_FRAMES = geom._CERES_FALLING_DOOR_SHUTTER_FRAMES
local _CERES_FALLING_REV_FLOOR_Y = geom._CERES_FALLING_REV_FLOOR_Y
local _CERES_FALLING_REV_SHELF_Y = geom._CERES_FALLING_REV_SHELF_Y
local _CERES_FALLING_REV_TURN_X = geom._CERES_FALLING_REV_TURN_X
local _CERES_MAGNET_BOT_Y = geom._CERES_MAGNET_BOT_Y
local _CERES_MAGNET_DOOR_STEAM_FRAMES = geom._CERES_MAGNET_DOOR_STEAM_FRAMES
local _CERES_MAGNET_SHELF_Y = geom._CERES_MAGNET_SHELF_Y
local _CERES_MAGNET_TOP_Y = geom._CERES_MAGNET_TOP_Y

local CERES_STEAM_ID = enemies.CERES_STEAM_ID or 0xE1FF
local CERES_DOOR_ID = enemies.CERES_DOOR_ID or 0xE23F

-- Escape hops that fly through a steam jet's box. Each jet holds one long
-- dwell spritemap (~68f) and then runs a short eruption sequence (a handful of
-- maps, 2-3f each) that fires at the dwell's *end*. So the frame to leave from
-- is early in a dwell, not on a spritemap change: a change means the eruption
-- is starting, and the flight is still inside it fourteen frames later.
--
-- "Early in a dwell" is readable without knowing which map is the dwell — the
-- dwell is the only state that holds longer than the eruption's steps. An age
-- of at least `_CERES_MAGNET_JET_SETTLED` means the current map is the dwell;
-- at most `_CERES_MAGNET_JET_STALE` means the eruption is far enough off to
-- cross the box first. Measured on the 219 hop across a full cycle: launch
-- ages up to ~45 clear it, 49+ eat the knockback.
--
-- The 219→139 hop leaves x≈203 y=219 and passes the (194, 176) jet about
-- fourteen frames later. That knockback throws Samus back onto 219 with the
-- hop clock spent and the leg never recovers, so the wait is worth its frames.
local _CERES_MAGNET_JET_SETTLED = 25
local _CERES_MAGNET_JET_STALE = 44
local _CERES_MAGNET_MID_JET = {194, 176}
local _CERES_MAGNET_MID_JET_WAIT = 150
-- Pace this band while waiting rather than standing: `LEFT+B+A` from a dead
-- stop is a normal jump (movement type 2), which tops out at y≈148 and never
-- reaches the 139 ledge. The band stays east of the x=188 line that hands the
-- seat back to `steam_hop`.
local _CERES_MAGNET_MID_WAIT_X = 203

local function I(v)
  return tonumber(v) or 0
end

local function state_str(st)
  if st == nil then
    return "nil"
  end
  return string.format(
    "room=0x%04X gs=%s xy=(%s,%s) pose=%s mx=%s inv=%s",
    I(st.room_id),
    tostring(st.game_state),
    tostring(st.samus_x),
    tostring(st.samus_y),
    tostring(st.pose),
    tostring(st.momentum_x),
    tostring(st.invincibility_timer)
  )
end

local function timeout(msg)
  error("TimeoutError: " .. msg, 2)
end

local function replace(track, upd)
  local n = {}
  for k, v in pairs(track) do
    n[k] = v
  end
  for k, v in pairs(upd) do
    n[k] = v
  end
  return n
end

local function step(session, names, reason)
  if names == nil or #names == 0 then
    if session.idle then
      session:idle(reason)
    else
      session:step({}, reason)
    end
  else
    session:step(names, reason)
  end
end

local function hop_ready(hop, state)
  if hop == nil or type(hop.ready) ~= "function" then
    return false
  end
  return hop:ready(state)
end

local function hop_covers_y(hop, y, slack)
  if hop == nil then
    return false
  end
  if type(hop.covers_y) == "function" then
    if slack == nil then
      return hop:covers_y(y)
    end
    return hop:covers_y(y, slack)
  end
  slack = slack or 16
  return math.abs(I(y) - I(hop.y)) <= slack
end

local function hop_x_range(hop)
  local tw = hop and hop.takeoff
  if tw == nil then
    return 0, 0
  end
  local xr = tw.x_range
  if type(xr) == "table" then
    return I(xr[1] or xr.lo), I(xr[2] or xr.hi)
  end
  return I(tw.x_lo), I(tw.x_hi)
end

local function hop_x_hi(hop)
  return I(hop.x_hi or hop.xHi)
end

local function in_ceres_leave(gs)
  local t = ram.GS_CERES_LEAVE
  if type(t) == "table" and t[gs] then
    return true
  end
  return gs == 32 or gs == 33 or gs == 34
end

local function walk_toward_x(x, target, slack)
  if takeoff.walk_toward_x then
    return takeoff.walk_toward_x(x, target, slack)
  end
  slack = slack or 6
  if I(x) > I(target) + slack then
    return {"LEFT"}
  end
  if I(x) < I(target) - slack then
    return {"RIGHT"}
  end
  return {}
end

local function with_action(track, fn)
  function track:action(state)
    return fn(state, self)
  end
  return track
end

function M.CeresMagnetEscapeTrack(fields)
  fields = fields or {}
  return with_action({
    phase = fields.phase or "door",
    held = fields.held or 0,
    pump_i = fields.pump_i or 0,
    contacted = fields.contacted or false,
    steam_shown = fields.steam_shown ~= false,
    mid_fresh = fields.mid_fresh ~= false,
    mid_held = fields.mid_held or 0,
  }, function(state, track)
    return M.ceres_magnet_escape_action(state, track)
  end)
end

function M.CeresFallingEscapeTrack(fields)
  fields = fields or {}
  return with_action({
    phase = fields.phase or "door",
    held = fields.held or 0,
    pump_i = fields.pump_i or 0,
    contacted = fields.contacted or false,
    boosted = fields.boosted or false,
  }, function(state, track)
    return M.ceres_falling_escape_action(state, track)
  end)
end

function M._ceres_magnet_reached_falling(state)
  return I(state.room_id) == ROOM_CERES_FALLING and I(state.game_state) == 8
end

local function _ceres_grounded(state)
  return I(state.vertical_direction) == 0 and math.abs(I(state.velocity_y)) <= 1
end

local function _ceres_planted_near(state, y, slack)
  -- Natural Ceres shelf plant; movement type 0 is the one-frame land.
  slack = slack or 5
  return (
    I(state.game_state) == 8
    and math.abs(I(state.samus_y) - y) <= slack
    and _ceres_grounded(state)
  )
end

local function _tas_l_pump(direction, i, state)
  -- TAS magnet run: dir+B, L on odd frames, only once already running.
  -- Not L↔R period-2. Not a force-pump while accelerating.
  local running = I(state.speed_flag) ~= 0 or math.abs(I(state.momentum_x)) >= 1
  if running and i % 2 == 1 then
    return {direction, "B", "L"}
  end
  return {direction, "B"}
end

local function _steam_kb(state)
  -- Ceres steam knockback is mt=10 / timer, not Zebes pose 137/138.
  return I(state.knockback_timer) > 0 or I(state.movement_type) == 10
end

function M.ceres_magnet_escape_action(state, track)
  -- One-frame Magnet Stairs → Falling policy.
  local room = I(state.room_id)
  local gs = I(state.game_state)
  local x = I(state.samus_x)
  local y = I(state.samus_y)
  local bot = CERES_MAGNET_HIGH_HOP
  local steam = CERES_MAGNET_STEAM_HOP
  local mid = CERES_MAGNET_MID_ESCAPE_HOP

  if M._ceres_magnet_reached_falling(state) then
    return {}, replace(track, {phase = "done"})
  end
  if room == ROOM_CERES_FALLING then
    return {"LEFT"}, replace(track, {phase = "exit"})
  end
  if gs ~= 8 then
    return {"LEFT"}, replace(track, {phase = "door", held = track.held + 1})
  end
  if room ~= ROOM_CERES_MAGNET then
    return {"LEFT", "B"}, replace(track, {phase = "exit"})
  end

  local kb = _steam_kb(state)
  local grounded = _ceres_grounded(state)
  local planted_347 = grounded and math.abs(y - _CERES_MAGNET_BOT_Y) <= 8

  if track.phase == "door" or track.phase == "slope" then
    if kb then
      return {"LEFT", "B", "A"}, replace(track, {phase = "slope"})
    end
    if track.phase == "door" and track.held < _CERES_MAGNET_DOOR_STEAM_FRAMES then
      return {"LEFT", "B", "A"}, replace(track, {phase = "door", held = track.held + 1})
    end
    if planted_347 and (hop_ready(bot, state) or x <= 70) then
      return {"LEFT", "B", "A"}, replace(track, {phase = "shelf_hop", held = 1})
    end
    -- Door hop shifts subpixel vs TAS; start L on the other phase so
    -- the 347 east corner is not the magnet-stop tile.
    local pump_i = (track.phase == "door") and 1 or track.pump_i
    local names = _tas_l_pump("LEFT", pump_i, state)
    return names, replace(track, {phase = "slope", pump_i = pump_i + 1})
  end

  if track.phase == "shelf_hop" then
    if grounded and hop_covers_y(steam, y) then
      local names = _tas_l_pump("RIGHT", 0, state)
      return names, replace(track, {phase = "shelf", pump_i = 1, held = 0})
    end
    if grounded and math.abs(y - _CERES_MAGNET_BOT_Y) <= 8 then
      local names = hop_ready(bot, state) and {"B", "A"} or {"LEFT", "B", "A"}
      return names, replace(track, {held = track.held + 1})
    end
    if not grounded then
      local held = track.held + 1
      -- 267 underside is x≳65 y=332. Reach the shaft (x≲60), then
      -- RIGHT onto 267. Release A near y=267 so we plant, not fly over.
      if x <= 66 then
        if y <= 275 then
          return {"RIGHT", "B"}, replace(track, {held = held})
        end
        return {"RIGHT", "A"}, replace(track, {held = held})
      end
      if held <= 3 then
        return {"LEFT", "B", "A"}, replace(track, {held = held})
      end
      return {"A"}, replace(track, {held = held})
    end
    local names = _tas_l_pump("LEFT", track.pump_i, state)
    return names, replace(track, {phase = "slope", pump_i = track.pump_i + 1})
  end

  if track.phase == "shelf" then
    if grounded and hop_covers_y(steam, y) and 60 <= x and x <= 140 then
      if hop_ready(steam, state) and (track.steam_shown or track.held >= 24) then
        return {"RIGHT", "B", "A"}, replace(track, {phase = "steam_hop", held = 1})
      end
      if not track.steam_shown then
        local direction = (x >= 122) and "LEFT" or "RIGHT"
        local names = _tas_l_pump(direction, track.pump_i, state)
        return names, replace(track, {pump_i = track.pump_i + 1, held = track.held + 1})
      end
    end
    if grounded and hop_ready(steam, state) then
      return {"RIGHT", "B", "A"}, replace(track, {phase = "steam_hop", held = 1})
    end
    local names = _tas_l_pump("RIGHT", track.pump_i, state)
    return names, replace(track, {pump_i = track.pump_i + 1})
  end

  if track.phase == "steam_hop" then
    if grounded and hop_covers_y(mid, y) then
      if x < 188 then
        local names = _tas_l_pump("RIGHT", track.pump_i, state)
        return names, replace(track, {pump_i = track.pump_i + 1, held = 0})
      end
      if not track.mid_fresh and track.mid_held < _CERES_MAGNET_MID_JET_WAIT then
        local names = (x > _CERES_MAGNET_MID_WAIT_X) and {"LEFT", "B"} or {"RIGHT", "B"}
        return names, replace(track, {mid_held = track.mid_held + 1})
      end
      return {"LEFT", "B", "A"}, replace(track, {phase = "mid_hop", held = 1})
    end
    if grounded and y <= _CERES_MAGNET_TOP_Y + 8 then
      local names = _tas_l_pump("LEFT", 0, state)
      return names, replace(track, {phase = "exit", pump_i = 1, held = 0})
    end
    local contacted = track.contacted or kb
    if kb then
      return {"RIGHT", "B", "A"}, replace(track, {contacted = true, held = track.held + 1})
    end
    if not grounded then
      local held = track.held + 1
      if y <= 225 and x >= 160 then
        return {"RIGHT", "B"}, replace(track, {held = held, contacted = contacted})
      end
      return {"RIGHT", "B", "A"}, replace(track, {held = held, contacted = contacted})
    end
    local names = _tas_l_pump("RIGHT", track.pump_i, state)
    return names, replace(track, {
      phase = "shelf",
      pump_i = track.pump_i + 1,
      contacted = contacted,
    })
  end

  if track.phase == "mid_hop" then
    if grounded and y <= _CERES_MAGNET_TOP_Y + 8 then
      local names = _tas_l_pump("LEFT", 0, state)
      return names, replace(track, {phase = "exit", pump_i = 1, held = 0})
    end
    if grounded and hop_covers_y(mid, y) and x < 188 then
      local names = _tas_l_pump("RIGHT", track.pump_i, state)
      return names, replace(track, {
        phase = "steam_hop",
        pump_i = track.pump_i + 1,
        held = 0,
      })
    end
    if grounded then
      return {"LEFT", "B", "A"}, replace(track, {held = track.held + 1})
    end
    if y <= 150 then
      return {"LEFT", "B"}, replace(track, {held = track.held + 1})
    end
    if y <= 180 then
      return {"LEFT", "B", "A"}, replace(track, {held = track.held + 1})
    end
    return {"A"}, replace(track, {held = track.held + 1})
  end

  -- 139 west magnet-stop is pose 138 at x≈45. Hop the planted corner.
  local pose = I(state.pose)
  if grounded and y <= _CERES_MAGNET_TOP_Y + 8 and (pose == 137 or pose == 138) then
    return {"LEFT", "B", "A"}, replace(track, {phase = "exit", held = track.held + 1})
  end
  local names = _tas_l_pump("LEFT", track.pump_i, state)
  return names, replace(track, {phase = "exit", pump_i = track.pump_i + 1})
end

local function _magnet_jet_map(session, seat)
  -- Current spritemap of the steam jet parked on `seat`.
  local list = enemies.list(session)
  if list == nil then
    return nil
  end
  for i = 1, #list do
    local enemy = list[i]
    if I(enemy.enemy_id) == CERES_STEAM_ID then
      if math.abs(I(enemy.x) - seat[1]) <= 6 and math.abs(I(enemy.y) - seat[2]) <= 6 then
        return I(enemy.spritemap)
      end
    end
  end
  return nil
end

local function _magnet_steam_shown(session)
  -- True when a 267-height jet is on the live spritemap cycle.
  local list = enemies.list(session)
  if list == nil then
    return false
  end
  for i = 1, #list do
    local enemy = list[i]
    if enemies.steam_jet_shown(enemy) then
      if math.abs(I(enemy.y) - _CERES_MAGNET_SHELF_Y) <= 40 then
        return true
      end
      if math.abs(I(enemy.x) - 62) <= 16 and math.abs(I(enemy.y) - 304) <= 16 then
        return true
      end
    end
  end
  return false
end

function M.play_magnet_to_falling(session, max_frames)
  -- Magnet Stairs escape. One trajectory. Raises if Falling gs=8 misses.
  max_frames = max_frames or 800
  local track = M.CeresMagnetEscapeTrack()
  local mid_map = nil
  local mid_age = 0
  for _ = 1, max_frames do
    local st = session.state
    if M._ceres_magnet_reached_falling(st) then
      return
    end
    local seen = _magnet_jet_map(session, _CERES_MAGNET_MID_JET)
    if seen ~= nil and seen == mid_map then
      mid_age = mid_age + 1
    else
      mid_age = 0
    end
    mid_map = seen
    track = replace(track, {
      steam_shown = _magnet_steam_shown(session),
      mid_fresh = (
        _CERES_MAGNET_JET_SETTLED <= mid_age and mid_age <= _CERES_MAGNET_JET_STALE
      ),
    })
    local names
    names, track = M.ceres_magnet_escape_action(st, track)
    step(session, names, "ceres_magnet_" .. track.phase)
    if track.phase == "done" or M._ceres_magnet_reached_falling(session.state) then
      return
    end
    if track.phase == "shelf_hop" and track.held > 50 then
      timeout("ceres magnet 347 hop stalled: " .. state_str(session.state))
    end
    if track.phase == "steam_hop" and track.held > 60 then
      timeout("ceres magnet missed 219 from 267: " .. state_str(session.state))
    end
    if track.phase == "mid_hop" and track.held > 55 then
      timeout("ceres magnet 219 hop stalled: " .. state_str(session.state))
    end
  end
  timeout(
    string.format(
      "ceres magnet escape missed Falling after %df: %s phase=%s inv=%d",
      max_frames,
      state_str(session.state),
      tostring(track.phase),
      I(session.state.invincibility_timer)
    )
  )
end

function M._ceres_fast_entry_window(state)
  -- Natural door-jump phase from which the y=475 wall jump is repeatable.
  --
  -- y=651 floor remap is a missed wall jump. Do not widen this band.
  -- Falling leave uses this same predicate.
  --
  -- `momentum_x >= 1` is the measured floor, not a relaxation of a >= 2 that
  -- ever held: ground momentum caps at 2.75 and halves once on the second
  -- airborne frame, and the y band needs air frame 3, so no takeoff out of the
  -- Falling door can carry 2 into this window.
  local pose = I(state.pose)
  return (
    I(state.room_id) == ROOM_CERES_ELEVATOR
    and I(state.game_state) == GS_ORDINARY
    and 210 <= I(state.samus_x) and I(state.samus_x) <= 220
    and 624 <= I(state.samus_y) and I(state.samus_y) <= 641
    and SPIN_POSES[pose] == true
    and I(state.vertical_direction) == 1
    and I(state.velocity_y) > 0
    and I(state.momentum_x) >= 1
    and I(state.invincibility_timer) > 0
  )
end

function M._ceres_falling_reached_elev(state)
  -- TAS dest gs=8: (216, 632) pose 25 mx=2 vy=+4 inv=36.
  -- Same predicate as the elevator fast-entry window. y=651 / mx=0 / inv=0
  -- is a missed door jump, not a leave.
  return M._ceres_fast_entry_window(state)
end

function M.ceres_falling_escape_action(state, track)
  -- One-frame Falling Tile → elev policy.
  local room = I(state.room_id)
  local gs = I(state.game_state)
  local x = I(state.samus_x)
  local y = I(state.samus_y)
  local pose = I(state.pose)
  local floor = CERES_FALLING_REV_FLOOR_HOP

  if M._ceres_falling_reached_elev(state) then
    return {}, replace(track, {phase = "done"})
  end
  if room == ROOM_CERES_ELEVATOR then
    -- gs=8 without the TAS window is a miss. Do not hold A into y=651.
    if gs == 8 then
      return {}, replace(track, {phase = "exit"})
    end
    return {"A"}, replace(track, {phase = "exit"})
  end
  if room == ROOM_CERES_FALLING and (gs == 9 or gs == 11) and track.phase == "exit" then
    return {"A"}, replace(track, {held = track.held + 1})
  end
  if gs ~= 8 then
    return {"LEFT"}, replace(track, {phase = "door", held = track.held + 1})
  end
  if room ~= ROOM_CERES_FALLING then
    return {"LEFT", "B"}, replace(track, {phase = "exit"})
  end

  local kb = _steam_kb(state)
  local grounded = _ceres_grounded(state)
  local contacted = track.contacted or kb
  local boosted = track.boosted or pose == _POSE_DBOOST
  local held = track.held + 1

  if track.phase == "door" then
    if grounded and y <= _CERES_FALLING_DOOR_LEDGE_Y + 8 then
      if I(state.invincibility_timer) < 6 and track.held < 2 then
        return {"LEFT", "B", "A"}, replace(track, {phase = "door", held = held})
      end
      return _tas_l_pump("LEFT", 0, state), replace(track, {phase = "run_off", pump_i = 1, held = 0})
    end
    return {"LEFT", "B"}, replace(track, {phase = "run_off", pump_i = 1, held = 0})
  end

  local planted_187 = _ceres_planted_near(state, _CERES_FALLING_REV_FLOOR_Y, 8)
  local planted_171 = _ceres_planted_near(state, _CERES_FALLING_REV_SHELF_Y, 8)
  local hop_lo, hop_hi = hop_x_range(floor)

  if track.phase == "run_off" then
    if planted_187 then
      if hop_ready(floor, state) or (hop_lo <= x and x <= hop_hi) then
        return {"LEFT", "B", "A"}, replace(track, {phase = "floor_hop", held = 1})
      end
      local names = _tas_l_pump("LEFT", track.pump_i, state)
      return names, replace(track, {pump_i = track.pump_i + 1})
    end
    -- y≈171 on the 139→187 slope is x≈421, not the 171 shelf (x≤334).
    if planted_171 and x <= hop_x_hi(floor) then
      local names = _tas_l_pump("LEFT", 0, state)
      return names, replace(track, {phase = "shelf", pump_i = 1, held = 0})
    end
    -- TAS f12797–98: drop B at the 139 lip. Dash here is p42 and dumps mx.
    if x > hop_x_hi(floor) and 142 <= y and y <= 154 then
      return {"LEFT"}, replace(track, {pump_i = track.pump_i + 1})
    end
    local names = _tas_l_pump("LEFT", track.pump_i, state)
    return names, replace(track, {pump_i = track.pump_i + 1})
  end

  if track.phase == "floor_hop" then
    if planted_171 then
      local names = _tas_l_pump("LEFT", 0, state)
      return names, replace(track, {phase = "shelf", pump_i = 1, held = 0})
    end
    if not grounded then
      -- TAS unspins (p24) then plants 165. Air: B+A, then B+DOWN+A.
      if held == 2 then
        return {"B", "A"}, replace(track, {held = held})
      end
      if held == 3 then
        return {"B", "DOWN", "A"}, replace(track, {held = held})
      end
      return {"LEFT", "B"}, replace(track, {held = held})
    end
    if planted_187 then
      if hop_lo <= x and x <= hop_hi then
        return {"LEFT", "B", "A"}, replace(track, {held = 1})
      end
      local names = _tas_l_pump("LEFT", track.pump_i, state)
      return names, replace(track, {pump_i = track.pump_i + 1})
    end
    local names = _tas_l_pump("LEFT", track.pump_i, state)
    return names, replace(track, {phase = "run_off", pump_i = track.pump_i + 1})
  end

  if track.phase == "shelf" then
    if kb then
      return {"LEFT", "A"}, replace(track, {phase = "dboost", contacted = true, held = 1})
    end
    -- TAS f12844: B+RIGHT at (250, 171) leftover LEFT mx. Not 314, not 294.
    if planted_171 and x <= _CERES_FALLING_REV_TURN_X then
      return {"RIGHT"}, replace(track, {phase = "dboost", held = 1})
    end
    local names = _tas_l_pump("LEFT", track.pump_i, state)
    return names, replace(track, {pump_i = track.pump_i + 1})
  end

  if track.phase == "dboost" then
    if kb then
      return {"LEFT", "B", "A"}, replace(track, {contacted = true, boosted = true, held = held})
    end
    if pose == _POSE_DBOOST or pose == 83 or pose == 84 or (boosted and not grounded) then
      return {"LEFT", "B", "A"}, replace(track, {contacted = contacted, boosted = true, held = held})
    end
    if contacted and not grounded and y < _CERES_FALLING_REV_SHELF_Y then
      return {"LEFT", "B", "A"}, replace(track, {contacted = true, boosted = true, held = held})
    end
    if boosted and grounded and y <= _CERES_FALLING_DOOR_LEDGE_Y + 8 then
      local names = _tas_l_pump("LEFT", 0, state)
      return names, replace(track, {
        phase = "exit",
        pump_i = 1,
        held = 0,
        contacted = contacted,
        boosted = true,
      })
    end
    if boosted and planted_171 and x <= 90 then
      local names = _tas_l_pump("LEFT", 0, state)
      return names, replace(track, {
        phase = "slope",
        pump_i = 1,
        held = 0,
        contacted = contacted,
        boosted = true,
      })
    end
    if not contacted and planted_171 then
      -- TAS f12845–47: RIGHT+A, p38 B+A+X, LEFT+A on p25 into the jet.
      -- snes9x leftover is p75; LEFT+A there is p47 and misses the jet.
      if pose == 38 then
        return {"B", "A", "X"}, replace(track, {held = held})
      end
      if SPIN_POSES[pose] then
        return {"B", "LEFT", "A"}, replace(track, {held = held})
      end
      if held <= 2 then
        return {"RIGHT", "B", "A"}, replace(track, {held = held})
      end
      return {"B", "A"}, replace(track, {held = held})
    end
    if not contacted and not grounded then
      if SPIN_POSES[pose] or pose == 19 then
        return {"B", "LEFT", "A"}, replace(track, {held = held})
      end
      -- X-unspin is p47. Release A at y<=165 so the jet is met near y=162.
      if y <= 110 then
        return {"LEFT"}, replace(track, {held = held})
      end
      if y <= 165 then
        return {"B"}, replace(track, {held = held})
      end
      return {"B", "A"}, replace(track, {held = held})
    end
    if planted_171 then
      local names = _tas_l_pump("LEFT", track.pump_i, state)
      return names, replace(track, {
        phase = "slope",
        pump_i = track.pump_i + 1,
        contacted = contacted,
        boosted = boosted,
      })
    end
    if not grounded then
      return {"LEFT", "B"}, replace(track, {contacted = contacted, boosted = boosted, held = held})
    end
    local names = _tas_l_pump("LEFT", track.pump_i, state)
    return names, replace(track, {
      phase = "slope",
      pump_i = track.pump_i + 1,
      contacted = contacted,
      boosted = boosted,
    })
  end

  if track.phase == "slope" then
    if grounded and y <= _CERES_FALLING_DOOR_LEDGE_Y + 8 then
      local names = _tas_l_pump("LEFT", 0, state)
      return names, replace(track, {phase = "exit", pump_i = 1, held = 0})
    end
    local names = _tas_l_pump("LEFT", track.pump_i, state)
    return names, replace(track, {pump_i = track.pump_i + 1})
  end

  -- Door: TAS jumps at x=46. snes9x $E23F is still shut then (pose-138,
  -- mx=0, y=108 ceiling), so wait it out east of the face, then jump.
  -- Air-turn B+RIGHT+A at y≈129 so dest faces right.
  local door = CERES_FALLING_DOOR_HOP
  if gs == 9 or gs == 11 then
    return {"A"}, replace(track, {phase = "exit", held = held})
  end
  if grounded and y <= _CERES_FALLING_DOOR_LEDGE_Y + 8 then
    if track.held < _CERES_FALLING_DOOR_SHUTTER_FRAMES then
      return {"DOWN"}, replace(track, {held = held})
    end
    if (
      hop_ready(door, state)
      and I(state.invincibility_timer) > 0
      and pose ~= 137 and pose ~= 138 and pose ~= 210
    ) then
      return {"LEFT", "B", "A"}, replace(track, {phase = "exit", held = 1})
    end
    local names = _tas_l_pump("LEFT", track.pump_i, state)
    return names, replace(track, {pump_i = track.pump_i + 1, held = held})
  end
  if not grounded then
    if SPIN_POSES[pose] and 127 <= y and y <= 131 then
      return {"B", "RIGHT", "A"}, replace(track, {held = held})
    end
    -- TAS air-4 is B+A at y=124 then A into gs=9. LEFT after the air-turn
    -- faces away from the TAS dest and rides the y=108 ceiling.
    if y <= 126 then
      return {"A"}, replace(track, {held = held})
    end
    return {"LEFT", "B", "A"}, replace(track, {held = held})
  end
  local names = _tas_l_pump("LEFT", track.pump_i, state)
  return names, replace(track, {pump_i = track.pump_i + 1, held = held})
end

function M.play_falling_to_elev(session, max_frames)
  -- Falling Tile reverse. Raises if elev gs=8 misses.
  max_frames = max_frames or 500
  local rid = I(session.state.room_id)
  if rid ~= ROOM_CERES_FALLING and rid ~= ROOM_CERES_ELEVATOR then
    error("expected Falling after magnet: " .. state_str(session.state), 2)
  end
  local track = M.CeresFallingEscapeTrack()
  for _ = 1, max_frames do
    local st = session.state
    if M._ceres_falling_reached_elev(st) then
      return
    end
    if I(st.room_id) == ROOM_CERES_ELEVATOR and I(st.game_state) == 8 then
      timeout(string.format(
        "falling leave missed TAS elev window: %s phase=%s contacted=%s boosted=%s",
        state_str(st),
        tostring(track.phase),
        tostring(track.contacted),
        tostring(track.boosted)
      ))
    end
    local names
    names, track = M.ceres_falling_escape_action(st, track)
    step(session, names, "ceres_falling_" .. track.phase)
    if track.phase == "done" or M._ceres_falling_reached_elev(session.state) then
      return
    end
    if track.phase == "dboost" and track.held > 70 then
      timeout(string.format(
        "falling d-boost stalled: %s contacted=%s boosted=%s",
        state_str(session.state),
        tostring(track.contacted),
        tostring(track.boosted)
      ))
    end
    if track.phase == "floor_hop" and track.held > 40 then
      timeout("falling missed y171 shelf: " .. state_str(session.state))
    end
  end
  timeout(string.format(
    "falling missed elev after %df: %s phase=%s contacted=%s boosted=%s",
    max_frames,
    state_str(session.state),
    tostring(track.phase),
    tostring(track.contacted),
    tostring(track.boosted)
  ))
end

function M._ceres_elev_ship_band(state)
  -- Grounded on ship pad (product leave ~x145 y75 pose 2/10 → gs 32).
  return (
    I(state.room_id) == ROOM_CERES_ELEVATOR
    and I(state.game_state) == GS_ORDINARY
    and I(state.samus_y) <= _CERES_ELEV_SHIP_Y
    and math.abs(I(state.velocity_y)) <= 1
  )
end

function M.ship_pad_action(state)
  -- Walk through the Ceres pad x that starts gs 32.
  return walk_toward_x(I(state.samus_x), _CERES_ELEV_SHIP_X)
end

function M._ceres_elev_leaving(state)
  -- True ship leave: left the elevator, or Ceres success / Zebes load.
  -- Inbound Falling→elev door (gs 9/11, often fake y≈139) is not leave.
  if I(state.room_id) ~= ROOM_CERES_ELEVATOR then
    return true
  end
  return in_ceres_leave(I(state.game_state))
end

local function _ceres_elev_entry_action(state)
  -- Inputs until the shaft may start. nil means the window is ready.
  --
  -- Hold LEFT only while the Falling door is still settling. Ordinary y≈628
  -- spin is the TAS wall-jump phase. Walking LEFT there leaves x=216 and
  -- dumps the well. A floor remap (y≈651) is a missed wall jump.
  if M._ceres_elev_leaving(state) then
    return nil
  end
  if I(state.room_id) ~= ROOM_CERES_ELEVATOR then
    return {}
  end
  if I(state.game_state) ~= GS_ORDINARY then
    -- Door-transition inputs are not movement frames. Preserve the
    -- predecessor's pose-25 rise; the first ordinary frame owns the WJ.
    return {}
  end
  if M._ceres_fast_entry_window(state) then
    return nil
  end
  if I(state.samus_y) >= _CERES_ELEV_BOTTOM_Y - 20 then
    return nil
  end
  return {}
end

local function _ceres_elev_budget(session, start)
  local used = I(session.frame) - start
  if used > M.CERES_ELEV_MAX_FRAMES then
    timeout(string.format(
      "ceres elev_to_landing %df exceeded %df: %s",
      used,
      M.CERES_ELEV_MAX_FRAMES,
      state_str(session.state)
    ))
  end
end

local function _ceres_planted_at(state, target_y, slack)
  slack = slack or 18
  local pose = I(state.pose)
  return (
    I(state.game_state) == GS_ORDINARY
    and math.abs(I(state.samus_y) - target_y) <= slack
    and math.abs(I(state.velocity_y)) <= 1
    and (STAND_LOCOMOTION_POSES[pose] or LAND_POSES[pose])
  )
end

local function _ceres_dumped_well(state)
  return (
    I(state.room_id) == ROOM_CERES_ELEVATOR
    and I(state.game_state) == GS_ORDINARY
    and I(state.samus_y) >= _CERES_ELEV_BOTTOM_Y - 20
    and math.abs(I(state.velocity_y)) <= 1
  )
end

local function _ceres_elev_grounded(state)
  -- Standing or walking on a shaft ledge (movement types 0/1).
  local mt = I(state.movement_type)
  return (mt == 0 or mt == 1) and math.abs(I(state.velocity_y)) <= 1
end

local function _ceres_elev_walk_to(session, target_x, limit)
  -- Walk the current ledge onto a measured launch x.
  limit = limit or 90
  for _ = 1, limit do
    local names = walk_toward_x(I(session.state.samus_x), target_x, 1)
    if names == nil or #names == 0 then
      return
    end
    session:step(names, "ceres_elev_walk_" .. tostring(target_x))
  end
end

function M._ceres_door_blocks_wj(session)
  -- True when a Ceres-door overlay occupies the right-wall WJ contact.
  local st = session.state
  local sx, sy = I(st.samus_x), I(st.samus_y)
  local list = enemies.list(session)
  if list == nil then
    return false
  end
  for i = 1, #list do
    local enemy = list[i]
    if I(enemy.enemy_id) == CERES_DOOR_ID then
      if enemies.overlaps(enemy, sx, sy, 16) then
        return true
      end
    end
  end
  return false
end

local function _ceres_entry_to_475(session)
  -- One precise wall jump off the shaft right wall onto the y=475 ledge.
  --
  -- The door leave arrives four air frames into its spin jump, so the entry
  -- rise alone tops out at y=608 and no amount of drift reaches 475. Riding
  -- RIGHT+A up the x=211 wall to y≈571 and kicking off it does: release A for
  -- two frames pressing LEFT (one frame reads as a jump cut and never
  -- latches), then LEFT+A for the pose-132 kick, then hold A while movement
  -- type 20 carries the kick to y=474.
  if not M._ceres_fast_entry_window(session.state) then
    return false
  end
  session:hold(_CERES_ELEV_ENTRY_RISE_FRAMES, {"RIGHT", "A"}, "ceres_elev_entry_rise")
  session:hold(_CERES_ELEV_WJ_RELEASE_FRAMES, {"LEFT"}, "ceres_elev_entry_release")
  local latched = false
  for _ = 1, _CERES_ELEV_WJ_KICK_FRAMES do
    session:step({"LEFT", "A"}, "ceres_elev_entry_kick")
    latched = latched or I(session.state.pose) == POSE_WALL_LATCH
  end
  for _ = 1, _CERES_ELEV_WJ_RIDE_FRAMES do
    session:step({"A"}, "ceres_elev_entry_ride")
    latched = latched or I(session.state.pose) == POSE_WALL_LATCH
  end
  if not latched then
    return false
  end
  for _ = 1, 40 do
    step(session, {}, "ceres_elev_entry_land")
    if _ceres_dumped_well(session.state) then
      timeout("ceres elev dumped well at entry→475: " .. state_str(session.state))
    end
    if _ceres_planted_at(session.state, 475, 4) then
      return true
    end
  end
  return false
end

function M._ceres_ledge_hop(session, target_y, launch_x, side, limit)
  -- Walk a shaft ledge to `launch_x` and spin-jump onto `target_y`.
  --
  -- Above y=475 the rungs are ground spin jumps, not wall jumps: a full ground
  -- spin jump rises 111px against gaps of 112/96/96. What has to be right is
  -- the launch x — off its band the jump clips a ledge lip and drops back down
  -- the shaft, which raises here rather than remapping to a lower floor.
  limit = limit or 120
  _ceres_elev_walk_to(session, launch_x)
  session:hold(2, {side}, "ceres_elev_turn_" .. tostring(target_y))
  local airborne = false
  for _ = 1, limit do
    session:step({side, "A"}, "ceres_elev_hop_" .. tostring(target_y))
    local state = session.state
    if M._ceres_elev_leaving(state) then
      return true
    end
    if _ceres_dumped_well(state) then
      timeout("ceres elev dumped well aiming " .. tostring(target_y) .. ": " .. state_str(state))
    end
    local grounded = _ceres_elev_grounded(state)
    airborne = airborne or not grounded
    if airborne and grounded then
      return math.abs(I(state.samus_y) - target_y) <= 4
    end
  end
  return false
end

function M._ceres_elev_top_to_ship(session)
  -- From the y=171 seat: walk into the left wall, spin-jump onto the pad.
  --
  -- The climb lands 171 on its west end (x≈66), not the s10 east seat the old
  -- right-wall knockback boost started from. From the x=45 wall a plain RIGHT
  -- spin jump peaks at y=60 and drops straight onto the ship pad, which is
  -- where Ceres success (game state 32) fires. `ship_pad_action` is the tail
  -- for a landing that is already on the pad but short of the trigger x.
  if I(session.state.room_id) ~= ROOM_CERES_ELEVATOR then
    return
  end
  if M._ceres_elev_leaving(session.state) then
    return
  end

  if CROUCH_POSES[I(session.state.pose)] then
    for _ = 1, 10 do
      if not CROUCH_POSES[I(session.state.pose)] then
        break
      end
      session:step({"UP"}, "ceres_elev_uncrouch")
    end
  end

  if not M._ceres_elev_ship_band(session.state) then
    _ceres_elev_walk_to(session, _CERES_ELEV_171_LAUNCH_X)
    session:hold(2, {"RIGHT"}, "ceres_elev_ship_turn")
    for _ = 1, 120 do
      local state = session.state
      if I(state.room_id) ~= ROOM_CERES_ELEVATOR or M._ceres_elev_leaving(state) then
        return
      end
      if M._ceres_elev_ship_band(state) then
        break
      end
      session:step({"RIGHT", "A"}, "ceres_elev_ship_hop")
    end
  end

  for _ = 1, 80 do
    local st = session.state
    if I(st.room_id) ~= ROOM_CERES_ELEVATOR or M._ceres_elev_leaving(st) then
      return
    end
    local names = M.ship_pad_action(st)
    step(session, names, "ceres_elev_ship")
  end

  if I(session.state.room_id) == ROOM_CERES_ELEVATOR and not M._ceres_elev_leaving(session.state) then
    timeout("ceres elev ship leave failed: " .. state_str(session.state))
  end
end

function M.reactive_elev_climb(session)
  -- Elev after Falling → ship leave: one wall jump, then three ledge hops.
  --
  -- Fast entry is x216 y624–641 spin. Missed fast-entry, 475, 363, 267, or
  -- 171 plants raise. There is no checkpoint recover. Over 2500f raises.
  local start = I(session.frame)
  -- Local `start` is the live cap; the info key is not. RouteSession.step
  -- overwrites `session.info` with the env step info every frame, so the
  -- elev_to_landing guard in play_ceres_escape_to_landing reads None and
  -- never fires. Left as the planner's call (see docs/plan.md).
  session.info = session.info or {}
  session.info.ceres_elev_start = start
  session:wait_until(function(s)
    return I(s.room_id) == ROOM_CERES_ELEVATOR
  end, 300, "ceres_elev_door")
  _ceres_elev_budget(session, start)
  for _ = 1, 160 do
    local names = _ceres_elev_entry_action(session.state)
    if names == nil then
      break
    end
    step(session, names, "ceres_elev_entry")
    _ceres_elev_budget(session, start)
  end
  if not M._ceres_fast_entry_window(session.state) then
    timeout("ceres elev wall jump missed: " .. state_str(session.state))
  end
  local overlay = M._ceres_door_blocks_wj(session)
  if not _ceres_entry_to_475(session) then
    timeout("ceres 475 plant missed overlay=" .. tostring(overlay) .. ": " .. state_str(session.state))
  end
  _ceres_elev_budget(session, start)
  local hops = {
    {363, _CERES_ELEV_475_LAUNCH_X, "RIGHT"},
    {267, _CERES_ELEV_363_LAUNCH_X, "LEFT"},
    {_CERES_ELEV_TOP_Y, _CERES_ELEV_267_LAUNCH_X, "LEFT"},
  }
  for i = 1, #hops do
    local target_y, launch_x, side = hops[i][1], hops[i][2], hops[i][3]
    if not M._ceres_ledge_hop(session, target_y, launch_x, side) then
      timeout("ceres " .. tostring(target_y) .. " plant missed: " .. state_str(session.state))
    end
    _ceres_elev_budget(session, start)
  end
  M._ceres_elev_top_to_ship(session)
  _ceres_elev_budget(session, start)
end

M.play_ceres_magnet_to_falling = M.play_magnet_to_falling
M.play_ceres_falling_to_elev = M.play_falling_to_elev
M._ceres_reactive_elev_climb = M.reactive_elev_climb

return M
