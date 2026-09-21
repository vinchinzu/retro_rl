-- Climb (0x96BA) first descent toward Morph — moonfall policy.
--
-- Public policy: https://wiki.supermetroid.run/Climb and
-- https://wiki.supermetroid.run/Moonwalk (Climb is the biggest moonfall save,
-- ~7.40s regular fall → ~3.45s).
--
-- Enter from Parlor's bottom-left vertical door (top of Climb). Land on the
-- start ledge, face right, moonwalk left to the lip (~x=349), spinning
-- moonfall, hold LEFT down the shaft (skips the pirate floater at ~395,107),
-- aim-down to clip the bottom platform, then run RIGHT into the Pit door
-- (~x=493, y=2187).
--
-- Moonwalk is a file option ($09E4). This hop pokes it on at entry and
-- off after Pit settle so later hash-pinned seeds (pit / elev / morph)
-- keep their moonwalk-off inputs.
--
-- Assisted product still plays the Climb seed. Flip
-- CLIMB_MOONFALL_ON_CLEAN only after the pin dual is green and faster.
-- Lua 5.1.

local ram = require("ram")
local rooms = require("rooms")
local moonfall = require("skills.moonfall")
local knockback = require("skills.knockback")
local seeds = require("morph.seeds")

local M = {}

local ROOM_CLIMB = rooms.ROOM_CLIMB
local ROOM_PIT = rooms.ROOM_PIT
local FACING_RIGHT = ram.FACING_RIGHT
local GS_ORDINARY = ram.GS_ORDINARY

local is_airborne = moonfall.is_airborne
local is_moonfalling = moonfall.is_moonfalling
local is_moonwalking = moonfall.is_moonwalking
local is_knockback = knockback.is_knockback

-- Climb 3×9; start ledge y=91 x=348–372. Pirate floater ~(395,107).
-- Left-lip moonfall holds LEFT down the shaft. Pit door is RIGHT at
-- floor ~(493, 2187) — seed exit, not the map-left node name.
M.FALL_X = 300
M.LIP_X = 349
M.TOP_Y_MAX = 120
M.AIM_Y = 1600
M.BOTTOM_Y = 2100
M.DOOR_X = 490
M.PIT_SETTLE = 180
M.JUMP_HOLD = 3
M.SPIN_HOLD = 4
M.WIKI_URL = moonfall.WIKI_URL

-- Default off: seed remains the assisted/clean product hop until the probe
-- dual-greens from a natural Climb enter pin.
M.CLIMB_MOONFALL_ON_CLEAN = false

local function num(v, default)
  v = tonumber(v)
  if v == nil then
    return default or 0
  end
  return v
end

local function sx(st)
  return num(st.samus_x or st.x)
end

local function sy(st)
  return num(st.samus_y or st.y)
end

local function format_state(st)
  return seeds.format_state(st)
end

local function poke_moonwalk(session, enabled)
  ram.set_moonwalk(enabled)
  if session.refresh then
    session:refresh()
  end
  if session.state then
    session.state.moonwalk = enabled and 1 or 0
  end
end

function M.climb_moonfall_enabled(session)
  if session.climb_moonfall ~= nil then
    return not not session.climb_moonfall
  end
  if not M.CLIMB_MOONFALL_ON_CLEAN then
    return false
  end
  local assist = session.assist
  if assist == nil then
    return false
  end
  local enabled = true
  if assist.enabled ~= nil then
    enabled = not not assist.enabled
  end
  return not enabled
end

-- One-frame Climb moonfall policy (ROM-free).
--
-- Live probe (warp pin): left-lip moonfall, LEFT down the shaft, RIGHT
-- along the floor into Pit. First floater + pirate sit at ~(395,107);
-- jumping right from the start ledge lands on them.
function M.climb_moonfall_action(state, track)
  local x = sx(state)
  local y = sy(state)
  local room = num(state.room_id)
  local phase = track.phase
  local held = track.held
  local grounded = not is_airborne(state)

  if room == ROOM_PIT then
    track.phase = "done"
    track.held = 0
    return {}
  end
  if room ~= ROOM_CLIMB and phase ~= "exit" then
    track.phase = "exit"
    track.held = 0
    return {"RIGHT", "X"}
  end

  if is_knockback(state) and phase ~= "exit" and phase ~= "done" and phase ~= "bottom" then
    if held > 24 then
      if y > 200 then
        track.phase = "fall"
      else
        track.phase = "plant"
      end
      track.held = 0
      return {"LEFT"}
    end
    track.held = held + 1
    return {}
  end

  if phase == "plant" then
    if y > 400 then
      track.phase = "fall"
      track.held = 0
      return {"LEFT"}
    end
    if is_airborne(state) then
      -- No d-pad during drop-in (RIGHT walks onto the pirate floater).
      track.held = held + 1
      return {"X", "L"}
    end
    track.phase = "face"
    track.held = 0
    return {"RIGHT"}
  end

  if phase == "face" then
    if grounded and num(state.facing) == FACING_RIGHT then
      held = held + 1
      if held >= 2 then
        track.phase = "moonwalk"
        track.held = 0
        return {"LEFT", "X", "L"}
      end
      track.held = held
      return {}
    end
    track.held = 0
    return {"RIGHT"}
  end

  if phase == "moonwalk" then
    if is_airborne(state) then
      track.phase = "fall"
      track.held = 0
      return {"LEFT", "A"}
    end
    if is_moonwalking(state) and x <= M.LIP_X then
      track.phase = "jump"
      track.held = 0
      return {"LEFT", "X", "L", "A"}
    end
    if held > 40 then
      track.phase = "jump"
      track.held = 0
      return {"LEFT", "X", "L", "A"}
    end
    track.held = held + 1
    return {"LEFT", "X", "L"}
  end

  if phase == "jump" then
    held = held + 1
    if is_moonfalling(state) and held >= M.JUMP_HOLD then
      track.phase = "fall"
      track.held = 0
      return {"LEFT", "A"}
    end
    if held <= M.JUMP_HOLD then
      track.held = held
      return {"LEFT", "X", "L", "A"}
    end
    if held <= M.JUMP_HOLD + M.SPIN_HOLD then
      track.held = held
      return {"LEFT", "A"}
    end
    track.phase = "fall"
    track.held = 0
    return {"LEFT"}
  end

  if phase == "fall" then
    local floor = grounded and y >= M.BOTTOM_Y - 50
    if floor or y >= M.BOTTOM_Y then
      track.phase = "bottom"
      track.held = 0
      if y > 2190 then
        return {"A"}
      end
      return {"RIGHT", "X"}
    end
    if grounded and y < M.BOTTOM_Y - 50 then
      track.held = held + 1
      return {"LEFT", "X", "L", "A"}
    end
    if y >= M.AIM_Y then
      track.held = held + 1
      return {"LEFT", "L", "X"}
    end
    track.held = held + 1
    return {"LEFT"}
  end

  if phase == "bottom" then
    if y > 2192 and held < 8 then
      track.held = held + 1
      return {"A"}
    end
    if x >= M.DOOR_X then
      track.phase = "exit"
      track.held = 0
      return {"RIGHT", "X"}
    end
    track.held = held + 1
    return {"RIGHT", "X"}
  end

  if phase == "exit" then
    if room == ROOM_PIT then
      track.phase = "done"
      track.held = 0
      return {}
    end
    track.held = held + 1
    return {"RIGHT", "X"}
  end

  return {}
end

-- RAM-driven Climb → Pit using spinning moonfall. Pokes $09E4 on.
function M.play_climb_to_pit_moonfall(session, max_frames, restore_moonwalk)
  max_frames = max_frames or 1200
  if restore_moonwalk == nil then
    restore_moonwalk = true
  end
  poke_moonwalk(session, true)
  moonfall.require_moonwalk_on(session.state, "climb_moonfall")
  local st = session.state
  if num(st.room_id) ~= ROOM_CLIMB and num(st.game_state) ~= 8 and num(st.game_state) ~= 11 then
    error("climb moonfall: expected Climb 0x" .. string.format("%04X", ROOM_CLIMB)
      .. ", got " .. format_state(st))
  end

  local track = {phase = "plant", held = 0}
  local done = false
  local i
  for i = 1, max_frames do
    local names = M.climb_moonfall_action(session.state, track)
    session:step(names, "climb_moonfall_" .. track.phase)
    if track.phase == "done" or num(session.state.room_id) == ROOM_PIT then
      done = true
      break
    end
  end
  if not done then
    error(string.format(
      "climb moonfall missed Pit after %df: %s phase=%s (%s)",
      max_frames,
      format_state(session.state),
      tostring(track.phase),
      M.WIKI_URL
    ))
  end

  st = session.state
  if num(st.room_id) ~= ROOM_PIT or num(st.game_state) == 11 then
    session:wait_until(function(s)
      return num(s.room_id) == ROOM_PIT and num(s.game_state) == GS_ORDINARY
    end, M.PIT_SETTLE, "climb_moonfall_pit_settle")
  end
  if restore_moonwalk then
    poke_moonwalk(session, false)
  end
end

-- In-room initiate_moonfall then idle-steer (practice / dump).
function M.setup_then_fall(session)
  poke_moonwalk(session, true)
  moonfall.initiate_moonfall(session, {reason = "climb_setup"})
end

M.ROOM_CLIMB = ROOM_CLIMB
M.ROOM_PIT = ROOM_PIT
M.is_airborne = is_airborne
M.is_moonfalling = is_moonfalling
M.is_moonwalking = is_moonwalking

return M
