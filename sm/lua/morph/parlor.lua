-- Parlor (0x92FD) first descent toward Climb — moonfall policy.
--
-- Public policy: https://wiki.supermetroid.run/Parlor_and_Alcatraz and
-- https://wiki.supermetroid.run/Moonwalk (Parlor fall ~8.10s regular →
-- ~7.50s moonfall; listed save 0.20s). Then a double set of downbacks into
-- the floor Climb door.
--
-- Enter from Landing Site (Parlor node 4, top-right). Dash left across the
-- top corridor (jump at x≈1127 clears the first Geemer), spinning moonfall
-- at the left-shaft lip (x≤360), LEFT off the grass platforms, LEFT+X+L
-- into the floor Climb door (node 7) ~(393, 1248).
--
-- Moonwalk is a file option ($09E4). This hop pokes it on at entry and
-- off after Climb settle so the Climb seed keeps moonwalk-off inputs.
--
-- Assisted product still plays the Parlor seed. Flip
-- PARLOR_MOONFALL_ON_CLEAN only after the pin dual is green and faster,
-- and Climb still clears from the new seat.
-- Lua 5.1.

local ram = require("ram")
local rooms = require("rooms")
local moonfall = require("skills.moonfall")
local knockback = require("skills.knockback")
local seeds = require("morph.seeds")

local M = {}

local ROOM_PARLOR = rooms.ROOM_PARLOR
local ROOM_CLIMB = rooms.ROOM_CLIMB
local FACING_RIGHT = ram.FACING_RIGHT
local GS_ORDINARY = ram.GS_ORDINARY

local is_airborne = moonfall.is_airborne
local is_moonfalling = moonfall.is_moonfalling
local is_moonwalking = moonfall.is_moonwalking
local is_knockback = knockback.is_knockback

-- Parlor 5×5. Top corridor y<220. Left shaft map col 1 (x 256–512).
-- Seed: dash LEFT+B from the landing door, jump at x≈1127 to clear the
-- first Geemer ledge, then moonfall from the left-shaft lip ~x=360.
M.LEDGE_X = 390
M.LIP_X = 365
M.SHAFT_LIP_X = 360
M.GEEMER_JUMP_X = 1130
M.SHAFT_X = 425
M.TOP_Y_MAX = 220
M.AIM_Y = 1300
M.BOTTOM_Y = 1180
M.DOOR_X = 393
M.DOOR_X_LO = 360
M.DOOR_X_HI = 430
M.CLIMB_SETTLE = 180
M.JUMP_HOLD = 3
M.SPIN_HOLD = 4
M.WIKI_URL = moonfall.WIKI_URL

-- Default off: seed remains the assisted/clean product hop until the probe
-- dual-greens from a Parlor enter pin and Climb still leaves from the seat.
M.PARLOR_MOONFALL_ON_CLEAN = false

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

local function steer_x(x, target)
  if x < target - 12 then
    return {"RIGHT"}
  end
  if x > target + 12 then
    return {"LEFT"}
  end
  return {}
end

local function append(a, b)
  local out = {}
  local i
  for i = 1, #a do
    out[#out + 1] = a[i]
  end
  for i = 1, #b do
    out[#out + 1] = b[i]
  end
  return out
end

function M.parlor_moonfall_enabled(session)
  if session.parlor_moonfall ~= nil then
    return not not session.parlor_moonfall
  end
  if not M.PARLOR_MOONFALL_ON_CLEAN then
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

-- One-frame Parlor moonfall policy (ROM-free).
--
-- Live probe: run left on the top, left-lip moonfall down the Alcatraz
-- shaft, downback into the floor Climb door. First-descent planet is
-- typically not awake (no Ripper/Geemer).
function M.parlor_moonfall_action(state, track)
  local x = sx(state)
  local y = sy(state)
  local room = num(state.room_id)
  local phase = track.phase
  local held = track.held
  local grounded = not is_airborne(state)

  if room == ROOM_CLIMB then
    track.phase = "done"
    track.held = 0
    return {}
  end
  if room ~= ROOM_PARLOR and phase ~= "exit" then
    track.phase = "exit"
    track.held = 0
    return {"LEFT"}
  end

  if is_knockback(state) and phase ~= "done" and phase ~= "exit" then
    local nxt = "run"
    if y > M.TOP_Y_MAX then
      nxt = "fall"
    end
    local names
    if held % 2 == 0 then
      names = {"LEFT", "B", "A"}
    else
      names = {"LEFT", "B"}
    end
    track.phase = nxt
    track.held = held + 1
    return names
  end

  if phase == "plant" then
    if num(state.game_state) == 11 then
      -- Match seed: dash LEFT+B through the landing door.
      track.held = held + 1
      return {"LEFT", "B"}
    end
    if y > 400 then
      track.phase = "fall"
      track.held = 0
      return {"LEFT"}
    end
    if grounded then
      track.phase = "run"
      track.held = 0
      return {"LEFT", "B"}
    end
    track.held = held + 1
    return {"LEFT", "B"}
  end

  if phase == "run" then
    if y > M.TOP_Y_MAX + 50 then
      track.phase = "fall"
      track.held = 0
      return {"LEFT", "A"}
    end
    if x <= M.LEDGE_X and y < M.TOP_Y_MAX + 40 then
      if grounded then
        track.phase = "face"
        track.held = 0
        return {"RIGHT"}
      end
      track.held = held + 1
      return {}
    end
    -- Seed jump at x≈1127 y≈163 (pose 26) clears the Geemer ledge.
    local jumping = is_airborne(state) and x > 900
    local want_jump = (grounded and 1000 < x and x <= M.GEEMER_JUMP_X) or jumping
    track.held = held + 1
    if want_jump then
      return {"LEFT", "B", "A"}
    end
    return {"LEFT", "B"}
  end

  if phase == "face" then
    if y > M.TOP_Y_MAX + 50 then
      track.phase = "fall"
      track.held = 0
      return {"LEFT"}
    end
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
    local lip = M.LIP_X
    if y >= 160 then
      lip = M.SHAFT_LIP_X
    end
    if is_moonwalking(state) and x <= lip then
      track.phase = "jump"
      track.held = 0
      return {"LEFT", "X", "L", "A"}
    end
    if held > 90 and x <= lip + 12 then
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
      return {"RIGHT", "A"}
    end
    if held <= M.JUMP_HOLD then
      track.held = held
      return {"LEFT", "X", "L", "A"}
    end
    if held <= M.JUMP_HOLD + M.SPIN_HOLD then
      track.held = held
      return {"RIGHT", "A"}
    end
    track.phase = "fall"
    track.held = 0
    return {"RIGHT"}
  end

  if phase == "fall" then
    if grounded and y >= M.BOTTOM_Y - 80 then
      track.phase = "downback"
      track.held = 0
      return {"LEFT", "X", "L"}
    end
    if y >= M.BOTTOM_Y then
      track.phase = "downback"
      track.held = 0
      return {"LEFT", "X", "L"}
    end
    if grounded and y < M.BOTTOM_Y - 80 then
      -- Seed: downback RIGHT off the y≈173 ledge, LEFT off later
      -- grass platforms. Do not re-moonfall on every seat.
      track.held = held + 1
      if 160 <= y and y <= 190 and x < 420 then
        return {"RIGHT", "DOWN", "B"}
      end
      return {"LEFT"}
    end
    if y >= M.AIM_Y then
      track.held = held + 1
      return append(steer_x(x, M.DOOR_X), {"L", "X"})
    end
    local steer = steer_x(x, M.SHAFT_X)
    track.held = held + 1
    if #steer == 0 then
      return {"RIGHT"}
    end
    return steer
  end

  if phase == "downback" then
    if room == ROOM_CLIMB then
      track.phase = "done"
      track.held = 0
      return {}
    end
    -- Live pin: LEFT+X+L from the falling y≈1183 seat clips the floor door.
    if y >= 1170 then
      track.phase = "exit"
      track.held = 0
      return {"LEFT", "X", "L"}
    end
    track.held = held + 1
    if x < 420 then
      return {"RIGHT"}
    end
    if x > 435 then
      return {"LEFT"}
    end
    return {"LEFT"}
  end

  if phase == "exit" then
    if room == ROOM_CLIMB then
      track.phase = "done"
      track.held = 0
      return {}
    end
    track.held = held + 1
    return {"LEFT", "X", "L"}
  end

  return {}
end

-- RAM-driven Parlor → Climb using spinning moonfall. Pokes $09E4 on.
function M.play_parlor_to_climb_moonfall(session, max_frames, restore_moonwalk)
  max_frames = max_frames or 1800
  if restore_moonwalk == nil then
    restore_moonwalk = true
  end
  poke_moonwalk(session, true)
  moonfall.require_moonwalk_on(session.state, "parlor_moonfall")
  local st = session.state
  if num(st.room_id) ~= ROOM_PARLOR and num(st.game_state) ~= 8 and num(st.game_state) ~= 11 then
    error("parlor moonfall: expected Parlor 0x" .. string.format("%04X", ROOM_PARLOR)
      .. ", got " .. format_state(st))
  end

  local track = {phase = "plant", held = 0}
  local done = false
  local i
  for i = 1, max_frames do
    local names = M.parlor_moonfall_action(session.state, track)
    session:step(names, "parlor_moonfall_" .. track.phase)
    if track.phase == "done" or num(session.state.room_id) == ROOM_CLIMB then
      done = true
      break
    end
  end
  if not done then
    error(string.format(
      "parlor moonfall missed Climb after %df: %s phase=%s (%s)",
      max_frames,
      format_state(session.state),
      tostring(track.phase),
      M.WIKI_URL
    ))
  end

  st = session.state
  if num(st.room_id) ~= ROOM_CLIMB or num(st.game_state) == 11 then
    session:wait_until(function(s)
      return num(s.room_id) == ROOM_CLIMB and num(s.game_state) == GS_ORDINARY
    end, M.CLIMB_SETTLE, "parlor_moonfall_climb_settle")
  end
  if restore_moonwalk then
    poke_moonwalk(session, false)
  end
end

-- In-room initiate_moonfall then idle-steer (practice / dump).
function M.setup_then_fall(session)
  poke_moonwalk(session, true)
  moonfall.initiate_moonfall(session, {reason = "parlor_setup"})
end

M.ROOM_PARLOR = ROOM_PARLOR
M.ROOM_CLIMB = ROOM_CLIMB
M.is_airborne = is_airborne
M.is_moonfalling = is_moonfalling
M.is_moonwalking = is_moonwalking

return M
