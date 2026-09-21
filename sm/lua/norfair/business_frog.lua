-- Business ↔ Frog Save / Speedway / Farm shortcuts.
-- Lua 5.1. Session: step / hold / wait_until / span.

local iceg = require("ice.geometry")
local geom = require("skills.geometry")

local ROOM_BUBBLE = 0xACB3
local ROOM_BUSINESS = 0xA7DE
local ROOM_FROG_SAVE = 0xB167
local ROOM_FROG_SPEEDWAY = 0xB106
local ROOM_FARM = 0xAF72
local STANDING = geom.STANDING_POSES or iceg.STANDING_POSES
local MAX_SCAFFOLD = 240
local ELEVATOR_Y = 680
local FLOOR_Y_MIN = 1405

local M = {}

local function scaffold_exit(session, entry_room, target_room, label, face)
  face = face or "RIGHT"
  iceg.require_room(session, entry_room, label)
  for _ = 1, MAX_SCAFFOLD do
    local state = session:hold(1, {face, "B"}, label .. "_scaffold")
    if state.room_id == target_room then
      return state
    end
  end
  local state = session.state
  error(string.format(
    "%s: scaffold timeout before room 0x%04X; room=0x%04X pose=%d xy=(%d,%d)",
    label, target_room, state.room_id, state.pose, state.samus_x, state.samus_y
  ))
end

function M.play_business_to_frog_save(session)
  local label = "business_to_frog_save"
  iceg.require_room(session, ROOM_BUSINESS, label)

  local stable = 0
  local settled = false
  for _ = 1, 600 do
    local state = session:hold(1, {}, label .. "_elevator_settle")
    if state.samus_y == ELEVATOR_Y then
      stable = stable + 1
      if stable >= 24 then
        settled = true
        break
      end
    else
      stable = 0
    end
  end
  if not settled then
    error(string.format("%s: elevator did not settle", label))
  end

  local floored = false
  for frame = 0, 649 do
    local state = session.state
    if state.samus_y >= FLOOR_Y_MIN and state.velocity_y == 0 and STANDING[state.pose] then
      floored = true
      break
    end
    local buttons
    if math.floor(frame / 70) % 2 == 0 then
      buttons = {"LEFT", "B"}
    else
      buttons = {"RIGHT", "B"}
    end
    session:hold(1, buttons, label .. "_descend")
  end
  if not floored then
    error(string.format("%s: floor band missed", label))
  end

  iceg.select_weapon(session, 0)
  local hit = false
  for _ = 1, 400 do
    local state = session:hold(1, {"RIGHT", "B", "X"}, label .. "_door")
    if state.room_id == ROOM_FROG_SAVE then
      hit = true
      break
    end
  end
  if not hit then
    error(string.format("%s: Frog door missed", label))
  end

  return iceg.wait_ordinary_room(session, ROOM_FROG_SAVE, 320, label)
end

function M.play_frog_save_to_speedway(session)
  local label = "frog_save_to_speedway"
  iceg.require_room(session, ROOM_FROG_SAVE, label)
  iceg.select_weapon(session, 0)
  local hit = false
  for frame = 0, 399 do
    local inputs = {"RIGHT", "B", "X"}
    if (frame >= 30 and frame < 40) or (frame >= 90 and frame < 100) then
      inputs = {"RIGHT", "B", "X", "A"}
    end
    local state = session:hold(1, inputs, label .. "_door")
    if state.room_id == ROOM_FROG_SPEEDWAY then
      hit = true
      break
    end
  end
  if not hit then
    local state = session.state
    error(string.format(
      "%s: right door missed; room=0x%04X pose=%d xy=(%d,%d)",
      label, state.room_id, state.pose, state.samus_x, state.samus_y
    ))
  end
  return iceg.wait_ordinary_room(session, ROOM_FROG_SPEEDWAY, 320, label)
end

function M.play_speedway_to_farm(session)
  local label = "speedway_to_farm"
  iceg.require_room(session, ROOM_FROG_SPEEDWAY, label)
  iceg.select_weapon(session, 0)
  local max_x = session.state.samus_x
  local hit = false
  for _ = 1, 1100 do
    local state = session:hold(1, {"RIGHT", "B", "X"}, label .. "_door")
    if state.samus_x > max_x then
      max_x = state.samus_x
    end
    if state.room_id == ROOM_FARM then
      hit = true
      break
    end
  end
  if not hit then
    local state = session.state
    local stall = ""
    if max_x <= 820 and state.samus_x <= 820 then
      stall = " (boost-block stall; no Speed)"
    end
    error(string.format(
      "%s: right door missed; room=0x%04X pose=%d xy=(%d,%d) max_x=%d%s",
      label, state.room_id, state.pose, state.samus_x, state.samus_y, max_x, stall
    ))
  end
  return iceg.wait_ordinary_room(session, ROOM_FARM, 320, label)
end

function M.play_farm_to_bubble(session)
  return scaffold_exit(session, ROOM_FARM, ROOM_BUBBLE, "farm_to_bubble", "RIGHT")
end

return M
