-- Speed Room → Hall → Bat Cave drop → Bubble (K4.7).
-- Lua 5.1. Session: step / hold / wait_until / span.

local iceg = require("ice.geometry")
local knockback = require("skills.knockback")
local geom = require("skills.geometry")

local ROOM_SPEED, ROOM_HALL, ROOM_BAT, ROOM_BUBBLE = 0xAD1B, 0xACF0, 0xB07A, 0xACB3
local ITEM_SPEED = 0x2000
local STANDING = geom.STANDING_POSES or iceg.STANDING_POSES
local band = iceg.band
local HOLE_X = {148, 154}
local SHELF_BOMB_X = {160, 180}
local CAVITY_Y = {220, 300}
local BAT_FLOOR_Y, LAVA_Y, BAT_DOOR_X = 360, 430, 45

local M = {}
M.ROOM_BAT_CAVE, M.ROOM_BUBBLE, M.ROOM_SPEED, M.ROOM_SPEED_HALL = ROOM_BAT, ROOM_BUBBLE, ROOM_SPEED, ROOM_HALL
M.ITEM_SPEED = ITEM_SPEED

local function ensure_morph(session, label)
  for _ = 1, 4 do
    local pose = session.state.pose
    if pose == 27 or pose == 28 or pose == 49 or pose == 50 or pose == 55
        or pose == 56 or pose == 65 or pose == 129 or pose == 130
        or pose == 131 or pose == 132 then
      return
    end
    session:hold(5, {"DOWN"}, label .. "_morph1")
    session:hold(4, {}, label .. "_morph_gap")
    session:hold(5, {"DOWN"}, label .. "_morph2")
    session:hold(8, {}, label .. "_morph_settle")
  end
end

local function leave_speed_room(session, label)
  iceg.require_room(session, ROOM_SPEED, label)
  if band(session.state.collected_items or 0, ITEM_SPEED) == 0 then
    error(string.format("%s: Speed not collected (items=0x%04X)", label, session.state.collected_items or 0))
  end
  iceg.unmorph(session)
  iceg.select_weapon(session, 0)
  session:hold(8, {}, label .. "_stand")
  local hit = false
  for frame = 0, 279 do
    local state = session.state
    if state.room_id == ROOM_HALL or state.room_id ~= ROOM_SPEED then
      hit = true
      break
    end
    if knockback.is_knockback(state) then
      knockback.escape_knockback_spin(session, {
        prefer_dir = "LEFT", run_frames = 6, spin_frames = 20, label = label,
        stop_room_id = ROOM_HALL,
      })
    elseif state.pose == 137 or state.pose == 138 then
      iceg.unmorph(session)
    elseif state.samus_x <= 40 and state.velocity_y == 0 then
      local phase = frame % 16
      if phase < 4 then
        session:hold(1, {"LEFT", "X"}, label .. "_door_shot")
      elseif phase < 10 then
        session:hold(1, {"LEFT", "B"}, label .. "_door_push")
      else
        session:hold(1, {"LEFT", "B", "A"}, label .. "_door_spin")
      end
    else
      local phase = frame % 18
      if phase < 10 then
        session:hold(1, {"LEFT", "B"}, label .. "_exit_run")
      elseif phase < 14 then
        session:hold(1, {"LEFT"}, label .. "_exit_walk")
      else
        session:hold(1, {"LEFT", "X"}, label .. "_exit_shot")
      end
    end
  end
  if session.state.room_id ~= ROOM_HALL then
    error(string.format("%s: left Speed door missed", label))
  end
  return iceg.wait_ordinary_room(session, ROOM_HALL, 280, label)
end

local function dash_speed_hall_return(session, label)
  iceg.require_room(session, ROOM_HALL, label)
  for _ = 1, 30 do
    local state = session:hold(1, {}, label .. "_hall_land")
    if state.velocity_y == 0 and STANDING[state.pose] then
      break
    end
  end
  iceg.select_weapon(session, 0)
  local min_x = session.state.samus_x
  local hit = false
  for frame = 0, 899 do
    local state = session:hold(1, {"LEFT", "B"}, label .. "_hall_dash")
    if state.samus_x < min_x then min_x = state.samus_x end
    if state.room_id == ROOM_BAT or state.room_id ~= ROOM_HALL then
      hit = true
      break
    end
    if state.samus_x <= 80 then
      if frame % 12 < 3 then
        session:hold(1, {"LEFT", "X"}, label .. "_hall_door_shot")
      else
        session:hold(1, {"LEFT", "B"}, label .. "_hall_door_push")
      end
      if session.state.room_id == ROOM_BAT then
        hit = true
        break
      end
    end
  end
  if session.state.room_id ~= ROOM_BAT then
    for frame = 0, 199 do
      local state = session:hold(1, {"LEFT", "B"}, label .. "_hall_retry")
      if state.room_id == ROOM_BAT then
        hit = true
        break
      end
      if frame % 20 == 0 then
        session:hold(2, {"LEFT", "X"}, label .. "_hall_retry_shot")
      end
    end
  end
  if session.state.room_id ~= ROOM_BAT then
    error(string.format("%s: Bat Cave not reached from Hall", label))
  end
  return iceg.wait_ordinary_room(session, ROOM_BAT, 280, label)
end

local function shelf_bomb_to_cavity(session, label)
  for _ = 1, 120 do
    local state = session.state
    if state.room_id ~= ROOM_BAT or state.samus_y >= CAVITY_Y[1] then
      return
    end
    if knockback.is_knockback(state) then
      knockback.escape_knockback_spin(session, {
        prefer_dir = "LEFT", run_frames = 4, spin_frames = 14, label = label,
        stop_room_id = ROOM_BUBBLE,
      })
    elseif SHELF_BOMB_X[1] <= state.samus_x and state.samus_x <= SHELF_BOMB_X[2]
        and state.velocity_y == 0 then
      break
    else
      local face = (state.samus_x > 168) and "LEFT" or "RIGHT"
      session:hold(1, {face}, label .. "_to_bomb")
    end
  end
  ensure_morph(session, label)
  for i = 0, 319 do
    local state = session.state
    if state.room_id ~= ROOM_BAT or state.samus_y >= CAVITY_Y[1] then
      return
    end
    if knockback.is_knockback(state) then
      iceg.unmorph(session)
      knockback.escape_knockback_spin(session, {
        prefer_dir = "LEFT", run_frames = 4, spin_frames = 12, label = label,
        stop_room_id = ROOM_BUBBLE,
      })
      ensure_morph(session, label)
    elseif i % 42 < 2 then
      session:hold(1, {"X"}, label .. "_shelf_bomb")
    elseif state.samus_x < SHELF_BOMB_X[1] then
      session:hold(1, {"RIGHT"}, label .. "_bomb_r")
    elseif state.samus_x > SHELF_BOMB_X[2] then
      session:hold(1, {"LEFT"}, label .. "_bomb_l")
    else
      session:hold(1, {}, label .. "_bomb_wait")
    end
  end
end

local function cavity_hole_to_floor(session, label)
  iceg.unmorph(session)
  iceg.select_weapon(session, 0)
  for _ = 1, 25 do
    session:hold(1, {}, label .. "_cavity_stand")
    if session.state.velocity_y == 0 and STANDING[session.state.pose] then
      break
    end
  end
  for i = 0, 399 do
    local state = session.state
    if state.room_id ~= ROOM_BAT or state.samus_y >= BAT_FLOOR_Y then
      return
    end
    if knockback.is_knockback(state) then
      knockback.escape_knockback_spin(session, {
        prefer_dir = "LEFT", run_frames = 4, spin_frames = 12, label = label,
        stop_room_id = ROOM_BUBBLE,
      })
    elseif state.velocity_y == 0 and state.samus_x < HOLE_X[1] then
      session:hold(1, {"RIGHT"}, label .. "_hole_r")
    elseif state.velocity_y == 0 and state.samus_x > HOLE_X[2] then
      session:hold(1, {"LEFT"}, label .. "_hole_l")
    else
      local phase = i % 20
      if phase < 10 then
        session:hold(1, {"DOWN", "X"}, label .. "_hole_shot")
      elseif phase < 14 then
        session:hold(1, {"A"}, label .. "_hole_hop")
      else
        session:hold(1, {"DOWN", "X"}, label .. "_hole_fall")
      end
    end
  end
end

local function floor_to_bubble_door(session, label)
  iceg.unmorph(session)
  iceg.select_weapon(session, 0)
  for _ = 1, 40 do
    session:hold(1, {}, label .. "_floor_land")
    if session.state.velocity_y == 0 and session.state.samus_y < LAVA_Y
        and STANDING[session.state.pose] then
      break
    end
  end
  for frame = 0, 899 do
    local state = session.state
    if state.room_id == ROOM_BUBBLE or state.room_id ~= ROOM_BAT then
      return
    end
    if knockback.is_knockback(state) then
      knockback.escape_knockback_spin(session, {
        prefer_dir = "LEFT", run_frames = 4, spin_frames = 14, label = label,
        stop_room_id = ROOM_BUBBLE,
      })
    elseif state.samus_y >= LAVA_Y then
      session:hold(1, {"LEFT", "B", "A"}, label .. "_lava")
    else
      local grounded = state.velocity_y == 0 and STANDING[state.pose]
      if state.samus_x <= BAT_DOOR_X + 15 then
        if state.samus_y < 380 and grounded then
          session:hold(1, {"LEFT"}, label .. "_to_sill")
          if frame % 20 == 10 then
            session:hold(2, {"DOWN"}, label .. "_sill_drop")
          end
        elseif state.samus_x <= BAT_DOOR_X then
          local phase = frame % 16
          if phase < 5 then
            session:hold(1, {"LEFT", "X"}, label .. "_bot_shot")
          elseif phase < 11 then
            session:hold(1, {"LEFT", "B"}, label .. "_bot_push")
          else
            session:hold(1, {"LEFT", "B", "A"}, label .. "_bot_spin")
          end
        else
          local phase = frame % 25
          if phase < 6 then
            session:hold(1, {"LEFT", "B", "A"}, label .. "_floor_hop")
          elseif phase < 14 then
            session:hold(1, {"LEFT", "B"}, label .. "_floor_run")
          else
            session:hold(1, {"LEFT"}, label .. "_floor_walk")
          end
        end
      else
        local phase = frame % 25
        if phase < 6 then
          session:hold(1, {"LEFT", "B", "A"}, label .. "_floor_hop")
        elseif phase < 14 then
          session:hold(1, {"LEFT", "B"}, label .. "_floor_run")
        else
          session:hold(1, {"LEFT"}, label .. "_floor_walk")
        end
      end
    end
  end
end

local function descend_bat_to_bubble(session, label)
  iceg.require_room(session, ROOM_BAT, label)
  iceg.unmorph(session)
  if session.state.samus_y < CAVITY_Y[1] then
    shelf_bomb_to_cavity(session, label)
  end
  if session.state.room_id == ROOM_BAT and session.state.samus_y < BAT_FLOOR_Y then
    if session.state.samus_y < CAVITY_Y[1] then
      shelf_bomb_to_cavity(session, label)
    end
    if session.state.samus_y < BAT_FLOOR_Y then
      cavity_hole_to_floor(session, label)
    end
  end
  if session.state.room_id == ROOM_BAT then
    floor_to_bubble_door(session, label)
  end
  if session.state.room_id ~= ROOM_BUBBLE then
    error(string.format("%s: Bat Cave bottom door missed; room=0x%04X xy=(%d,%d)",
      label, session.state.room_id, session.state.samus_x, session.state.samus_y))
  end
  return iceg.wait_ordinary_room(session, ROOM_BUBBLE, 320, label)
end

function M.play_speed_return_to_bubble(session)
  local label = "speed_return_to_bubble"
  iceg.require_room(session, ROOM_SPEED, label)
  local start = session.frame
  leave_speed_room(session, label)
  dash_speed_hall_return(session, label)
  local state = descend_bat_to_bubble(session, label)
  if state.room_id ~= ROOM_BUBBLE then
    error(string.format("%s: finished outside Bubble; room=0x%04X frames=%d",
      label, state.room_id, session.frame - start))
  end
  return state
end

return M
