-- Bat Cave → Speed Hall → Speed Booster collect (K4.5).
-- Lua 5.1. Session: step / hold / wait_until / span.

local iceg = require("ice.geometry")
local knockback = require("skills.knockback")
local geom = require("skills.geometry")

local ROOM_BAT = 0xB07A
local ROOM_HALL = 0xACF0
local ROOM_SPEED = 0xAD1B
local ITEM_SPEED = 0x2000
local STANDING = geom.STANDING_POSES or iceg.STANDING_POSES
local band = iceg.band

local HOLE_X = {148, 154}
local CAVITY_Y = {230, 290}
local DOOR_SHELF_Y = 160
local LAVA_Y = 430
local FLOOR_Y = 395

local M = {}
M.ROOM_BAT_CAVE, M.ROOM_SPEED, M.ROOM_SPEED_HALL = ROOM_BAT, ROOM_SPEED, ROOM_HALL
M.ITEM_SPEED = ITEM_SPEED

local function on_door_shelf(state)
  return state.room_id == ROOM_BAT and state.velocity_y == 0
    and state.samus_y <= DOOR_SHELF_Y
    and (STANDING[state.pose] or state.pose == 9 or state.pose == 10 or state.pose == 11)
end

local function in_cavity(state)
  return state.room_id == ROOM_BAT
    and CAVITY_Y[1] <= state.samus_y and state.samus_y <= CAVITY_Y[2]
    and state.samus_x >= 140
end

local function land_door_ledge(session, label)
  for _ = 1, 60 do
    local state = session:hold(1, {}, label .. "_land")
    if state.velocity_y == 0 and STANDING[state.pose] and state.samus_y < LAVA_Y then
      return state
    end
  end
  return session.state
end

local function shoot_ceiling_hole(session, label)
  iceg.select_weapon(session, 0)
  session:hold(2, {"RIGHT"}, label .. "_face")
  session:hold(3, {}, label .. "_face_settle")
  for i = 0, 13 do
    local state = session.state
    if state.samus_x > 66 then
      session:hold(1, {"LEFT"}, label .. "_ledge_back")
    else
      if i % 2 == 0 then
        session:hold(1, {"R", "X"}, label .. "_angle_shot")
      else
        session:hold(1, {"UP", "X"}, label .. "_up_shot")
      end
      session:hold(3, {}, label .. "_shot_travel")
    end
  end
end

local function gap_skip_to_mid(session, label)
  for _ = 1, 6 do
    session:hold(1, {"RIGHT", "B"}, label .. "_gap_runup")
  end
  for frame = 0, 54 do
    local state = session:hold(1, {"RIGHT", "B", "A"}, label .. "_gap_jump")
    if state.room_id ~= ROOM_BAT then
      return state
    end
    if frame > 20 and state.velocity_y == 0 and state.samus_y < LAVA_Y and state.samus_x > 100 then
      break
    end
  end
  for _ = 1, 40 do
    local state = session:hold(1, {}, label .. "_gap_land")
    if state.velocity_y == 0 and state.samus_y < LAVA_Y and STANDING[state.pose] then
      break
    end
  end
  return session.state
end

local function lava_recover(session, label)
  for _ = 1, 40 do
    local state = session:hold(1, {"LEFT", "B", "A"}, label .. "_lava")
    if state.samus_y < FLOOR_Y + 10 and state.velocity_y == 0 then
      break
    end
  end
  for _ = 1, 20 do
    session:hold(1, {}, label .. "_lava_settle")
  end
end

local function walk_to_hole_band(session, label)
  for _ = 1, 70 do
    local state = session.state
    if state.samus_y >= LAVA_Y then
      return false
    end
    if HOLE_X[1] <= state.samus_x and state.samus_x <= HOLE_X[2]
        and state.velocity_y == 0 and state.samus_y < LAVA_Y then
      return true
    end
    if state.velocity_y ~= 0 then
      session:hold(1, {}, label .. "_hole_air")
    else
      local face = (state.samus_x < 151) and "RIGHT" or "LEFT"
      session:hold(1, {face}, label .. "_to_hole")
    end
  end
  local st = session.state
  return HOLE_X[1] <= st.samus_x and st.samus_x <= HOLE_X[2] and st.samus_y < LAVA_Y
end

local function clear_shot_block(session, label)
  iceg.select_weapon(session, 0)
  for _ = 1, 180 do
    local state = session.state
    if state.samus_y >= LAVA_Y or state.room_id ~= ROOM_BAT then
      return
    end
    if knockback.is_knockback(state) then
      knockback.escape_knockback_spin(session, {
        prefer_dir = "LEFT", run_frames = 3, spin_frames = 12, label = label,
      })
    elseif state.velocity_y == 0 and state.samus_x < HOLE_X[1] then
      session:hold(1, {"RIGHT"}, label .. "_hole_recenter")
    elseif state.velocity_y == 0 and state.samus_x > HOLE_X[2] then
      session:hold(1, {"LEFT"}, label .. "_hole_recenter")
    else
      session:hold(1, {"UP", "X"}, label .. "_hole_spam")
    end
  end
end

local function settle_cavity(session, label)
  for _ = 1, 60 do
    local state = session.state
    if state.room_id ~= ROOM_BAT then
      return state
    end
    if knockback.is_knockback(state) then
      knockback.escape_knockback_spin(session, {
        prefer_dir = "LEFT", run_frames = 4, spin_frames = 14, label = label,
      })
    elseif state.velocity_y == 0 and (STANDING[state.pose] or state.pose == 9
        or state.pose == 10 or state.pose == 11 or state.pose == 230) then
      session:hold(2, {}, label .. "_cavity_settle")
      return session.state
    else
      session:hold(1, {}, label .. "_cavity_fall")
    end
  end
  return session.state
end

local function cavity_to_door_shelf(session, label)
  local min_y = session.state.samus_y
  iceg.select_weapon(session, 0)
  for _ = 1, 90 do
    local state = session.state
    if state.room_id ~= ROOM_BAT or state.samus_y >= LAVA_Y then
      return state, min_y
    end
    if knockback.is_knockback(state) then
      knockback.escape_knockback_spin(session, {
        prefer_dir = "LEFT", run_frames = 3, spin_frames = 12, label = label,
      })
    elseif state.velocity_y == 0 and state.samus_x < 150 then
      session:hold(1, {"RIGHT"}, label .. "_cavity_center")
    elseif state.velocity_y == 0 and state.samus_x > 180 then
      session:hold(1, {"LEFT"}, label .. "_cavity_center")
    else
      session:hold(1, {"UP", "X"}, label .. "_cavity_ceil")
    end
  end
  for frame = 0, 99 do
    local state = session.state
    if state.samus_y < min_y then min_y = state.samus_y end
    if state.room_id == ROOM_HALL or state.room_id ~= ROOM_BAT then
      return state, min_y
    end
    if knockback.is_knockback(state) then
      knockback.escape_knockback_spin(session, {
        prefer_dir = "RIGHT", run_frames = 4, spin_frames = 14, label = label,
        stop_room_id = ROOM_HALL,
      })
    elseif on_door_shelf(state) and frame > 25 then
      return state, min_y
    elseif frame < 20 then
      session:hold(1, {"A"}, label .. "_shelf_up")
    else
      session:hold(1, {"RIGHT", "B", "A", "X"}, label .. "_shelf_R")
    end
  end
  return session.state, min_y
end

local function jump_through_hole(session, label)
  local min_y = session.state.samus_y
  for frame = 0, 89 do
    local state = session.state
    if state.samus_y < min_y then min_y = state.samus_y end
    if state.room_id ~= ROOM_BAT then
      return state, min_y
    end
    if knockback.is_knockback(state) then
      knockback.escape_knockback_spin(session, {
        prefer_dir = "RIGHT", run_frames = 4, spin_frames = 16, label = label,
        stop_room_id = ROOM_HALL,
      })
    elseif on_door_shelf(state) then
      return state, min_y
    elseif frame > 40 and state.velocity_y == 0 and in_cavity(state) then
      return state, min_y
    elseif frame < 16 then
      session:hold(1, {"A"}, label .. "_hole_up")
    else
      session:hold(1, {"RIGHT", "B", "A", "X"}, label .. "_hole_R")
    end
  end
  return session.state, min_y
end

local function climb_toward_upper(session, label)
  local min_y = session.state.samus_y
  for attempt = 0, 5 do
    local state = session.state
    if state.room_id ~= ROOM_BAT then
      return state, min_y, attempt
    end
    if on_door_shelf(state) or state.room_id == ROOM_HALL then
      return state, min_y, attempt
    end
    if state.samus_y >= LAVA_Y then
      lava_recover(session, label)
      if session.state.samus_x < 80 and session.state.samus_y < LAVA_Y then
        gap_skip_to_mid(session, label)
      end
    elseif in_cavity(state) or (state.samus_y < 320 and state.samus_y > DOOR_SHELF_Y
        and state.samus_x >= 140) then
      settle_cavity(session, label)
      local jump_min
      state, jump_min = cavity_to_door_shelf(session, label)
      if jump_min < min_y then min_y = jump_min end
      if state.room_id ~= ROOM_BAT or on_door_shelf(state) then
        return state, min_y, attempt
      end
    elseif state.samus_y >= 340 then
      if not walk_to_hole_band(session, label) then
        if session.state.samus_y >= LAVA_Y then
          -- retry
        elseif session.state.samus_x < 80 then
          gap_skip_to_mid(session, label)
        end
      else
        for _ = 1, 12 do
          local st = session:hold(1, {}, label .. "_hole_settle")
          if st.velocity_y == 0 and STANDING[st.pose] then
            break
          end
        end
        clear_shot_block(session, label)
        local jump_min
        state, jump_min = jump_through_hole(session, label)
        if jump_min < min_y then min_y = jump_min end
        if state.room_id ~= ROOM_BAT or on_door_shelf(state) then
          return state, min_y, attempt
        end
        settle_cavity(session, label)
      end
    else
      session:hold(1, {}, label .. "_reclass")
    end
  end
  return session.state, min_y, 6
end

local function upper_to_speed_hall(session, label)
  iceg.select_weapon(session, 0)
  local min_y, max_x = session.state.samus_y, session.state.samus_x
  local hit = false
  for frame = 0, 499 do
    local state = session.state
    if state.room_id == ROOM_HALL or state.room_id ~= ROOM_BAT then
      hit = true
      break
    end
    if state.samus_y < min_y then min_y = state.samus_y end
    if state.samus_x > max_x then max_x = state.samus_x end
    if state.samus_y > 350 then
      error(string.format("%s: fell from upper band; xy=(%d,%d)", label, state.samus_x, state.samus_y))
    end
    if knockback.is_knockback(state) then
      knockback.escape_knockback_spin(session, {
        prefer_dir = "RIGHT", run_frames = 6, spin_frames = 24, label = label,
        stop_room_id = ROOM_HALL,
      })
    elseif state.samus_y > 200 then
      if in_cavity(state) or state.velocity_y == 0 then
        settle_cavity(session, label)
        local jump_min
        state, jump_min = cavity_to_door_shelf(session, label)
        if jump_min < min_y then min_y = jump_min end
        if state.room_id == ROOM_HALL then
          hit = true
          break
        end
      else
        session:hold(1, {"UP", "X"}, label .. "_reclear")
      end
    else
      local phase = frame % 20
      if phase < 8 then
        session:hold(1, {"RIGHT", "B", "X"}, label .. "_upper_run")
      elseif phase < 12 then
        session:hold(1, {"RIGHT", "B", "A", "X"}, label .. "_upper_hop")
      elseif phase < 16 then
        session:hold(1, {"RIGHT", "X"}, label .. "_upper_shoot")
      else
        session:hold(1, {"R", "X"}, label .. "_upper_ang")
      end
    end
  end
  if session.state.room_id ~= ROOM_HALL then
    local state = session.state
    error(string.format("%s: top-right door missed; room=0x%04X xy=(%d,%d)", label, state.room_id, state.samus_x, state.samus_y))
  end
  return iceg.wait_ordinary_room(session, ROOM_HALL, 320, label)
end

function M.play_bat_cave_to_speed_hall(session)
  local label = "bat_cave_to_speed_hall"
  iceg.require_room(session, ROOM_BAT, label)
  local start = session.frame
  land_door_ledge(session, label)
  shoot_ceiling_hole(session, label)
  gap_skip_to_mid(session, label)
  local state, min_y, attempts = climb_toward_upper(session, label)
  if state.room_id == ROOM_HALL then
    return iceg.wait_ordinary_room(session, ROOM_HALL, 320, label)
  end
  if state.room_id == ROOM_BAT and (on_door_shelf(state) or state.samus_y < 300) then
    return upper_to_speed_hall(session, label)
  end
  error(string.format(
    "%s: upper band / Speed Hall missed; room=0x%04X pose=%d xy=(%d,%d) min_y=%d attempts=%d frames=%d",
    label, session.state.room_id, session.state.pose, session.state.samus_x, session.state.samus_y,
    min_y, attempts, session.frame - start
  ))
end

local function dash_speed_hall(session, label)
  for _ = 1, 30 do
    local state = session:hold(1, {}, label .. "_land")
    if state.velocity_y == 0 and STANDING[state.pose] then
      break
    end
  end
  local max_x = session.state.samus_x
  for _ = 1, 900 do
    local state = session:hold(1, {"RIGHT", "B"}, label .. "_dash")
    if state.samus_x > max_x then max_x = state.samus_x end
    if state.room_id == ROOM_SPEED then
      return state
    end
    if state.room_id ~= ROOM_HALL then
      break
    end
    if state.samus_x >= 2950 and state.velocity_y == 0 and STANDING[state.pose] then
      return state
    end
  end
  if session.state.room_id == ROOM_HALL and session.state.samus_x < 2950 then
    error(string.format("%s: hall dash missed door band; xy=(%d,%d) max_x=%d", label, session.state.samus_x, session.state.samus_y, max_x))
  end
  return session.state
end

local function open_speed_hall_super_door(session, label)
  if session.state.room_id == ROOM_SPEED then
    return iceg.wait_ordinary_room(session, ROOM_SPEED, 320, label)
  end
  iceg.unmorph(session)
  session:hold(10, {"LEFT"}, label .. "_door_back")
  session:hold(10, {}, label .. "_door_settle")
  iceg.select_weapon(session, 2)
  session:hold(4, {"RIGHT"}, label .. "_face_door")
  session:hold(4, {}, label .. "_face_release")
  session:hold(2, {"RIGHT", "X"}, label .. "_super")
  session:hold(70, {}, label .. "_fuse")
  local hit = false
  for frame = 0, 499 do
    local state = session:hold(1, {"RIGHT", "B"}, label .. "_enter")
    if state.room_id == ROOM_SPEED then
      hit = true
      break
    end
    if frame > 0 and frame % 120 == 0 then
      iceg.select_weapon(session, 2)
      session:hold(2, {"RIGHT", "X"}, label .. "_super_retry")
      session:hold(40, {}, label .. "_fuse_retry")
    end
  end
  if not hit then
    error(string.format("%s: right Super door missed", label))
  end
  return iceg.wait_ordinary_room(session, ROOM_SPEED, 320, label)
end

local function collect_speed_booster(session, label)
  iceg.require_room(session, ROOM_SPEED, label)
  if band(session.state.collected_items or 0, ITEM_SPEED) ~= 0 then
    return session.state
  end
  iceg.select_weapon(session, 0)
  iceg.unmorph(session)
  local got = false
  for frame = 0, 279 do
    local state = session.state
    if band(state.collected_items or 0, ITEM_SPEED) ~= 0 then
      got = true
      break
    end
    if state.room_id ~= ROOM_SPEED then
      error(string.format("%s: left Speed room during collect", label))
    end
    if state.pose == 137 or state.pose == 138 then
      iceg.unmorph(session)
    elseif state.samus_x > 190 and state.velocity_y == 0 then
      session:hold(12, {"LEFT"}, label .. "_rebound")
    elseif frame % 12 == 0 then
      session:hold(1, {"X"}, label .. "_shot")
    else
      session:hold(1, {"RIGHT"}, label .. "_walk")
    end
  end
  if band(session.state.collected_items or 0, ITEM_SPEED) == 0 then
    error(string.format("%s: Speed Booster PLM not collected; items=0x%04X", label, session.state.collected_items or 0))
  end
  session:hold(500, {}, label .. "_fanfare")
  iceg.unmorph(session)
  for _ = 1, 30 do
    local state = session:hold(1, {}, label .. "_stand")
    if state.velocity_y == 0 and STANDING[state.pose] then
      break
    end
  end
  session:hold(4, {"LEFT"}, label .. "_face_exit")
  session:hold(6, {}, label .. "_exit_settle")
  return session.state
end

function M.play_speed_hall_to_speed(session)
  local label = "speed_hall_to_speed"
  iceg.require_room(session, ROOM_HALL, label)
  local state = dash_speed_hall(session, label)
  if state.room_id ~= ROOM_SPEED then
    state = open_speed_hall_super_door(session, label)
  else
    state = iceg.wait_ordinary_room(session, ROOM_SPEED, 320, label)
  end
  state = collect_speed_booster(session, label)
  if band(state.collected_items or 0, ITEM_SPEED) == 0 then
    error(string.format("%s: finished without Speed bit", label))
  end
  return session.state
end

return M
