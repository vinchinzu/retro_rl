-- K4.9 Single Chamber → Double Chamber (missile red door).
-- Lua 5.1. Session: step / hold / wait_until / span.

local G = require("wave.geometry")
local knockback = require("skills.knockback")

local ROOM_SC = G.ROOM_SINGLE_CHAMBER
local ROOM_DC = G.ROOM_DOUBLE_CHAMBER
local STANDING = G.STANDING_POSES

local M = {}

local function descend_to_floor(session, label)
  G.unmorph(session)
  G.select_weapon(session, 0)
  for _ = 1, 20 do
    local state = session:hold(1, {}, label .. "_top_stand")
    if state.velocity_y == 0 and STANDING[state.pose] then
      break
    end
  end

  for frame = 0, 119 do
    local state = session.state
    if state.room_id ~= ROOM_SC then
      return
    end
    if state.samus_y > G.SC_TOP_Y then
      break
    end
    if knockback.is_knockback(state) then
      knockback.escape_kb(session, label, "LEFT", {stop_room_id = ROOM_DC})
    elseif state.samus_x < 130 then
      if frame % 12 < 4 then
        session:hold(1, {"RIGHT", "X"}, label .. "_top_shot")
      else
        session:hold(1, {"RIGHT"}, label .. "_top_walk")
      end
    else
      session:hold(1, {"RIGHT"}, label .. "_top_edge")
    end
  end

  for _ = 1, 140 do
    local state = session.state
    if state.room_id ~= ROOM_SC then
      return
    end
    if G.SC_MID_Y[1] <= state.samus_y and state.samus_y <= G.SC_MID_Y[2]
        and state.velocity_y == 0 then
      break
    end
    if G.SC_FLOOR_Y[1] <= state.samus_y and state.samus_y <= G.SC_FLOOR_Y[2]
        and state.velocity_y == 0 then
      return
    end
    if knockback.is_knockback(state) then
      knockback.escape_kb(session, label, "LEFT", {stop_room_id = ROOM_DC})
    elseif state.samus_x > 150 then
      session:hold(1, {"LEFT"}, label .. "_air_l")
    else
      session:hold(1, {}, label .. "_air")
    end
  end

  for frame = 0, 99 do
    local state = session.state
    if state.room_id ~= ROOM_SC then
      return
    end
    if state.samus_y > G.SC_MID_Y[2] + 10 then
      break
    end
    if G.SC_FLOOR_Y[1] <= state.samus_y and state.samus_y <= G.SC_FLOOR_Y[2]
        and state.velocity_y == 0 then
      break
    end
    if knockback.is_knockback(state) then
      knockback.escape_kb(session, label, "LEFT", {stop_room_id = ROOM_DC})
    elseif state.samus_x > 62 then
      if frame % 10 < 3 then
        session:hold(1, {"LEFT", "X"}, label .. "_mid_shot")
      else
        session:hold(1, {"LEFT"}, label .. "_mid_walk")
      end
    else
      break
    end
  end

  for frame = 0, 139 do
    local state = session.state
    if state.room_id ~= ROOM_SC then
      return
    end
    if G.SC_FLOOR_Y[1] <= state.samus_y and state.samus_y <= G.SC_FLOOR_Y[2]
        and state.velocity_y == 0 then
      break
    end
    if knockback.is_knockback(state) then
      knockback.escape_kb(session, label, "RIGHT", {stop_room_id = ROOM_DC})
    elseif G.SC_MID_Y[1] <= state.samus_y and state.samus_y <= G.SC_MID_Y[2]
        and state.velocity_y == 0 then
      if frame < 8 then
        session:hold(1, {"LEFT"}, label .. "_step_off")
      else
        session:hold(1, {}, label .. "_lip_wait")
      end
    elseif state.samus_x < 75 then
      session:hold(1, {"RIGHT"}, label .. "_floor_drift_r")
    elseif state.samus_x > 100 then
      session:hold(1, {"LEFT"}, label .. "_floor_drift_l")
    else
      session:hold(1, {}, label .. "_floor_fall")
    end
  end

  G.unmorph(session)
  for _ = 1, 60 do
    local state = session.state
    if state.room_id ~= ROOM_SC or state.room_id == ROOM_DC then
      return
    end
    if knockback.is_knockback(state) then
      knockback.escape_kb(session, label, "RIGHT", {stop_room_id = ROOM_DC})
    elseif not (G.SC_FLOOR_Y[1] <= state.samus_y and state.samus_y <= G.SC_FLOOR_Y[2]
        and state.velocity_y == 0) then
      if state.samus_y > G.SC_FLOOR_Y[2] + 40 then
        return
      end
      session:hold(1, {}, label .. "_floor_wait")
    elseif G.SC_SHOT_X[1] <= state.samus_x and state.samus_x <= G.SC_SHOT_X[2] then
      return
    elseif state.samus_x < G.SC_SHOT_X[1] then
      session:hold(1, {"RIGHT"}, label .. "_seat_r")
    else
      session:hold(1, {"LEFT"}, label .. "_seat_l")
    end
  end
end

local function missile_door_and_enter(session, label)
  G.unmorph(session)
  G.select_weapon(session, 1)
  session:hold(3, {"RIGHT"}, label .. "_face")
  session:hold(8, {}, label .. "_face_release")

  for frame = 0, 109 do
    local state = session.state
    if state.room_id == ROOM_DC or state.room_id ~= ROOM_SC then
      return
    end
    if knockback.is_knockback(state) then
      knockback.escape_kb(session, label, "RIGHT", {stop_room_id = ROOM_DC})
      G.select_weapon(session, 1)
    elseif state.samus_x > G.SC_SHOT_X[2] + 20 and state.velocity_y == 0 then
      session:hold(1, {"LEFT"}, label .. "_reseat")
    elseif state.samus_x < G.SC_SHOT_X[1] - 15 and state.velocity_y == 0 then
      session:hold(1, {"RIGHT"}, label .. "_reseat_r")
    elseif frame % 10 < 2 then
      session:hold(1, {"X"}, label .. "_missile")
    else
      session:hold(1, {}, label .. "_missile_wait")
    end
  end

  session:hold(12, {}, label .. "_fuse")
  for _ = 1, 30 do
    local state = session.state
    if state.room_id ~= ROOM_SC then
      return
    end
    if state.samus_x >= 145 and state.velocity_y == 0 then
      break
    end
    if state.samus_y > G.SC_FLOOR_Y[2] + 20 then
      break
    end
    session:hold(1, {"RIGHT"}, label .. "_walkup")
  end

  for _ = 1, 12 do
    local state = session.state
    if state.room_id == ROOM_DC or state.room_id ~= ROOM_SC then
      return
    end
    session:hold(1, {"RIGHT", "B", "A"}, label .. "_gap_hop")
  end

  for frame = 0, 259 do
    local state = session.state
    if state.room_id == ROOM_DC or state.room_id ~= ROOM_SC then
      return
    end
    if knockback.is_knockback(state) then
      knockback.escape_kb(session, label, "RIGHT", {stop_room_id = ROOM_DC})
    elseif state.samus_y > G.SC_FLOOR_Y[2] + 50 then
      if state.samus_x > 100 then
        session:hold(1, {"LEFT"}, label .. "_under_left")
      else
        session:hold(1, {"LEFT", "A"}, label .. "_under_up")
      end
    elseif state.velocity_y ~= 0 or state.samus_y < G.SC_FLOOR_Y[1] - 5 then
      session:hold(1, {"RIGHT"}, label .. "_air_r")
    elseif state.samus_x < G.SC_DOOR_X then
      if frame > 0 and frame % 90 == 0 then
        G.select_weapon(session, 1)
        session:hold(2, {"RIGHT", "X"}, label .. "_reopen")
        session:hold(20, {}, label .. "_reopen_fuse")
      else
        session:hold(1, {"RIGHT", "B"}, label .. "_run")
      end
    else
      session:hold(1, {"RIGHT"}, label .. "_door_push")
    end
  end
end

function M.play_single_to_double_chamber(session)
  local label = "single_to_double_chamber"
  G.require_room(session, ROOM_SC, label)
  local start = session.frame
  if session.state.room_id == ROOM_SC then
    descend_to_floor(session, label)
  end
  if session.state.room_id == ROOM_SC then
    missile_door_and_enter(session, label)
  end
  if session.state.room_id ~= ROOM_DC then
    local state = session.state
    error(string.format(
      "%s: Double Chamber door missed; room=0x%04X pose=%d xy=(%d,%d) missiles=%d frames=%d",
      label, state.room_id, state.pose, state.samus_x, state.samus_y,
      state.missiles or 0, session.frame - start
    ))
  end
  return G.wait_ordinary_room(session, ROOM_DC, G.SC_DOUBLE_SETTLE, label)
end

return M
