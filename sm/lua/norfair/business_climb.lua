-- Hi-Jump platform climb in Business Center (elevator return).
-- Lua 5.1. Session: step / hold / wait_until / span. No save-state dumps.

local iceg = require("ice.geometry")
local geom = require("skills.geometry")

local ROOM_BUSINESS, ROOM_WAREHOUSE = 0xA7DE, 0xA6A1
local ITEM_HI_JUMP = 0x0100
local STANDING = {
  [1] = true, [2] = true, [9] = true, [10] = true, [25] = true, [26] = true,
  [27] = true, [28] = true, [37] = true, [38] = true, [137] = true, [138] = true,
}
local band = iceg.band

local M = {}

local function wait_standing_y(session, y, timeout, reason)
  timeout = timeout or 90
  reason = reason or "business_standing"
  for _ = 1, timeout do
    local state = session.state
    if state.samus_y == y and STANDING[state.pose] and state.velocity_y == 0 then
      return state
    end
    session:hold(1, {}, reason)
  end
  error(string.format("%s: expected y=%d", reason, y))
end

local function on_1339(state)
  return state.samus_y == 1339 and STANDING[state.pose] and state.velocity_y == 0
end

local function on_elevator(state)
  return state.samus_y == 683 and STANDING[state.pose] and state.velocity_y == 0
end

function M.business_high_jump_platforms(session, runup_907, pos_1339, bound_floor_left)
  runup_907 = runup_907 or 14
  pos_1339 = pos_1339 or 84
  bound_floor_left = bound_floor_left or false
  iceg.unmorph(session)
  if not on_1339(session.state) then
    for _, direction in ipairs({"LEFT", "LEFT", "RIGHT"}) do
      session:hold(12, {}, "business_climb_release")
      for _ = 1, 85 do
        local st = session.state
        if st.room_id ~= ROOM_BUSINESS then
          error("business_climb_setup: left Business")
        end
        if direction == "LEFT" and st.samus_y >= 1350 and st.samus_x <= 40 then
          session:hold(1, {"RIGHT", "B", "A"}, "business_climb_setup_door")
        else
          session:hold(1, {direction, "B", "A"}, "business_climb_setup")
        end
      end
      session:hold(30, {}, "business_climb_setup_land")
    end
  end

  iceg.unmorph(session)
  session:hold(12, {}, "business_1339_settle")
  wait_standing_y(session, 1339, 60, "business_1339_ground")
  for _ = 1, 80 do
    local state = session.state
    if state.samus_x <= pos_1339 then
      break
    end
    if state.samus_y ~= 1339 or not STANDING[state.pose] then
      session:hold(1, {}, "business_1339_replant")
      if session.state.samus_y ~= 1339 then
        break
      end
    else
      session:hold(1, {"LEFT"}, "business_1339_position")
    end
  end
  if bound_floor_left then
    session:hold(3, {"RIGHT"}, "business_1339_brake")
  else
    session:hold(4, {"RIGHT"}, "business_1339_brake")
  end
  session:hold(8, {}, "business_1339_release")
  wait_standing_y(session, 1339, 40, "business_1339_prejump")
  for frame = 0, 119 do
    local buttons
    if frame < 14 then buttons = {"LEFT", "A"}
    elseif frame < 24 then buttons = {"A"}
    else buttons = {"RIGHT", "A"} end
    local state = session:hold(1, buttons, "business_to_1227")
    if frame > 45 and state.samus_y == 1227 and state.samus_x >= 120 then
      break
    end
  end
  session:hold(3, {"LEFT"}, "business_1227_brake")
  session:hold(12, {}, "business_1227_settle")
  wait_standing_y(session, 1227, 50, "business_1227_land")

  iceg.unmorph(session)
  session:hold(15, {}, "business_1227_release")
  for _ = 1, 80 do
    if session.state.samus_x <= 105 then break end
    session:hold(1, {"LEFT"}, "business_1227_back")
  end
  session:hold(4, {"RIGHT"}, "business_1227_brake2")
  session:hold(4, {}, "business_1227_run_release")
  session:hold(8, {"RIGHT", "B"}, "business_1227_runup")
  for frame = 0, 139 do
    local buttons = (frame < 90) and {"RIGHT", "B", "A"} or {"LEFT", "A"}
    local state = session:hold(1, buttons, "business_to_1147")
    if frame > 88 and state.samus_y == 1147 and state.samus_x >= 192 then
      break
    end
  end
  session:hold(3, {"LEFT"}, "business_1147_brake")
  session:hold(12, {}, "business_1147_settle")
  wait_standing_y(session, 1147, 50, "business_1147_land")

  iceg.unmorph(session)
  session:hold(16, {}, "business_1147_release")
  for frame = 0, 149 do
    local buttons = (frame < 85) and {"LEFT", "B", "A"} or {"RIGHT", "A"}
    local state = session:hold(1, buttons, "business_to_1067")
    if frame > 100 and state.samus_y == 1067 and 95 <= state.samus_x and state.samus_x <= 160 then
      break
    end
  end
  session:hold(30, {}, "business_1067_settle")
  wait_standing_y(session, 1067, 50, "business_1067_land")

  iceg.unmorph(session)
  session:hold(12, {}, "business_1067_release")
  for _ = 1, 80 do
    if session.state.samus_x <= 92 then break end
    session:hold(1, {"LEFT"}, "business_1067_position")
  end
  session:hold(4, {"RIGHT"}, "business_1067_brake")
  session:hold(8, {}, "business_1067_jump_release")
  for frame = 0, 99 do
    local buttons = (frame < 14) and {"A"} or {"RIGHT", "B", "A"}
    local state = session:hold(1, buttons, "business_to_987")
    if frame > 25 and state.samus_y == 987 and (state.pose == 1 or state.pose == 2 or state.pose == 9 or state.pose == 10) then
      break
    end
  end
  session:hold(4, {"RIGHT"}, "business_987_brake")
  session:hold(12, {}, "business_987_settle")
  wait_standing_y(session, 987, 50, "business_987_land")

  iceg.unmorph(session)
  session:hold(12, {}, "business_987_release")
  wait_standing_y(session, 987, 40, "business_987_pre_907")
  session:hold(runup_907, {"RIGHT", "B"}, "business_987_runup")
  for frame = 0, 99 do
    local state = session:hold(1, {"RIGHT", "B", "A"}, "business_to_907")
    if frame > 35 and state.samus_y == 907 and state.samus_x >= 160 then
      break
    end
  end
  for _ = 1, 60 do
    if session.state.samus_x <= 165 then break end
    session:hold(1, {"LEFT"}, "business_907_brake")
  end
  session:hold(2, {"RIGHT"}, "business_907_brake")
  session:hold(12, {}, "business_907_settle")
  wait_standing_y(session, 907, 50, "business_907_land")

  iceg.unmorph(session)
  session:hold(12, {}, "business_907_release")
  for _ = 1, 80 do
    if session.state.samus_x >= 205 then break end
    session:hold(1, {"RIGHT"}, "business_907_back")
  end
  session:hold(3, {"LEFT"}, "business_907_brake2")
  session:hold(5, {}, "business_907_run_release")
  session:hold(8, {"LEFT", "B"}, "business_907_runup")
  for frame = 0, 89 do
    local state = session:hold(1, {"LEFT", "B", "A"}, "business_to_843")
    if frame > 35 and state.samus_y == 843 and 108 <= state.samus_x and state.samus_x <= 160 then
      break
    end
  end
  session:hold(2, {"RIGHT"}, "business_843_brake")
  session:hold(12, {}, "business_843_settle")
  wait_standing_y(session, 843, 50, "business_843_land")

  iceg.unmorph(session)
  session:hold(12, {}, "business_843_release")
  for _ = 1, 80 do
    if session.state.samus_x >= 145 then break end
    session:hold(1, {"RIGHT"}, "business_843_position")
  end
  session:hold(3, {"LEFT"}, "business_843_brake2")
  session:hold(6, {}, "business_843_jump_release")
  for frame = 0, 89 do
    local buttons = (frame < 10) and {"A"} or {"LEFT", "B", "A"}
    local state = session:hold(1, buttons, "business_to_779")
    if frame > 25 and state.samus_y == 779 and state.samus_x <= 115 then
      break
    end
  end
  session:hold(2, {"RIGHT"}, "business_779_brake")
  session:hold(12, {}, "business_779_settle")
  wait_standing_y(session, 779, 50, "business_779_land")

  iceg.unmorph(session)
  session:hold(12, {}, "business_779_release")
  wait_standing_y(session, 779, 40, "business_779_pre_elev")
  for _ = 1, 80 do
    local state = session.state
    if state.samus_x <= 80 then break end
    if state.samus_y ~= 779 or not STANDING[state.pose] then
      session:hold(1, {}, "business_779_replant")
      if session.state.samus_y ~= 779 then
        error("business_779_position: walked off platform")
      end
    else
      session:hold(1, {"LEFT"}, "business_779_position")
    end
  end
  session:hold(3, {"RIGHT"}, "business_779_brake2")
  session:hold(6, {}, "business_779_jump_release")
  wait_standing_y(session, 779, 30, "business_779_prejump")
  for frame = 0, 119 do
    local buttons = (frame < 18) and {"A"} or {"RIGHT", "B", "A"}
    local state = session:hold(1, buttons, "business_to_elevator")
    if frame > 45 and state.samus_y == 683 and 95 <= state.samus_x and state.samus_x <= 160 then
      break
    end
  end
  session:hold(2, {"LEFT"}, "business_elevator_brake")
  session:hold(12, {}, "business_elevator_settle")
  wait_standing_y(session, 683, 50, "business_elevator_land")
end

-- Python name used by ice/business_to_gate.
M._business_high_jump_platforms = function(session, opts)
  opts = opts or {}
  return M.business_high_jump_platforms(
    session, opts.runup_907, opts.pos_1339, opts.bound_floor_left
  )
end

local function fall_to_floor(session)
  local direction = "RIGHT"
  local landed = false
  for frame = 0, 599 do
    local state = session.state
    if STANDING[state.pose] and state.velocity_y == 0 and state.samus_y >= 1405 then
      landed = true
      break
    end
    if state.samus_x <= 50 then
      direction = "RIGHT"
    elseif state.samus_x >= 210 then
      direction = "LEFT"
    end
    local phase = frame % 70
    local buttons = (phase < 45) and {direction, "B"} or {direction, "B", "A"}
    session:hold(1, buttons, "business_floor_recover")
  end
  if not landed then
    error("business_floor_recover")
  end
  for _ = 1, 80 do
    if session.state.samus_x >= 88 then break end
    session:hold(1, {"RIGHT"}, "business_floor_anchor")
  end
  session:hold(4, {"LEFT"}, "business_floor_anchor_brake")
  session:hold(15, {}, "business_floor_recover_settle")
end

local function climb_to_elevator(session, label)
  iceg.unmorph(session)
  if session.state.room_id ~= ROOM_BUSINESS then
    error(label .. ": not in Business")
  end
  if on_elevator(session.state) then
    return
  end
  local y0 = session.state.samus_y
  if y0 < 1350 and not on_elevator(session.state) then
    fall_to_floor(session)
  end
  local attempts = {{14, 84, false}, {8, 84, false}, {18, 90, false}, {20, 90, false}}
  local last_err
  for i, row in ipairs(attempts) do
    local ok, err = pcall(function()
      if i > 1 then
        if session.state.room_id ~= ROOM_BUSINESS then
          error(label .. ": left Business during climb retry")
        end
        fall_to_floor(session)
      end
      M.business_high_jump_platforms(session, row[1], row[2], row[3])
    end)
    if ok then
      last_err = nil
      break
    else
      last_err = err
      if session.state.room_id ~= ROOM_BUSINESS then
        error(label .. ": left Business during climb")
      end
    end
  end
  if last_err then
    error(last_err)
  end
end

function M.play_business_to_warehouse(session)
  iceg.require_room(session, ROOM_BUSINESS, "business_to_warehouse")
  if band(session.state.collected_items or 0, ITEM_HI_JUMP) == 0 then
    error("business_to_warehouse: Hi-Jump not collected")
  end
  climb_to_elevator(session, "business_to_warehouse")
  if session.state.samus_y ~= 683 or not STANDING[session.state.pose] then
    error("business_to_warehouse: not on elevator platform")
  end
  local hit = false
  for _ = 1, 1000 do
    local state = session:hold(1, {"UP"}, "business_elevator_up")
    if state.room_id == ROOM_WAREHOUSE then
      hit = true
      break
    end
  end
  if not hit then
    error("business_to_warehouse: elevator failed")
  end
  iceg.wait_ordinary_room(session, ROOM_WAREHOUSE, 360, "business_to_warehouse")
  session:hold(30, {}, "warehouse_elevator_top")
  for _ = 1, 160 do
    local state = session.state
    if state.samus_x <= 40 and state.samus_y <= 145 then
      break
    end
    session:hold(1, {"LEFT"}, "warehouse_elevator_exit")
  end
  session:hold(30, {}, "warehouse_elevator_exit_settle")
  return session.state
end

return M
