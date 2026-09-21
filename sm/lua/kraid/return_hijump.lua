-- Hi-Jump room → shaft → Business → Warehouse after collect.
-- Port of return_hijump.py plus play_business_to_warehouse (needed on Kraid path;
-- Norfair owns the Ice-tuned Business climb extras).

local geo = require("red_tower.ctrl")

local M = {}

local STANDING = geo.STANDING

local function wait_standing_y(session, y, timeout, reason)
  timeout = timeout or 90
  reason = reason or "business_standing"
  for _ = 1, timeout do
    local state = session.state
    if state.samus_y == y and STANDING[state.pose] and (state.velocity_y or 0) == 0 then
      return state
    end
    geo.hold(session, 1, {}, reason)
  end
  error(string.format("%s: expected y=%s: %s", reason, tostring(y), geo.fmt_state(session.state)))
end

local function on_business_1339(state)
  return state.samus_y == 1339 and STANDING[state.pose] and (state.velocity_y or 0) == 0
end

local function on_elevator(state)
  return state.samus_y == 683 and STANDING[state.pose] and (state.velocity_y or 0) == 0
end

local function fall_to_business_floor(session)
  local direction = "RIGHT"
  local landed = false
  for frame = 0, 599 do
    local state = session.state
    if STANDING[state.pose] and (state.velocity_y or 0) == 0 and state.samus_y >= 1405 then
      landed = true
      break
    end
    if state.samus_x <= 50 then
      direction = "RIGHT"
    elseif state.samus_x >= 210 then
      direction = "LEFT"
    end
    local names
    if frame % 70 < 45 then
      names = {direction, "B"}
    else
      names = {direction, "B", "A"}
    end
    geo.hold(session, 1, names, "business_floor_recover")
  end
  if not landed then
    error("business_floor_recover: " .. geo.fmt_state(session.state))
  end
  for _ = 1, 80 do
    local state = session.state
    if state.samus_x >= 88 then
      break
    end
    geo.hold(session, 1, {"RIGHT"}, "business_floor_anchor")
  end
  geo.hold(session, 4, {"LEFT"}, "business_floor_anchor_brake")
  geo.hold(session, 15, {}, "business_floor_recover_settle")
end

local function anchor_floor_midright(session, label)
  geo.unmorph(session)
  for _ = 1, 200 do
    local st = session.state
    if st.room_id ~= geo.ROOM_BUSINESS then
      error(label .. ": left Business during floor anchor: " .. geo.fmt_state(st))
    end
    local x, y = st.samus_x, st.samus_y
    local grounded = (st.velocity_y or 0) == 0 and STANDING[st.pose]
    if y < 1350 then
      geo.hold(session, 1, {"RIGHT", "B"}, label .. "_anchor_air")
    elseif grounded and x >= 170 and x <= 230 then
      geo.hold(session, 10, {}, label .. "_anchor_settle")
      return
    elseif x < 170 then
      geo.hold(session, 1, {"RIGHT", "B"}, label .. "_anchor_r")
    elseif x > 230 and grounded then
      if x > 80 then
        geo.hold(session, 1, {"LEFT"}, label .. "_anchor_l")
      else
        geo.hold(session, 1, {"RIGHT"}, label .. "_anchor_bounce")
      end
    else
      geo.hold(session, 1, {}, label .. "_anchor_idle")
    end
  end
end

local function classic_setup_to_1339(session)
  for _, direction in ipairs({"LEFT", "LEFT", "RIGHT"}) do
    geo.hold(session, 12, {}, "business_climb_release")
    for _ = 1, 85 do
      local st = session.state
      if st.room_id ~= geo.ROOM_BUSINESS then
        error("business_climb_setup: left Business: " .. geo.fmt_state(st))
      end
      if direction == "LEFT" and st.samus_y >= 1350 and st.samus_x <= 40 then
        geo.hold(session, 1, {"RIGHT", "B", "A"}, "business_climb_setup_door")
      else
        geo.hold(session, 1, {direction, "B", "A"}, "business_climb_setup")
      end
    end
    if session.state.room_id ~= geo.ROOM_BUSINESS then
      error("business_climb_setup: left Business: " .. geo.fmt_state(session.state))
    end
    geo.hold(session, 30, {}, "business_climb_setup_land")
    if session.state.room_id ~= geo.ROOM_BUSINESS then
      error("business_climb_setup_land: left Business: " .. geo.fmt_state(session.state))
    end
  end
end

local function business_high_jump_platforms(session, runup_907, pos_1339)
  runup_907 = runup_907 or 14
  pos_1339 = pos_1339 or 84
  geo.unmorph(session)
  if not on_business_1339(session.state) then
    classic_setup_to_1339(session)
  end

  geo.unmorph(session)
  geo.hold(session, 12, {}, "business_1339_settle")
  wait_standing_y(session, 1339, 60, "business_1339_ground")
  for _ = 1, 80 do
    local state = session.state
    if state.samus_x <= pos_1339 then
      break
    end
    if state.samus_y ~= 1339 or not STANDING[state.pose] then
      geo.hold(session, 1, {}, "business_1339_replant")
      if session.state.samus_y ~= 1339 then
        geo.unmorph(session)
        classic_setup_to_1339(session)
        wait_standing_y(session, 1339, 60, "business_1339_ground_retry")
        break
      end
    else
      geo.hold(session, 1, {"LEFT"}, "business_1339_position")
    end
  end
  geo.hold(session, 4, {"RIGHT"}, "business_1339_brake")
  geo.hold(session, 8, {}, "business_1339_release")
  wait_standing_y(session, 1339, 40, "business_1339_prejump")
  for frame = 0, 119 do
    local names
    if frame < 14 then
      names = {"LEFT", "A"}
    elseif frame < 24 then
      names = {"A"}
    else
      names = {"RIGHT", "A"}
    end
    local state = geo.hold(session, 1, names, "business_to_1227")
    if frame > 45 and state.samus_y == 1227 and state.samus_x >= 120 then
      break
    end
  end
  geo.hold(session, 3, {"LEFT"}, "business_1227_brake")
  geo.hold(session, 12, {}, "business_1227_settle")
  wait_standing_y(session, 1227, 50, "business_1227_land")

  geo.unmorph(session)
  geo.hold(session, 15, {}, "business_1227_release")
  for _ = 1, 80 do
    if session.state.samus_x <= 105 then
      break
    end
    geo.hold(session, 1, {"LEFT"}, "business_1227_back")
  end
  geo.hold(session, 4, {"RIGHT"}, "business_1227_brake2")
  geo.hold(session, 4, {}, "business_1227_run_release")
  geo.hold(session, 8, {"RIGHT", "B"}, "business_1227_runup")
  for frame = 0, 139 do
    local names = (frame < 90) and {"RIGHT", "B", "A"} or {"LEFT", "A"}
    local state = geo.hold(session, 1, names, "business_to_1147")
    if frame > 88 and state.samus_y == 1147 and state.samus_x >= 192 then
      break
    end
  end
  geo.hold(session, 3, {"LEFT"}, "business_1147_brake")
  geo.hold(session, 12, {}, "business_1147_settle")
  wait_standing_y(session, 1147, 50, "business_1147_land")

  geo.unmorph(session)
  geo.hold(session, 16, {}, "business_1147_release")
  for frame = 0, 149 do
    local names = (frame < 85) and {"LEFT", "B", "A"} or {"RIGHT", "A"}
    local state = geo.hold(session, 1, names, "business_to_1067")
    if frame > 100 and state.samus_y == 1067 and state.samus_x >= 95 and state.samus_x <= 160 then
      break
    end
  end
  geo.hold(session, 30, {}, "business_1067_settle")
  wait_standing_y(session, 1067, 50, "business_1067_land")

  geo.unmorph(session)
  geo.hold(session, 12, {}, "business_1067_release")
  for _ = 1, 80 do
    if session.state.samus_x <= 92 then
      break
    end
    geo.hold(session, 1, {"LEFT"}, "business_1067_position")
  end
  geo.hold(session, 4, {"RIGHT"}, "business_1067_brake")
  geo.hold(session, 8, {}, "business_1067_jump_release")
  local stand_land = geo.set(1, 2, 9, 10)
  for frame = 0, 99 do
    local names = (frame < 14) and {"A"} or {"RIGHT", "B", "A"}
    local state = geo.hold(session, 1, names, "business_to_987")
    if frame > 25 and state.samus_y == 987 and stand_land[state.pose] then
      break
    end
  end
  geo.hold(session, 4, {"RIGHT"}, "business_987_brake")
  geo.hold(session, 12, {}, "business_987_settle")
  wait_standing_y(session, 987, 50, "business_987_land")

  geo.unmorph(session)
  geo.hold(session, 12, {}, "business_987_release")
  wait_standing_y(session, 987, 40, "business_987_pre_907")
  geo.hold(session, runup_907, {"RIGHT", "B"}, "business_987_runup")
  for frame = 0, 99 do
    local state = geo.hold(session, 1, {"RIGHT", "B", "A"}, "business_to_907")
    if frame > 35 and state.samus_y == 907 and state.samus_x >= 160 then
      break
    end
  end
  for _ = 1, 60 do
    if session.state.samus_x <= 165 then
      break
    end
    geo.hold(session, 1, {"LEFT"}, "business_907_brake")
  end
  geo.hold(session, 2, {"RIGHT"}, "business_907_brake")
  geo.hold(session, 12, {}, "business_907_settle")
  wait_standing_y(session, 907, 50, "business_907_land")

  geo.unmorph(session)
  geo.hold(session, 12, {}, "business_907_release")
  for _ = 1, 80 do
    if session.state.samus_x >= 205 then
      break
    end
    geo.hold(session, 1, {"RIGHT"}, "business_907_back")
  end
  geo.hold(session, 3, {"LEFT"}, "business_907_brake2")
  geo.hold(session, 5, {}, "business_907_run_release")
  geo.hold(session, 8, {"LEFT", "B"}, "business_907_runup")
  for frame = 0, 89 do
    local state = geo.hold(session, 1, {"LEFT", "B", "A"}, "business_to_843")
    if frame > 35 and state.samus_y == 843 and state.samus_x >= 108 and state.samus_x <= 160 then
      break
    end
  end
  geo.hold(session, 2, {"RIGHT"}, "business_843_brake")
  geo.hold(session, 12, {}, "business_843_settle")
  wait_standing_y(session, 843, 50, "business_843_land")

  geo.unmorph(session)
  geo.hold(session, 12, {}, "business_843_release")
  for _ = 1, 80 do
    if session.state.samus_x >= 145 then
      break
    end
    geo.hold(session, 1, {"RIGHT"}, "business_843_position")
  end
  geo.hold(session, 3, {"LEFT"}, "business_843_brake2")
  geo.hold(session, 6, {}, "business_843_jump_release")
  for frame = 0, 89 do
    local names = (frame < 10) and {"A"} or {"LEFT", "B", "A"}
    local state = geo.hold(session, 1, names, "business_to_779")
    if frame > 25 and state.samus_y == 779 and state.samus_x <= 115 then
      break
    end
  end
  geo.hold(session, 2, {"RIGHT"}, "business_779_brake")
  geo.hold(session, 12, {}, "business_779_settle")
  wait_standing_y(session, 779, 50, "business_779_land")

  geo.unmorph(session)
  geo.hold(session, 12, {}, "business_779_release")
  wait_standing_y(session, 779, 40, "business_779_pre_elev")
  for _ = 1, 80 do
    local state = session.state
    if state.samus_x <= 80 then
      break
    end
    if state.samus_y ~= 779 or not STANDING[state.pose] then
      geo.hold(session, 1, {}, "business_779_replant")
      if session.state.samus_y ~= 779 then
        error("business_779_position: walked off platform: " .. geo.fmt_state(session.state))
      end
    else
      geo.hold(session, 1, {"LEFT"}, "business_779_position")
    end
  end
  geo.hold(session, 3, {"RIGHT"}, "business_779_brake2")
  geo.hold(session, 6, {}, "business_779_jump_release")
  wait_standing_y(session, 779, 30, "business_779_prejump")
  for frame = 0, 119 do
    local names = (frame < 18) and {"A"} or {"RIGHT", "B", "A"}
    local state = geo.hold(session, 1, names, "business_to_elevator")
    if frame > 45 and state.samus_y == 683 and state.samus_x >= 95 and state.samus_x <= 160 then
      break
    end
  end
  geo.hold(session, 2, {"LEFT"}, "business_elevator_brake")
  geo.hold(session, 12, {}, "business_elevator_settle")
  wait_standing_y(session, 683, 50, "business_elevator_land")
  if not (session.state.samus_x >= 95 and session.state.samus_x <= 160) then
    for _ = 1, 40 do
      local x = session.state.samus_x
      if x >= 100 and x <= 150 then
        break
      end
      local dir = (x > 150) and "LEFT" or "RIGHT"
      geo.hold(session, 1, {dir}, "business_elevator_center")
    end
    geo.hold(session, 8, {}, "business_elevator_center_settle")
    wait_standing_y(session, 683, 40, "business_elevator_recenter")
  end
end

local function climb_business_to_elevator(session, label)
  label = label or "business_climb"
  geo.unmorph(session)
  if session.state.room_id ~= geo.ROOM_BUSINESS then
    error(label .. ": not in Business: " .. geo.fmt_state(session.state))
  end
  if on_elevator(session.state) then
    return
  end
  local y0 = session.state.samus_y
  if y0 < 1350 and not on_elevator(session.state) then
    fall_to_business_floor(session)
    anchor_floor_midright(session, label)
  end

  local attempts = {
    {14, 84},
    {8, 84},
    {18, 90},
    {20, 90},
  }
  local last_err
  for i = 1, #attempts do
    local ok, err = pcall(function()
      if i > 1 then
        if session.state.room_id ~= geo.ROOM_BUSINESS then
          error(label .. ": left Business during climb retry: " .. geo.fmt_state(session.state))
        end
        fall_to_business_floor(session)
        anchor_floor_midright(session, label .. "_retry" .. (i - 1))
      end
      business_high_jump_platforms(session, attempts[i][1], attempts[i][2])
    end)
    if ok then
      last_err = nil
      break
    end
    last_err = err
    if session.state.room_id ~= geo.ROOM_BUSINESS then
      error(label .. ": left Business during climb: " .. geo.fmt_state(session.state))
    end
  end
  if last_err then
    error(last_err)
  end
  if not on_elevator(session.state) then
    if session.state.samus_y ~= 683 or not STANDING[session.state.pose] then
      error(label .. ": not on elevator platform: " .. geo.fmt_state(session.state))
    end
  end
end

function M.play_business_to_warehouse(session)
  geo.require_room(session, geo.ROOM_BUSINESS, "business_to_warehouse")
  if not geo.has_hi_jump(session.state) then
    error("business_to_warehouse: Hi-Jump not collected: " .. geo.fmt_state(session.state))
  end
  climb_business_to_elevator(session, "business_to_warehouse")
  if session.state.samus_y ~= 683 or not STANDING[session.state.pose] then
    error("business_to_warehouse: not on elevator platform: " .. geo.fmt_state(session.state))
  end
  local reached = false
  local state
  for _ = 1, 1000 do
    state = geo.hold(session, 1, {"UP"}, "business_elevator_up")
    if state.room_id == geo.ROOM_WAREHOUSE then
      reached = true
      break
    end
  end
  if not reached then
    error("business_to_warehouse: elevator failed: " .. geo.fmt_state(session.state))
  end
  geo.wait_ordinary_room(session, geo.ROOM_WAREHOUSE, {
    settle_frames = 360,
    label = "business_to_warehouse",
  })
  geo.hold(session, 30, {}, "warehouse_elevator_top")
  for _ = 1, 160 do
    state = session.state
    if state.samus_x <= 40 and state.samus_y <= 145 then
      break
    end
    geo.hold(session, 1, {"LEFT"}, "warehouse_elevator_exit")
  end
  geo.hold(session, 30, {}, "warehouse_elevator_exit_settle")
  return session.state
end

function M.play_hj_room_to_shaft(session)
  geo.require_room(session, geo.ROOM_HJ, "hj_room_to_shaft")
  geo.unmorph(session)
  geo.hold(session, 20, {}, "hj_room_return_settle")
  for _ = 1, 80 do
    local state = session.state
    if state.samus_x <= 80 then
      break
    end
    geo.hold(session, 1, {"LEFT", "B"}, "hj_room_return_backoff")
  end
  geo.hold(session, 8, {"RIGHT"}, "hj_room_return_brake")
  geo.hold(session, 10, {}, "hj_room_return_release")
  geo.hold(session, 12, {"RIGHT", "B"}, "hj_room_return_runup")
  for _ = 1, 120 do
    local state = geo.hold(session, 1, {"RIGHT", "B", "A"}, "hj_room_return_cross")
    if state.samus_x >= 181 then
      break
    end
  end
  geo.hold(session, 80, {}, "hj_room_return_land")

  geo.unmorph(session)
  geo.select_weapon(session, 0)
  for _ = 1, 80 do
    local state = session.state
    if state.samus_x <= 185 then
      break
    end
    geo.hold(session, 1, {"LEFT"}, "hj_room_return_door_backoff")
  end
  geo.hold(session, 8, {"RIGHT"}, "hj_room_return_door_brake")
  geo.hold(session, 8, {}, "hj_room_return_door_settle")
  geo.hold(session, 3, {"RIGHT"}, "hj_room_return_face_door")
  geo.hold(session, 3, {}, "hj_room_return_face_release")
  geo.hold(session, 1, {"X"}, "hj_room_return_door_shot")
  geo.hold(session, 40, {}, "hj_room_return_door_open")
  local entered = false
  local state
  for _ = 1, 420 do
    state = geo.hold(session, 1, {"RIGHT", "B", "A"}, "hj_room_return_enter")
    if state.room_id == geo.ROOM_HJ_SHAFT then
      entered = true
      break
    end
  end
  if not entered then
    error("hj_room_to_shaft: " .. geo.fmt_state(session.state))
  end
  return geo.wait_ordinary_room(session, geo.ROOM_HJ_SHAFT, {
    settle_frames = 280,
    label = "hj_room_to_shaft",
  })
end

function M.play_hj_shaft_to_business(session)
  geo.require_room(session, geo.ROOM_HJ_SHAFT, "hj_shaft_to_business")
  geo.unmorph(session)
  geo.hold(session, 50, {}, "hj_return_bottom_land")

  geo.hold(session, 10, {}, "hj_return_jump_release")
  for frame = 0, 124 do
    local names = (frame < 18) and {"A"} or {"RIGHT", "A"}
    geo.hold(session, 1, names, "hj_return_first_jump")
  end
  geo.hold(session, 80, {}, "hj_return_first_land")

  geo.unmorph(session)
  geo.hold(session, 50, {}, "hj_return_shelf_stand")
  for _ = 1, 80 do
    local state = session.state
    if state.samus_x <= 82 then
      break
    end
    geo.hold(session, 1, {"LEFT"}, "hj_return_shelf_position")
  end
  geo.hold(session, 6, {"RIGHT"}, "hj_return_shelf_brake")
  geo.hold(session, 8, {}, "hj_return_shelf_release")
  for frame = 0, 129 do
    local names = (frame < 65) and {"A"} or {"LEFT", "A"}
    geo.hold(session, 1, names, "hj_return_second_jump")
  end
  geo.hold(session, 50, {}, "hj_return_second_land")

  geo.unmorph(session)
  geo.hold(session, 40, {}, "hj_return_slope_stand")
  local top_poses = geo.set(1, 2, 9, 10, 137, 138)
  for frame = 0, 109 do
    local names = (frame < 18) and {"A"} or {"RIGHT", "B", "A"}
    local state = geo.hold(session, 1, names, "hj_return_top_jump")
    if frame > 55 and state.samus_y <= 95 and top_poses[state.pose] then
      break
    end
  end
  geo.hold(session, 40, {}, "hj_return_top_land")

  geo.ensure_morph(session)
  local state
  local tunnel = false
  for frame = 0, 1099 do
    local names = (frame % 30 < 3) and {"RIGHT", "X"} or {"RIGHT"}
    state = geo.hold(session, 1, names, "hj_return_bomb_tunnel")
    if state.samus_x >= 350 then
      tunnel = true
      break
    end
  end
  if not tunnel then
    error("hj_shaft_to_business: tunnel stalled: " .. geo.fmt_state(session.state))
  end
  if (state.enemies_killed or 0) < 1 then
    for frame = 0, 499 do
      local names = (frame % 40 < 2) and {"RIGHT", "X"} or {"RIGHT"}
      state = geo.hold(session, 1, names, "hj_return_sova_cleanup")
      if (state.enemies_killed or 0) >= 1 then
        break
      end
    end
  end

  geo.hold(session, 80, {"RIGHT"}, "hj_return_gray_approach")
  geo.unmorph(session)
  geo.select_weapon(session, 0)
  local entered = false
  for frame = 0, 599 do
    local names = (frame < 4) and {"RIGHT", "B", "X"} or {"RIGHT", "B"}
    state = geo.hold(session, 1, names, "hj_return_gray_exit")
    if state.room_id == geo.ROOM_BUSINESS then
      entered = true
      break
    end
  end
  if not entered then
    error("hj_shaft_to_business: gray door failed: " .. geo.fmt_state(session.state))
  end
  state = geo.wait_ordinary_room(session, geo.ROOM_BUSINESS, {
    settle_frames = 180,
    label = "hj_shaft_to_business",
  })
  local floor_poses = geo.set(1, 2, 9, 10, 137, 138)
  for _ = 1, 60 do
    state = geo.hold(session, 1, {}, "hj_return_business_floor")
    if state.samus_y >= 1419 and floor_poses[state.pose] then
      break
    end
  end
  for _ = 1, 60 do
    state = session.state
    if state.samus_x >= 88 then
      break
    end
    geo.hold(session, 1, {"RIGHT"}, "hj_return_business_climb_anchor")
  end
  geo.hold(session, 4, {"LEFT"}, "hj_return_business_anchor_brake")
  geo.hold(session, 20, {}, "hj_return_business_anchor_settle")
  return session.state
end

function M.play_hijump_to_warehouse(session)
  M.play_hj_room_to_shaft(session)
  M.play_hj_shaft_to_business(session)
  return M.play_business_to_warehouse(session)
end

return M
