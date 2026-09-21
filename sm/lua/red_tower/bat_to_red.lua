-- Bat Room → Red Tower LEFT across dry pipe platforms (K5 hop 11).

local rooms = require("rooms")
local ctrl = require("red_tower.ctrl")
local geom = require("red_tower.geometry")

local M = {}
local ROOM_BAT = rooms.ROOM_BAT or 0xA3DD
local ROOM_RED_TOWER = rooms.ROOM_RED_TOWER or 0xA253
local MORPH = {
  [29]=true,[30]=true,[31]=true,[32]=true,[37]=true,[38]=true,
  [39]=true,[40]=true,[41]=true,[42]=true,[43]=true,[49]=true,[50]=true,[65]=true,[66]=true,
}
local CROUCH = {[11]=true,[12]=true}
local KB = {[83]=true,[84]=true,[109]=true,[137]=true,[138]=true,[143]=true,[158]=true,[159]=true,[160]=true}

local function in_water(state)
  return ctrl.y(state) > geom.BAT_TO_RED_HIGH_Y
end

local function on_left_seat(state)
  return ctrl.num(state.room_id) == ROOM_BAT
    and ctrl.x(state) <= geom.BAT_TO_RED_DOOR_SEAT_X
    and ctrl.y(state) <= geom.BAT_TO_RED_HIGH_Y
end

local function stand_up(session)
  local pose = ctrl.num(session.state.pose)
  if not CROUCH[pose] and not MORPH[pose] then
    return
  end
  ctrl.hold(session, 8, {"UP"}, "bat_to_red_stand")
  pose = ctrl.num(session.state.pose)
  if CROUCH[pose] or MORPH[pose] then
    ctrl.hold(session, 8, {"UP"}, "bat_to_red_stand")
  end
end

local function water_dir(state)
  if ctrl.x(state) < 40 then
    return "RIGHT"
  end
  return "LEFT"
end

local function climb_water(session)
  local x0 = ctrl.x(session.state)
  local direction = water_dir(session.state)
  if MORPH[ctrl.num(session.state.pose)] then
    local i
    for i = 1, 40 do
      ctrl.hold(session, 1, {direction}, "bat_to_red_water_roll")
      if ctrl.num(session.state.room_id) ~= ROOM_BAT then
        return
      end
      if not MORPH[ctrl.num(session.state.pose)] then
        break
      end
      if ctrl.x(session.state) < x0 - 24 then
        break
      end
    end
    stand_up(session)
    if MORPH[ctrl.num(session.state.pose)] or not in_water(session.state) then
      return
    end
  else
    stand_up(session)
  end
  direction = water_dir(session.state)
  ctrl.hold(session, geom.BAT_TO_RED_WATER_CJ_CROUCH, {"DOWN"}, "bat_to_red_water_cj")
  ctrl.hold(session, geom.BAT_TO_RED_WATER_CJ_JUMP, {direction, "A"}, "bat_to_red_water_cj")
  ctrl.hold(session, geom.BAT_TO_RED_WATER_GRAB, {direction, "DOWN", "A"}, "bat_to_red_water_grab")
end

local function clear_obstacle(session, label)
  if in_water(session.state) then
    climb_water(session)
    return
  end
  local direction = "LEFT"
  if ctrl.x(session.state) <= 40 then
    direction = "RIGHT"
  end
  local i
  for i = 1, 18 do
    ctrl.hold(session, 1, {direction, "A"}, label .. "_jump")
  end
  for i = 0, 33 do
    local buttons = {direction}
    if (i % 3) == 0 then
      buttons[2] = "X"
    end
    ctrl.hold(session, 1, buttons, label .. "_aim_shoot")
  end
  ctrl.hold(session, 10, {}, label .. "_land")
end

local function break_kb(session)
  ctrl.hold(session, 12, {"A"}, "bat_to_red_kb_a")
  local i
  for i = 1, 24 do
    local state = ctrl.hold(session, 1, {}, "bat_to_red_kb_land")
    if not KB[ctrl.num(state.pose)] then
      return
    end
  end
  if KB[ctrl.num(session.state.pose)] then
    ctrl.escape_kb(session, {
      prefer_dir = "LEFT",
      run_frames = 3,
      spin_frames = 12,
      label = "bat_to_red",
      stop_room_id = ROOM_RED_TOWER,
    })
  end
end

local function seat_red_bottom(session)
  local i
  for i = 1, 120 do
    local st = session.state
    if ctrl.num(st.room_id) ~= ROOM_RED_TOWER then
      return st
    end
    local x, y = ctrl.x(st), ctrl.y(st)
    local grounded = math.abs(ctrl.num(st.velocity_y)) == 0 and y >= 2430
    if x >= 222 then
      ctrl.hold(session, 1, {"LEFT"}, "bat_to_red_bat_door")
    elseif not grounded then
      ctrl.hold(session, 1, {}, "bat_to_red_land")
    elseif x > 216 then
      ctrl.hold(session, 1, {"LEFT"}, "bat_to_red_seat_l")
    elseif x < 216 then
      ctrl.hold(session, 1, {"RIGHT"}, "bat_to_red_seat_r")
    else
      return st
    end
  end
  ctrl.hold(session, 4, {"LEFT"}, "bat_to_red_pose10")
  return session.state
end

local function exit_to_red(session)
  ctrl.play_run_shoot_exit(session, {
    from_room = ROOM_BAT,
    to_room = ROOM_RED_TOWER,
    direction = "LEFT",
    label = "bat_to_red",
    run_frames = geom.BAT_TO_RED_EXIT_RUN,
    shoot_frames = geom.BAT_TO_RED_EXIT_SHOOT,
    spin_frames = geom.BAT_TO_RED_EXIT_SPIN,
    hold_frames = geom.BAT_TO_RED_EXIT_HOLD,
    settle_frames = 40,
  })
  return seat_red_bottom(session)
end

function M.play_bat_to_red(session)
  ctrl.require_room(session, ROOM_BAT, "bat_to_red")
  stand_up(session)
  pcall(ctrl.select_weapon, session, 0)
  local best_x = ctrl.x(session.state)
  local stale = 0
  local deadline = ctrl.num(session.frame) + geom.BAT_TO_RED_TRAVERSE_BUDGET
  local frame = 0
  while ctrl.num(session.frame) < deadline do
    local state = session.state
    if ctrl.num(state.room_id) == ROOM_RED_TOWER then
      return seat_red_bottom(session)
    end
    if ctrl.num(state.room_id) ~= ROOM_BAT then
      ctrl.timeout("bat_to_red: unexpected room " .. ctrl.brief(state))
    end
    if KB[ctrl.num(state.pose)] then
      break_kb(session)
    elseif on_left_seat(state) then
      if math.abs(ctrl.num(state.velocity_y)) > 0 then
        ctrl.hold(session, 1, {"LEFT", "B", "X"}, "bat_to_red_door_air")
      else
        local ok, err = pcall(exit_to_red, session)
        if not ok then
          break_kb(session)
        else
          return err
        end
      end
    elseif in_water(state) then
      climb_water(session)
      stale = 0
      best_x = ctrl.x(session.state)
    elseif MORPH[ctrl.num(state.pose)] or CROUCH[ctrl.num(state.pose)] then
      stand_up(session)
    else
      local x = ctrl.x(state)
      if x < best_x - 2 then
        best_x = x
        stale = 0
      else
        stale = stale + 1
      end
      if stale >= geom.BAT_TO_RED_PROGRESS_WINDOW then
        clear_obstacle(session, "bat_to_red_stall")
        stale = 0
        best_x = ctrl.x(session.state)
      else
        local buttons = {"LEFT", "B", "X"}
        if frame >= geom.BAT_TO_RED_RUNUP
            and (frame % geom.BAT_TO_RED_JUMP_PERIOD) < geom.BAT_TO_RED_JUMP_HOLD then
          buttons[4] = "A"
        end
        ctrl.hold(session, 1, buttons, "bat_to_red_traverse")
        frame = frame + 1
      end
    end
  end
  ctrl.timeout("bat_to_red: traverse timeout: " .. ctrl.brief(session.state))
end

return M
