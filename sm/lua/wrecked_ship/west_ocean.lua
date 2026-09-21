-- West Ocean shinespark controllers (K6). Over-ocean spark → green Super WS.

local rooms = require("rooms")
local ctrl = require("wrecked_ship.ctrl")
local spark = require("skills.shinespark")

local M = {}
M.ROOM_WEST_OCEAN = rooms.ROOM_WEST_OCEAN or 0x93FE
M.ROOM_BOWLING = rooms.ROOM_BOWLING or 0xC98E
M.ROOM_WS_ENTRANCE = rooms.ROOM_WS_ENTRANCE or 0xCA08
M.DEFAULT_OCEAN_HOP_FRAMES = 4
M.DEFAULT_OCEAN_PRE_STAND = 4
M.DEFAULT_OCEAN_SPARK_TRAVEL = 500
M.DEFAULT_SUPER_DOOR_BUDGET = 280
M.DEFAULT_BACK_FRAMES = 8
M.DEFAULT_HOP_FRAMES = 4
M.DEFAULT_EDGE_BUDGET = 80
M.DEFAULT_SPARK_TRAVEL = 400
M.DEFAULT_DOOR_BUDGET = 200

function M.run_to_water_edge(session, opts)
  opts = opts or {}
  local budget = opts.budget or M.DEFAULT_EDGE_BUDGET
  local y_slop = opts.y_slop or 24
  local label = opts.label or "wo_edge"
  local y0 = ctrl.y(session.state)
  local edge_x = ctrl.x(session.state)
  local i
  for i = 0, budget - 1 do
    local prev = ctrl.x(session.state)
    ctrl.hold(session, 1, {"RIGHT", "B"}, label .. "_" .. i)
    local st = session.state
    if ctrl.y(st) > y0 + y_slop then
      break
    end
    if i > 4 and ctrl.x(st) <= prev then
      edge_x = ctrl.x(st)
      break
    end
    edge_x = ctrl.x(st)
  end
  return edge_x
end

function M.open_green_super_ws(session, opts)
  opts = opts or {}
  local budget = opts.budget or M.DEFAULT_SUPER_DOOR_BUDGET
  local settle_frames = opts.settle_frames or 120
  local label = opts.label or "wo_super_ws"
  ctrl.require_room(session, M.ROOM_WEST_OCEAN, label)
  pcall(ctrl.select_weapon, session, 2)
  local i
  for i = 0, budget - 1 do
    local st = session.state
    if ctrl.num(st.room_id) == M.ROOM_WS_ENTRANCE
        and ctrl.num(st.door_transition) == 0
        and ctrl.num(st.game_state) == 8 then
      ctrl.hold(session, 8, {}, label .. "_settle")
      return session.state
    end
    if ctrl.num(st.room_id) == M.ROOM_WS_ENTRANCE then
      ctrl.hold(session, 1, {}, label .. "_trans")
    else
      local phase = i % 24
      if phase < 8 then
        ctrl.hold(session, 1, {"RIGHT", "X"}, label .. "_sup")
      elseif phase < 14 then
        ctrl.hold(session, 1, {"RIGHT"}, label .. "_face")
      else
        ctrl.hold(session, 1, {"RIGHT", "B"}, label .. "_run")
      end
    end
  end
  for i = 1, settle_frames do
    local st = session.state
    if ctrl.num(st.room_id) == M.ROOM_WS_ENTRANCE
        and ctrl.num(st.door_transition) == 0
        and ctrl.num(st.game_state) == 8 then
      return st
    end
    ctrl.hold(session, 1, {}, label .. "_final")
  end
  local st = session.state
  if ctrl.num(st.room_id) == M.ROOM_WS_ENTRANCE then
    return st
  end
  ctrl.timeout(label .. ": green Super did not open into 0xCA08 " .. ctrl.brief(st))
end

function M.play_west_ocean_over_ocean_spark(session, opts)
  opts = opts or {}
  local hop_frames = opts.hop_frames or M.DEFAULT_OCEAN_HOP_FRAMES
  local pre_stand = opts.pre_stand_frames or M.DEFAULT_OCEAN_PRE_STAND
  local spark_travel = opts.spark_travel or M.DEFAULT_OCEAN_SPARK_TRAVEL
  local super_budget = opts.super_budget or M.DEFAULT_SUPER_DOOR_BUDGET
  local label = opts.label or "wo_over_ocean"
  local charge_mode = opts.charge_mode or "stutter"
  ctrl.require_room(session, M.ROOM_WEST_OCEAN, label)
  local charge = spark.charge_until_boost(session, "RIGHT", {
    budget = 300, label = label .. "_charge", mode = charge_mode,
  })
  if type(charge) == "table" and charge.ok == false then
    ctrl.timeout(label .. ": charge failed")
  end
  local store = spark.crouch_store(session, {label = label .. "_store"})
  if type(store) == "table" and store.ok == false then
    ctrl.timeout(label .. ": store failed")
  end
  ctrl.hold(session, 2, {}, label .. "_pre_hop")
  if hop_frames > 0 then
    ctrl.hold(session, hop_frames, {"A"}, label .. "_hop")
  end
  local act = spark.activate_shinespark(session, "RIGHT", {
    pre_stand_frames = pre_stand,
    pre_stand_buttons = {"UP"},
    hold_frames = 30,
    travel_budget = spark_travel,
    label = label .. "_spark",
  })
  if type(act) == "table" and act.spark_pose_seen == false then
    ctrl.timeout(label .. ": spark did not arm/travel")
  end
  if ctrl.num(session.state.room_id) == M.ROOM_WS_ENTRANCE then
    ctrl.hold(session, 8, {}, label .. "_spark_enter")
    return session.state
  end
  return M.open_green_super_ws(session, {budget = super_budget, label = label .. "_super"})
end

function M.play_west_ocean_to_ws(session, opts)
  opts = opts or {}
  opts.label = opts.label or "west_ocean_to_ws"
  return M.play_west_ocean_over_ocean_spark(session, opts)
end

return M
