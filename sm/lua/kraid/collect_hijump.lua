-- Warehouse → Business → Hi-Jump shaft → Hi-Jump collect.
-- Port of snes/super_metroid/routes/kpdr/kraid/collect_hijump.py.

local geo = require("red_tower.ctrl")
local wh = require("kraid.warehouse_stack")

local M = {}

function M.play_business_to_hj_shaft(session)
  geo.require_room(session, geo.ROOM_BUSINESS, "business_to_hj_shaft")
  for _ = 1, 500 do
    local state = geo.hold(session, 1, {}, "business_incoming_elevator")
    if state.pose == 0 and state.samus_y >= 675 and state.samus_y <= 690 then
      break
    end
  end
  for _ = 1, 120 do
    local state = geo.hold(session, 1, {"RIGHT"}, "business_elevator_dismount")
    if state.pose ~= 0 and state.samus_x >= 145 then
      break
    end
  end
  geo.unmorph(session)

  local direction = "LEFT"
  local state = session.state
  local descended = false
  for frame = 0, 4199 do
    state = session.state
    if state.samus_y >= 1390 then
      descended = true
      break
    end
    if state.pose == 137 or state.pose == 138 then
      geo.unmorph(session)
    end
    if state.samus_x <= 45 then
      direction = "RIGHT"
    elseif state.samus_x >= 215 then
      direction = "LEFT"
    end
    local names
    if frame % 90 < 58 then
      names = {direction, "B"}
    else
      names = {direction, "B", "A"}
    end
    geo.hold(session, 1, names, "business_descend")
  end
  if not descended then
    error("business_to_hj_shaft: descent stalled: " .. geo.fmt_state(session.state))
  end
  geo.hold(session, 60, {}, "business_bottom_settle")

  for _ = 1, 320 do
    state = session.state
    if state.samus_x <= 70 then
      break
    end
    geo.hold(session, 1, {"LEFT", "B"}, "business_red_door_approach")
  end
  for _ = 1, 100 do
    state = session.state
    if state.samus_x >= 92 then
      break
    end
    geo.hold(session, 1, {"RIGHT"}, "business_red_door_standoff")
  end
  geo.hold(session, 5, {"LEFT"}, "business_red_door_brake")
  geo.hold(session, 20, {}, "business_red_door_settle")
  geo.select_weapon(session, 2)
  geo.hold(session, 3, {"LEFT"}, "business_face_red_door")
  geo.hold(session, 3, {}, "business_face_red_door_release")
  geo.hold(session, 2, {"LEFT", "X"}, "business_red_door_super")
  geo.hold(session, 80, {}, "business_red_door_fuse")
  local entered = false
  for _ = 1, 500 do
    state = geo.hold(session, 1, {"LEFT", "B", "A"}, "business_enter_hj_shaft")
    if state.room_id == geo.ROOM_HJ_SHAFT then
      entered = true
      break
    end
  end
  if not entered then
    error("business_to_hj_shaft: red door failed: " .. geo.fmt_state(session.state))
  end
  return geo.wait_ordinary_room(session, geo.ROOM_HJ_SHAFT, {
    settle_frames = 280,
    label = "business_to_hj_shaft",
  })
end

function M.play_hj_shaft_to_hj_room(session)
  geo.require_room(session, geo.ROOM_HJ_SHAFT, "hj_shaft_to_hj")
  geo.unmorph(session)
  geo.select_weapon(session, 0)

  for _ = 1, 220 do
    local state = session.state
    if state.samus_x <= 390 then
      break
    end
    geo.hold(session, 1, {"LEFT", "B"}, "hj_shaft_etank_approach")
  end
  geo.hold(session, 480, {}, "hj_shaft_etank_fanfare")
  geo.ensure_morph(session)
  for _ = 1, 120 do
    local state = geo.hold(session, 1, {"RIGHT"}, "hj_shaft_etank_backoff")
    if state.samus_x >= 470 then
      break
    end
  end
  geo.unmorph(session)
  geo.hold(session, 20, {}, "hj_shaft_etank_stand")
  geo.hold(session, 20, {"LEFT", "B"}, "hj_shaft_etank_runup")
  for _ = 1, 140 do
    local state = geo.hold(session, 1, {"LEFT", "B", "A"}, "hj_shaft_etank_jump")
    if state.samus_x <= 310 and state.samus_y >= 180 then
      break
    end
  end
  geo.hold(session, 30, {}, "hj_shaft_etank_jump_land")
  for _ = 1, 160 do
    local state = session.state
    if state.samus_x <= 310 and state.samus_y >= 180 then
      break
    end
    geo.hold(session, 1, {"LEFT", "B"}, "hj_shaft_low_tunnel")
  end
  geo.ensure_morph(session)
  local tunnel = false
  local state
  for _ = 1, 700 do
    state = geo.hold(session, 1, {"LEFT"}, "hj_shaft_morph_left")
    if state.samus_x <= 40 and state.samus_y >= 450 then
      tunnel = true
      break
    end
  end
  if not tunnel then
    error("hj_shaft_to_hj: lower tunnel stalled: " .. geo.fmt_state(session.state))
  end

  geo.unmorph(session)
  geo.select_weapon(session, 0)
  geo.hold(session, 12, {}, "hj_shaft_door_release")
  for _ = 1, 80 do
    state = geo.hold(session, 1, {"A"}, "hj_shaft_door_jump")
    if state.samus_y <= 390 then
      break
    end
  end
  geo.hold(session, 2, {"LEFT", "A", "X"}, "hj_shaft_blue_door_shot")
  local entered = false
  for _ = 1, 420 do
    state = geo.hold(session, 1, {"LEFT", "A"}, "hj_shaft_enter_hj")
    if state.room_id == geo.ROOM_HJ then
      entered = true
      break
    end
  end
  if not entered then
    error("hj_shaft_to_hj: blue door failed: " .. geo.fmt_state(session.state))
  end
  return geo.wait_ordinary_room(session, geo.ROOM_HJ, {
    settle_frames = 260,
    label = "hj_shaft_to_hj",
  })
end

local function pillar_downshot_burst(session, reason, frames)
  frames = frames or 25
  for i = 0, frames - 1 do
    if i % 3 == 0 then
      geo.hold(session, 1, {"DOWN", "X"}, reason)
    else
      geo.hold(session, 1, {"DOWN"}, reason .. "_aim")
    end
  end
end

function M.play_hj_room_collect(session)
  geo.require_room(session, geo.ROOM_HJ, "hj_room_collect")
  geo.unmorph(session)
  geo.select_weapon(session, 0)
  geo.hold(session, 20, {}, "hj_room_entry_settle")

  geo.hold(session, 12, {"LEFT", "B"}, "hj_room_first_runup")
  for _ = 1, 70 do
    local state = geo.hold(session, 1, {"LEFT", "B", "A"}, "hj_room_first_jump")
    if state.samus_y <= 52 then
      break
    end
  end
  pillar_downshot_burst(session, "hj_room_first_downshot")
  geo.hold(session, 50, {}, "hj_room_first_land")

  geo.hold(session, 2, {"RIGHT"}, "hj_room_face_right")
  geo.hold(session, 10, {}, "hj_room_face_right_settle")
  for _ = 1, 80 do
    local state = geo.hold(session, 1, {"A"}, "hj_room_second_jump")
    if state.samus_y <= 53 then
      break
    end
  end
  pillar_downshot_burst(session, "hj_room_second_downshot")
  geo.hold(session, 50, {}, "hj_room_second_land")

  geo.hold(session, 12, {"RIGHT", "B"}, "hj_room_cross_backoff")
  geo.hold(session, 15, {"LEFT", "B"}, "hj_room_cross_runup")
  for _ = 1, 120 do
    local state = geo.hold(session, 1, {"LEFT", "B", "A"}, "hj_room_cross_pillar")
    if state.samus_x < 120 then
      break
    end
  end
  geo.hold(session, 80, {}, "hj_room_left_land")

  for _ = 1, 80 do
    local state = session.state
    if state.samus_x >= 115 then
      break
    end
    geo.hold(session, 1, {"RIGHT"}, "hj_room_statue_approach")
  end
  geo.hold(session, 12, {"LEFT"}, "hj_room_statue_brake")
  geo.hold(session, 8, {}, "hj_room_statue_settle")
  geo.hold(session, 3, {"LEFT"}, "hj_room_statue_face")
  geo.hold(session, 3, {}, "hj_room_statue_face_release")
  geo.hold(session, 1, {"X"}, "hj_room_statue_shot")
  geo.hold(session, 60, {}, "hj_room_statue_open")
  local collected = false
  local state
  for _ = 1, 180 do
    state = geo.hold(session, 1, {"LEFT"}, "hj_room_collect_item")
    if geo.has_hi_jump(state) then
      collected = true
      break
    end
  end
  if not collected then
    error("hj_room_collect: Hi-Jump PLM not collected: " .. geo.fmt_state(session.state))
  end
  geo.hold(session, 480, {}, "hj_room_item_fanfare")
  return session.state
end

function M.play_warehouse_to_hijump(session)
  wh.play_warehouse_to_business(session)
  M.play_business_to_hj_shaft(session)
  M.play_hj_shaft_to_hj_room(session)
  return M.play_hj_room_collect(session)
end

return M
