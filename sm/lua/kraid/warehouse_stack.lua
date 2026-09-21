-- Warehouse Entrance wall stack and elevator to Business Center.
-- Port of snes/super_metroid/routes/kpdr/kraid/warehouse_stack.py.

local geo = require("red_tower.ctrl")

local M = {}

local function open_warehouse_stack(session, face, label)
  geo.select_weapon(session, 2)
  geo.hold(session, 6, {face}, label .. "_face")
  geo.hold(session, 8, {"DOWN"}, label .. "_crouch")
  geo.hold(session, 1, {"X"}, label .. "_bottom_super")
  geo.hold(session, 30, {}, label .. "_bottom_open")
  geo.hold(session, 5, {"UP"}, label .. "_stand")
  geo.settle_hold(session, 4, label .. "_stand_settle")
  geo.hold(session, 1, {"X"}, label .. "_middle_super")
  geo.hold(session, 30, {}, label .. "_middle_open")
  geo.hold(session, 5, {"A"}, label .. "_tiny_hop")
  geo.hold(session, 1, {face, "X"}, label .. "_top_super")
  local wait = (face == "LEFT") and 30 or 24
  geo.hold(session, wait, {}, label .. "_top_open")
end

local function play_warehouse_reverse_stack(session)
  local label = "warehouse_reverse"
  if not geo.has_hi_jump(session.state) then
    error(label .. ": right-ledge return requires Hi-Jump")
  end

  for _ = 1, 120 do
    geo.hold(session, 1, {"LEFT", "B", "A"}, label .. "_drop_left")
  end
  local lip = false
  for _ = 1, 100 do
    local state = geo.hold(session, 1, {"RIGHT"}, label .. "_lower_lip")
    if state.samus_x >= 445 and state.samus_x <= 510
        and state.samus_y >= 300 and state.samus_y <= 320
        and (state.velocity_y or 0) == 0 then
      lip = true
      break
    end
  end
  if not lip then
    error(label .. ": lower lip missed: " .. geo.fmt_state(session.state))
  end

  open_warehouse_stack(session, "LEFT", label .. "_lower_stack")
  for _ = 1, 180 do
    geo.hold(session, 1, {"LEFT", "B", "A"}, label .. "_lower_cross")
  end
  geo.hold(session, 8, {"UP"}, label .. "_mid_stand")
  geo.settle_hold(session, 20, label .. "_mid_settle")
  for _ = 1, 105 do
    geo.hold(session, 1, {"LEFT", "B", "A"}, label .. "_mid_climb")
  end
  geo.hold(session, 8, {"UP"}, label .. "_upper_stand")
  geo.settle_hold(session, 20, label .. "_upper_settle")
  for _ = 1, 105 do
    geo.hold(session, 1, {"LEFT", "B", "A"}, label .. "_upper_climb")
  end

  open_warehouse_stack(session, "LEFT", label .. "_upper_stack")
  geo.hold(session, 8, {"UP"}, label .. "_exit_stand")
  geo.settle_hold(session, 20, label .. "_exit_settle")
  local elev = false
  for _ = 1, 180 do
    local state = geo.hold(session, 1, {"LEFT", "B", "A"}, label .. "_exit")
    if state.samus_x <= 40 and state.samus_y <= 150 then
      elev = true
      break
    end
  end
  if not elev then
    error(label .. ": left elevator platform missed: " .. geo.fmt_state(session.state))
  end
  return session.state
end

function M.play_warehouse_wall_to_lower_lip(session)
  geo.require_room(session, geo.ROOM_WAREHOUSE, "warehouse_wall")
  geo.unmorph(session)
  for _ = 1, 160 do
    local state = geo.hold(session, 1, {"RIGHT", "B"}, "warehouse_wall_runup")
    if state.samus_x >= 75 then
      break
    end
  end
  geo.hold(session, 30, {}, "warehouse_super_cooldown")
  open_warehouse_stack(session, "RIGHT", "warehouse_wall")

  local reached = false
  local state
  for _ = 1, 500 do
    state = geo.hold(session, 1, {"RIGHT", "B", "A"}, "warehouse_cross_stack")
    if state.samus_x >= 500 and state.samus_y >= 300 then
      reached = true
      break
    end
  end
  if not reached then
    error("warehouse_wall: lower lip not reached: " .. geo.fmt_state(session.state))
  end
  geo.settle_hold(session, 30, "warehouse_lower_lip_settle")
  state = session.state
  if state.samus_x < 500 or state.samus_y < 300 then
    error("warehouse_wall: unstable lower lip: " .. geo.fmt_state(state))
  end
  return state
end

function M.resolve_warehouse_entry_mode(state, entry_mode)
  entry_mode = entry_mode or "auto"
  if entry_mode == "auto" then
    if state.samus_x > 400 then
      return "right_reverse_stack"
    end
    return "left_elevator"
  end
  return entry_mode
end

function M.play_warehouse_to_business(session, entry_mode)
  entry_mode = entry_mode or "auto"
  geo.require_room(session, geo.ROOM_WAREHOUSE, "warehouse_to_business")
  geo.unmorph(session)
  local mode = M.resolve_warehouse_entry_mode(session.state, entry_mode)
  if mode == "right_reverse_stack" then
    play_warehouse_reverse_stack(session)
  end
  for _ = 1, 180 do
    local state = session.state
    if state.samus_x >= 126 then
      break
    end
    geo.hold(session, 1, {"RIGHT"}, "warehouse_elevator_position")
  end
  geo.hold(session, 5, {"LEFT"}, "warehouse_elevator_brake")
  geo.settle_hold(session, 20, "warehouse_elevator_settle")
  local reached = false
  local state
  for _ = 1, 700 do
    state = geo.hold(session, 1, {"DOWN"}, "warehouse_elevator_down")
    if state.room_id == geo.ROOM_BUSINESS then
      reached = true
      break
    end
  end
  if not reached then
    error("warehouse_to_business: " .. geo.fmt_state(session.state))
  end
  return geo.wait_ordinary_room(session, geo.ROOM_BUSINESS, {
    settle_frames = 320,
    label = "warehouse_to_business",
  })
end

return M
