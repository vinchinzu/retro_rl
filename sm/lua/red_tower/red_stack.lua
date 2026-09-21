-- Red Tower descent → Bat → Below Spazer → Warehouse tunnels.
-- Port of snes/super_metroid/routes/kpdr/red_tower/red_stack.py.

local geo = require("red_tower.ctrl")
local west = require("red_tower.below_spazer_west")

local M = {}

M.play_below_spazer_floor_to_west = west.play_below_spazer_floor_to_west

local LAND_POSES = geo.set(1, 2, 25, 26, 27, 28, 137, 138)

function M.play_red_tower_to_bat(session)
  geo.require_room(session, geo.ROOM_RED_TOWER, "red_to_bat")
  geo.unmorph(session)

  local direction = "RIGHT"
  local state = session.state
  local descended = false
  for _ = 1, 1600 do
    state = session.state
    if state.samus_y >= 1600 then
      descended = true
      break
    end
    if state.samus_x >= 225 then
      direction = "LEFT"
    elseif state.samus_x <= 45 then
      direction = "RIGHT"
    end
    geo.hold(session, 1, {direction, "B"}, "red_upper_zigzag")
  end
  if not descended then
    error("red_to_bat: upper descent stalled: " .. geo.fmt_state(session.state))
  end
  geo.settle_hold(session, 40, "red_floor_settle")

  geo.ensure_morph(session)
  for _ = 1, 100 do
    state = session.state
    if state.samus_x >= 148 and state.samus_x <= 152 then
      break
    end
    if state.samus_x > 152 then
      direction = "LEFT"
    else
      direction = "RIGHT"
    end
    geo.hold(session, 1, {direction}, "red_floor_bomb_position")
  end
  geo.hold(session, 2, {"X"}, "red_floor_bomb")
  geo.hold(session, 8, {"LEFT"}, "red_floor_bomb_retreat")
  geo.hold(session, 40, {}, "red_floor_bomb_wait")
  local crossed = false
  for _ = 1, 180 do
    state = geo.hold(session, 1, {"RIGHT"}, "red_floor_cross")
    if state.samus_y >= 1650 then
      crossed = true
      break
    end
  end
  if not crossed then
    error("red_to_bat: timed floor crossing failed: " .. geo.fmt_state(session.state))
  end

  local tunnel = false
  for frame = 0, 499 do
    local names
    if frame % 45 < 2 then
      names = {"LEFT", "X"}
    else
      names = {"LEFT"}
    end
    state = geo.hold(session, 1, names, "red_tunnel_bomb_roll")
    if state.samus_y >= 1880 then
      tunnel = true
      break
    end
  end
  if not tunnel then
    error("red_to_bat: upper tunnel descent stalled: " .. geo.fmt_state(session.state))
  end

  local lower_entry = false
  for _ = 1, 300 do
    state = geo.hold(session, 1, {"RIGHT"}, "red_tunnel_right")
    if state.samus_y >= 2090 then
      lower_entry = true
      break
    end
  end
  if not lower_entry then
    error("red_to_bat: lower tunnel entry stalled: " .. geo.fmt_state(session.state))
  end

  direction = "LEFT"
  local lower = false
  for _ = 1, 600 do
    state = session.state
    if state.samus_y >= 2440 then
      lower = true
      break
    end
    if state.samus_x >= 220 then
      direction = "LEFT"
    elseif state.samus_x <= 40 then
      direction = "RIGHT"
    end
    geo.hold(session, 1, {direction}, "red_lower_zigzag")
  end
  if not lower then
    error("red_to_bat: lower descent stalled: " .. geo.fmt_state(session.state))
  end

  geo.settle_hold(session, 40, "red_bottom_settle")
  geo.unmorph(session)
  geo.select_weapon(session, 0)
  local to_bat = false
  for frame = 0, 419 do
    local phase = frame % 30
    local names
    if phase < 5 then
      names = {"RIGHT", "B", "X"}
    elseif phase >= 21 then
      names = {"RIGHT", "B", "A"}
    else
      names = {"RIGHT", "B"}
    end
    state = geo.hold(session, 1, names, "red_bottom_exit")
    if state.room_id == geo.ROOM_BAT then
      to_bat = true
      break
    end
  end
  if not to_bat then
    error("red_to_bat: Bat Room door not reached: " .. geo.fmt_state(session.state))
  end

  for frame = 0, 239 do
    state = geo.hold(session, 1, {"LEFT"}, "red_to_bat_brake")
    if state.room_id == geo.ROOM_BAT
        and state.game_state == 8
        and (state.door_transition or 0) == 0
        and frame > 8
        and state.samus_y <= 125 then
      return state
    end
    if state.room_id == geo.ROOM_BAT
        and state.game_state == 8
        and (state.door_transition or 0) == 0
        and frame > 40
        and (state.velocity_y or 0) == 0 then
      return state
    end
  end
  return geo.wait_ordinary_room(session, geo.ROOM_BAT, {
    settle_frames = 60,
    label = "red_to_bat",
  })
end

function M.play_bat_to_below_spazer(session)
  geo.require_room(session, geo.ROOM_BAT, "bat_to_below_spazer")
  geo.unmorph(session)
  geo.select_weapon(session, 0)

  local state = session.state
  if state.samus_y > 125 or math.abs(state.velocity_y or 0) > 0 then
    for _ = 1, 60 do
      state = geo.hold(session, 1, {}, "bat_land_wait")
      if (state.velocity_y or 0) == 0 and LAND_POSES[state.pose] then
        break
      end
    end
    geo.unmorph(session)
  end

  if session.state.samus_y <= 125 then
    geo.hold(session, 5, {}, "bat_entry_glide")
    geo.hold(session, 35, {"RIGHT", "B"}, "bat_first_runup")
    geo.hold(session, 60, {"RIGHT", "B", "A"}, "bat_first_jump")
    geo.settle_hold(session, 30, "bat_first_land")
  else
    geo.hold(session, 8, {}, "bat_low_ready")
    geo.hold(session, 15, {"RIGHT", "B"}, "bat_low_runup")
    geo.hold(session, 80, {"RIGHT", "B", "A"}, "bat_low_jump")
    geo.settle_hold(session, 40, "bat_low_land")
  end

  state = session.state
  if not (state.samus_x >= 210 and state.samus_y <= 165) then
    error("bat_to_below_spazer: missed first platform: " .. geo.fmt_state(state))
  end

  geo.hold(session, 8, {"RIGHT", "B"}, "bat_second_runup")
  geo.hold(session, 20, {"RIGHT", "B", "A"}, "bat_second_jump")
  geo.settle_hold(session, 80, "bat_second_land")
  state = session.state
  if not (state.samus_x >= 330 and state.samus_x <= 400 and state.samus_y <= 185) then
    error("bat_to_below_spazer: missed middle platform: " .. geo.fmt_state(state))
  end

  geo.hold(session, 48, {"RIGHT", "B", "A"}, "bat_third_jump")
  geo.settle_hold(session, 60, "bat_third_land")
  if session.state.samus_x < 400 then
    error("bat_to_below_spazer: missed right platform: " .. geo.fmt_state(session.state))
  end
  return geo.play_run_shoot_exit(session, {
    from_room = geo.ROOM_BAT,
    to_room = geo.ROOM_BELOW_SPAZER,
    direction = "RIGHT",
    label = "bat_to_below_spazer",
    run_frames = 20,
    shoot_frames = 4,
    spin_frames = 30,
    hold_frames = 240,
    settle_frames = 260,
  })
end

function M.play_below_spazer_to_west(session)
  geo.require_room(session, geo.ROOM_BELOW_SPAZER, "below_spazer_to_west")
  local detour = require("spazer.detour")
  return detour.play_spazer_detour(session)
end

function M.play_west_to_glass(session)
  return geo.play_run_shoot_exit(session, {
    from_room = geo.ROOM_WEST_TUNNEL,
    to_room = geo.ROOM_GLASS,
    direction = "RIGHT",
    label = "west_to_glass",
    run_frames = 80,
    shoot_frames = 5,
    spin_frames = 50,
    hold_frames = 300,
    settle_frames = 260,
  })
end

function M.play_glass_to_east(session)
  return geo.play_run_shoot_exit(session, {
    from_room = geo.ROOM_GLASS,
    to_room = geo.ROOM_EAST_TUNNEL,
    direction = "RIGHT",
    label = "glass_to_east",
    run_frames = 80,
    shoot_frames = 5,
    spin_frames = 50,
    hold_frames = 300,
    settle_frames = 260,
  })
end

function M.play_east_to_warehouse(session)
  geo.require_room(session, geo.ROOM_EAST_TUNNEL, "east_to_warehouse")
  local state
  local reached = false
  for frame = 0, 1599 do
    local phase = frame % 25
    local names
    if phase < 5 then
      names = {"RIGHT", "B", "X"}
    elseif phase >= 18 then
      names = {"RIGHT", "B", "A"}
    else
      names = {"RIGHT", "B"}
    end
    state = geo.hold(session, 1, names, "east_tunnel_right")
    if state.room_id == geo.ROOM_WAREHOUSE then
      reached = true
      break
    end
  end
  if not reached then
    error("east_to_warehouse: Warehouse not reached: " .. geo.fmt_state(session.state))
  end
  return geo.wait_ordinary_room(session, geo.ROOM_WAREHOUSE, {
    settle_frames = 900,
    label = "east_to_warehouse",
  })
end

function M.play_red_tower_to_warehouse(session)
  M.play_red_tower_to_bat(session)
  M.play_bat_to_below_spazer(session)
  M.play_below_spazer_to_west(session)
  M.play_west_to_glass(session)
  M.play_glass_to_east(session)
  return M.play_east_to_warehouse(session)
end

return M
