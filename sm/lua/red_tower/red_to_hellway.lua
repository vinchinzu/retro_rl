-- Red Tower → Hellway (K5 hop 12). Ice-pin climb, then hop-tape fallback.

local rooms = require("rooms")
local ctrl = require("red_tower.ctrl")
local geom = require("red_tower.geometry")

local M = {}
local ROOM_RED_TOWER = rooms.ROOM_RED_TOWER or 0xA253
local ROOM_HELLWAY = rooms.ROOM_HELLWAY or 0xA2F7
local MORPH = {
  [0x1D]=true,[0x1E]=true,[0x1F]=true,[0x20]=true,
  [29]=true,[30]=true,[39]=true,[40]=true,[41]=true,[42]=true,[81]=true,[82]=true,
}
local TRUE_MORPH = {[29]=true,[30]=true,[31]=true,[32]=true}

local function in_red(state)
  return ctrl.num(state.room_id) == ROOM_RED_TOWER
end
local function in_hellway(state)
  return ctrl.num(state.room_id) == ROOM_HELLWAY
end

local ascent = require("red_tower.tapes.red_to_hellway_ascent")
local function slice_rle(runs, n_frames)
  local out, used, i = {}, 0, 1
  for i = 1, #runs do
    if used >= n_frames then
      break
    end
    local take = math.min(runs[i][1], n_frames - used)
    if take > 0 then
      out[#out + 1] = {take, runs[i][2]}
      used = used + take
    end
  end
  return out
end
M.HUMAN_FLOOR_RLE = slice_rle(ascent, 850)

function M.play_upper_rle(session, runs, label)
  return ctrl.play_script(session, runs, {
    reason = label,
    room_id = ROOM_RED_TOWER,
    stop_when = function(st)
      return in_hellway(st) or not in_red(st)
    end,
  })
end

function M.ibj_double(session, label, center_x, stop_y)
  center_x = center_x or 150
  stop_y = stop_y or (geom.RED_FLOOR_Y - 40)
  if not in_red(session.state) then
    return session.state
  end
  if ctrl.is_knockback(session.state) then
    ctrl.escape_kb(session, {prefer_dir = "LEFT", run_frames = 2, spin_frames = 8, label = label, break_on_motion_clear = true})
  end
  if not ctrl.is_morph(session.state.pose) and not MORPH[ctrl.num(session.state.pose)] then
    pcall(ctrl.ensure_morph, session)
  end
  local st = session.state
  local x, y = ctrl.x(st), ctrl.y(st)
  if y >= geom.RED_BOTTOM_Y - 120 and x >= 200 then
    ctrl.hold(session, 4, {"LEFT"}, label .. "_bat_bias")
  elseif x > center_x + 18 then
    ctrl.hold(session, 2, {"LEFT"}, label .. "_cL")
  elseif x < center_x - 18 then
    ctrl.hold(session, 2, {"RIGHT"}, label .. "_cR")
  end
  ctrl.hold(session, 2, {"X"}, label .. "_b1")
  local i
  for i = 1, 18 do
    st = ctrl.hold(session, 1, {}, label .. "_w1")
    if not in_red(st) or ctrl.y(st) <= stop_y then
      return st
    end
    if ctrl.y(st) >= geom.RED_BOTTOM_Y - 100 and ctrl.x(st) >= 200 then
      ctrl.hold(session, 1, {"LEFT"}, label .. "_bat_w1")
    end
  end
  ctrl.hold(session, 2, {"X"}, label .. "_b2")
  for i = 1, 30 do
    st = ctrl.hold(session, 1, {}, label .. "_w2")
    if not in_red(st) or ctrl.y(st) <= stop_y then
      return st
    end
    if ctrl.y(st) >= geom.RED_BOTTOM_Y - 100 and ctrl.x(st) >= 200 then
      ctrl.hold(session, 1, {"LEFT"}, label .. "_bat_w2")
    end
  end
  return session.state
end

function M.tunnel_to_midplat(session, label)
  if not in_red(session.state) then
    return session.state
  end
  local i
  for i = 1, 12 do
    if not (ctrl.is_morph(session.state.pose) or MORPH[ctrl.num(session.state.pose)]) then
      break
    end
    ctrl.hold(session, 1, {"UP"}, label .. "_unmorph")
  end
  for i = 1, 16 do
    ctrl.hold(session, 1, {"UP"}, label .. "_stand")
  end
  ctrl.hold(session, 5, {}, label .. "_tun_s")
  for i = 1, 35 do
    local st = session.state
    if not in_red(st) then
      return st
    end
    if ctrl.y(st) >= geom.RED_TUNNEL_Y then
      break
    end
    ctrl.hold(session, 1, {"LEFT"}, label .. "_to_tun")
  end
  for i = 1, 40 do
    if not in_red(session.state) or ctrl.num(session.state.velocity_y) == 0 then
      break
    end
    ctrl.hold(session, 1, {}, label .. "_tun_land")
  end
  for i = 1, 50 do
    local st = session.state
    if not in_red(st) then
      return st
    end
    if math.abs(ctrl.x(st) - 104) < 6 and ctrl.num(st.velocity_y) == 0 then
      break
    end
    local dir = (ctrl.x(st) > 104) and "LEFT" or "RIGHT"
    ctrl.hold(session, 1, {dir}, label .. "_tun_x")
  end
  ctrl.hold(session, 5, {}, label .. "_tun_seat")
  for i = 1, 8 do
    ctrl.hold(session, 2, {"UP", "X"}, label .. "_tun_shot")
  end
  for i = 0, 49 do
    local st = session.state
    if not in_red(st) or ctrl.y(st) <= geom.RED_FLOOR_Y + 80 then
      return st
    end
    if i < 15 then
      ctrl.hold(session, 1, {"A"}, label .. "_tun_j")
    elseif i < 25 then
      ctrl.hold(session, 1, {"A", "X"}, label .. "_tun_jx")
    else
      ctrl.hold(session, 1, {"RIGHT", "A", "X"}, label .. "_tun_jrx")
    end
  end
  for i = 1, 35 do
    if not in_red(session.state) or ctrl.num(session.state.velocity_y) == 0 then
      break
    end
    ctrl.hold(session, 1, {}, label .. "_mid_land")
  end
  return session.state
end

function M.seat_left_after_handoff(session, label)
  if not in_red(session.state) then
    return session.state
  end
  local i
  for i = 1, 90 do
    local st = session.state
    if not in_red(st) then
      return st
    end
    if ctrl.num(st.velocity_y) == 0 and ctrl.x(st) <= 50
        and 1480 <= ctrl.y(st) and ctrl.y(st) <= 1520 then
      break
    end
    if TRUE_MORPH[ctrl.num(st.pose)] then
      ctrl.hold(session, 1, {"UP"}, label .. "_u")
    else
      ctrl.hold(session, 1, {"LEFT", "B", "A"}, label .. "_seat_spin")
    end
  end
  ctrl.hold(session, 8, {}, label .. "_seat_s")
  for i = 1, 20 do
    local st = session.state
    if not in_red(st) then
      return st
    end
    local pose = ctrl.num(st.pose)
    if pose == 1 or pose == 2 then
      break
    end
    if TRUE_MORPH[pose] or pose == 137 or pose == 138 or pose == 9 or pose == 10 then
      ctrl.hold(session, 1, {"UP"}, label .. "_stand")
    else
      break
    end
  end
  ctrl.hold(session, 4, {}, label .. "_seat_s2")
  return session.state
end

function M.period_wj(session, label, side, frames, stop_y, period, into, flip)
  period = period or 16
  into = into or 6
  flip = flip or 8
  local opp = (side == "LEFT") and "RIGHT" or "LEFT"
  local i
  for i = 0, frames - 1 do
    local st = session.state
    if in_hellway(st) or not in_red(st) then
      return st
    end
    local y = ctrl.y(st)
    if stop_y and y <= stop_y then
      return st
    end
    if y <= geom.RED_TOP_DOOR_Y + 25 or y >= geom.RED_BOTTOM_Y - 80 then
      return st
    end
    if TRUE_MORPH[ctrl.num(st.pose)] then
      ctrl.hold(session, 1, {"UP"}, label .. "_u")
    else
      local ph = i % period
      if ph < into then
        ctrl.hold(session, 1, {side, "A"}, label .. "_into")
      elseif ph < into + flip then
        ctrl.hold(session, 1, {opp, "A"}, label .. "_flip")
      else
        ctrl.hold(session, 1, {opp, "B", "A"}, label .. "_spin")
      end
    end
  end
  return session.state
end

function M.play_red_to_hellway(session)
  local label = "red_to_hellway"
  ctrl.require_room(session, ROOM_RED_TOWER, label)
  local ice = require("red_tower.red_ice_climb")
  if ice.can_attach_bottom_edge(session.state) then
    return ice.play_ice_climb_to_hellway(session)
  end
  local hop = require("red_tower.tapes.red_to_hellway_hop")
  ctrl.play_script(session, hop, {
    reason = label,
    room_id = ROOM_RED_TOWER,
    stop_when = function(s)
      return ctrl.num(s.room_id) ~= ROOM_RED_TOWER
    end,
  })
  if in_hellway(session.state) then
    return ctrl.wait_ordinary_room(session, ROOM_HELLWAY, {
      settle_frames = geom.RED_TO_HELLWAY_EXIT_SETTLE,
      label = label,
    })
  end
  if not in_red(session.state) then
    ctrl.timeout(label .. ": hop body left Red to " .. ctrl.brief(session.state))
  end
  ctrl.unmorph(session)
  pcall(ctrl.select_weapon, session, 0)
  ctrl.hold(session, 6, {}, label .. "_entry_glide")
  local i
  for i = 1, 100 do
    local st = session.state
    if not in_red(st) then
      break
    end
    if ctrl.x(st) <= 165 and ctrl.num(st.velocity_y) == 0 then
      break
    end
    ctrl.hold(session, 1, {"LEFT", "B"}, label .. "_clear_bat")
  end
  ctrl.hold(session, 6, {}, label .. "_bottom_settle")
  -- IBJ mid climb from bottom pin.
  if not ctrl.is_morph(session.state.pose) then
    pcall(ctrl.ensure_morph, session)
  end
  local c
  for c = 0, 89 do
    local st = session.state
    if in_hellway(st) then
      return ctrl.wait_ordinary_room(session, ROOM_HELLWAY, {
        settle_frames = geom.RED_TO_HELLWAY_EXIT_SETTLE, label = label,
      })
    end
    if not in_red(st) then
      ctrl.timeout(label .. ": left Red p1 " .. ctrl.brief(st))
    end
    if ctrl.y(st) <= 1820 then
      break
    end
    M.ibj_double(session, label .. "_p1_" .. c, 150, 1820)
  end
  if ctrl.y(session.state) > 1680 then
    M.tunnel_to_midplat(session, label .. "_tun")
  end
  if in_hellway(session.state) then
    return ctrl.wait_ordinary_room(session, ROOM_HELLWAY, {
      settle_frames = geom.RED_TO_HELLWAY_EXIT_SETTLE, label = label,
    })
  end
  M.play_upper_rle(session, M.HUMAN_FLOOR_RLE, label .. "_human850")
  if in_hellway(session.state) then
    return ctrl.wait_ordinary_room(session, ROOM_HELLWAY, {
      settle_frames = geom.RED_TO_HELLWAY_EXIT_SETTLE, label = label,
    })
  end
  local y_h = ctrl.y(session.state)
  if y_h <= 1550 and y_h >= 1300 then
    M.seat_left_after_handoff(session, label .. "_seat")
  end
  if ctrl.y(session.state) > 500 then
    ctrl.hold(session, 3, {"LEFT", "B"}, label .. "_wj_run")
    for i = 1, 12 do
      local st = ctrl.hold(session, 1, {"LEFT", "B", "A"}, label .. "_wj_j")
      if in_hellway(st) or not in_red(st) or ctrl.y(st) <= geom.RED_TOP_DOOR_Y + 40 then
        break
      end
    end
    local phases = {
      {"LEFT", 600, 1200}, {"RIGHT", 800, 1050}, {"LEFT", 800, 900},
      {"RIGHT", 800, 750}, {"LEFT", 800, 600}, {"RIGHT", 800, 450},
      {"LEFT", 800, 300}, {"RIGHT", 800, 200},
    }
    for i = 1, #phases do
      M.period_wj(session, label .. "_pwj" .. (i - 1), phases[i][1], phases[i][2], phases[i][3])
      if in_hellway(session.state) or not in_red(session.state) then
        break
      end
      if ctrl.y(session.state) <= geom.RED_TOP_DOOR_Y + 40 then
        break
      end
    end
  end
  if in_hellway(session.state) then
    return ctrl.wait_ordinary_room(session, ROOM_HELLWAY, {
      settle_frames = geom.RED_TO_HELLWAY_EXIT_SETTLE, label = label,
    })
  end
  if not in_red(session.state) then
    ctrl.timeout(label .. ": left Red unexpectedly: " .. ctrl.brief(session.state))
  end
  if ctrl.y(session.state) > geom.RED_TOP_DOOR_Y + 120 then
    ctrl.timeout(label .. ": upper residual " .. ctrl.brief(session.state))
  end
  return ctrl.play_run_shoot_exit(session, {
    from_room = ROOM_RED_TOWER,
    to_room = ROOM_HELLWAY,
    direction = "RIGHT",
    label = label,
    run_frames = geom.RED_TO_HELLWAY_EXIT_RUN,
    shoot_frames = geom.RED_TO_HELLWAY_EXIT_SHOOT,
    spin_frames = geom.RED_TO_HELLWAY_EXIT_SPIN,
    hold_frames = geom.RED_TO_HELLWAY_EXIT_HOLD,
    settle_frames = geom.RED_TO_HELLWAY_EXIT_SETTLE,
  })
end

return M
