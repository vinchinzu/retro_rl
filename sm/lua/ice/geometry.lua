-- Ice branch geometry, room ids, RLE, and hop-local session helpers.
-- Lua 5.1. Session API: step / hold / wait_until / span.
-- Pins from human tape Phase B (Business Super → Ice PLM / return).

local knockback = require("skills.knockback")
local geom = require("skills.geometry")

local M = {}

local function band(a, b)
  if bit and bit.band then
    return bit.band(a, b)
  end
  local r, p = 0, 1
  a = a % 65536
  b = b % 65536
  while a > 0 and b > 0 do
    if a % 2 == 1 and b % 2 == 1 then
      r = r + p
    end
    a = math.floor(a / 2)
    b = math.floor(b / 2)
    p = p * 2
  end
  return r
end
M.band = band

M.ROOM_BUSINESS = 0xA7DE
M.ROOM_HJ_SHAFT = 0xAA41
M.ROOM_ICE_GATE = 0xA815
M.ROOM_ICE_ACID = 0xA75D
M.ROOM_ICE_TUTORIAL = 0xA865
M.ROOM_ICE_SNAKE = 0xA8B9
M.ROOM_ICE = 0xA890

M.STANDING_POSES = geom.STANDING_POSES or {
  [1] = true, [2] = true, [9] = true, [10] = true,
  [25] = true, [26] = true, [27] = true, [28] = true,
  [37] = true, [38] = true, [137] = true, [138] = true,
}
M.LEDGE_POSES = geom.LEDGE_POSES or {
  [1] = true, [2] = true, [9] = true, [10] = true,
  [25] = true, [26] = true, [27] = true, [28] = true,
  [37] = true, [38] = true,
}
M.MORPH_POSES = {
  [29] = true, [30] = true, [31] = true, [32] = true,
  [49] = true, [50] = true, [65] = true, [66] = true,
}

M.BUSINESS_ELEVATOR_Y = 680
M.ICE_SUPER_Y_MIN = 880
M.ICE_SUPER_Y_MAX = 960
M.ICE_SUPER_LIP_X_MAX = 90
M.ICE_SUPER_DOOR_X = 40
M.ICE_APPROACH_Y = {980, 1100}
M.ICE_APPROACH_X = {90, 180}
M.ELEVATOR_SETTLE_FRAMES = 600
M.DOOR_BAND_FRAMES = 700
M.SUPER_PRESSURE_FRAMES = 400
M.ICE_GATE_SETTLE_FRAMES = 320

M.ACID_HANDOFF_X = {400, 520}
M.ACID_FLOOR_Y = {120, 160}
M.ACID_FLOOR_Y_MAX = 160
M.ACID_LEFT_DOOR_X = 40
M.ACID_SNAKE_SETTLE_FRAMES = 320

M.SNAKE_HANDOFF_X = {80, 250}
M.SNAKE_HANDOFF_Y = {600, 720}
M.SNAKE_L1_Y = {560, 600}
M.SNAKE_L2_Y = {500, 540}
M.SNAKE_L3_Y = {440, 480}
M.SNAKE_L4_Y = {380, 420}
M.SNAKE_L5_Y = {310, 350}
M.SNAKE_L6_Y = {250, 290}
M.SNAKE_L7_Y = {190, 220}
M.SNAKE_TOP_Y = {120, 160}
M.SNAKE_TOP_X = {80, 180}
M.SNAKE_TUNNEL_Y = {365, 395}
M.SNAKE_TUNNEL_FLOOR_Y = 377
M.SNAKE_FALSE_LEDGE_Y = {400, 430}
M.SNAKE_TUNNEL_X_MIN = 195
M.SNAKE_TUNNEL_EXIT_X = 320
M.SNAKE_WALL_X = 171
M.SNAKE_MID_SHELF_Y = {480, 540}
M.SNAKE_MID_SHELF_X = {180, 230}
M.ICE_BEAM_MASK = 0x0002
M.ICE_PLM_X = 187
M.ICE_ROOM_SETTLE = 280
M.SNAKE_CLIMB_FRAMES = 2500
M.SNAKE_TUNNEL_FRAMES = 900
M.SNAKE_ICE_COLLECT_FRAMES = 500
M.SNAKE_DOOR_X = 470

M.ICE_LEAVE_DOOR_X = 40
M.ICE_LEAVE_FRAMES = 480
M.ICE_SNAKE_RETURN_SETTLE = 280

M.SNAKE_TUTORIAL_DOOR_X = 210
M.SNAKE_TUTORIAL_DOOR_Y = {100, 175}
M.SNAKE_TO_TUTORIAL_DROP_FRAMES = 700
M.SNAKE_TOP_TO_TUTORIAL_FRAMES = 500
M.TUTORIAL_RETURN_SETTLE = 280

M.TUTORIAL_SHELF_Y = {120, 155}
M.TUTORIAL_FLOOR_Y = {120, 160}
M.TUTORIAL_LOWER_Y = {180, 220}
M.TUTORIAL_DOOR_X = 450
M.TUTORIAL_DOOR_Y = {100, 175}
M.TUTORIAL_TO_GATE_FRAMES = 2200
M.GATE_RETURN_SETTLE = 280

M.GATE_MID_TOP_X = {450, 900}
M.GATE_MID_TOP_Y = {100, 200}
M.GATE_TUNNEL_X = {860, 920}
M.GATE_TUNNEL_Y = {555, 585}
M.GATE_SUPER_DOOR_X = 1740
M.GATE_TO_BUSINESS_FRAMES = 2200
M.BUSINESS_RETURN_SETTLE = 280

M.ACID_TO_SNAKE_RLE = require("ice.acid_to_snake_rle")
M.TUTORIAL_TO_GATE_RLE = require("ice.tutorial_to_gate_rle")
M.GATE_TO_BUSINESS_RLE = require("ice.gate_to_business_rle")

do
  local acc = 0
  local mid = {}
  for i = 1, #M.TUTORIAL_TO_GATE_RLE do
    if acc >= 500 then
      break
    end
    local n = M.TUTORIAL_TO_GATE_RLE[i][1]
    local btns = M.TUTORIAL_TO_GATE_RLE[i][2]
    local take = n
    if acc + take > 500 then
      take = 500 - acc
    end
    mid[#mid + 1] = {take, btns}
    acc = acc + take
  end
  M.TUTORIAL_MID_RLE = mid
end

function M.in_set(set, v)
  return set[v] == true
end

function M.is_morph(pose)
  return M.MORPH_POSES[pose] == true
end

function M.require_room(session, room_id, label)
  local st = session.state
  if st.room_id ~= room_id then
    error(string.format(
      "%s: expected room 0x%04X, got 0x%04X at frame %d",
      label, room_id, st.room_id, session.frame
    ))
  end
end

function M.select_weapon(session, target, max_cycles)
  max_cycles = max_cycles or 8
  for _ = 1, max_cycles do
    if session.state.selected_item == target then
      return
    end
    session:hold(1, {"SELECT"}, "select_weapon")
    session:hold(25, {}, "select_weapon_settle")
  end
  if session.state.selected_item ~= target then
    error(string.format(
      "could not select weapon %d, still %d",
      target, session.state.selected_item
    ))
  end
end

function M.unmorph(session)
  local pose = session.state.pose
  if pose == 39 or pose == 40 or pose == 137 or pose == 138
      or pose == 9 or pose == 10 or M.is_morph(pose) then
    session:hold(8, {"UP"}, "unmorph")
    if not M.is_morph(session.state.pose) then
      session:hold(8, {"A"}, "unmorph")
    end
    session:hold(10, {}, "unmorph_settle")
  end
end

function M.wait_ordinary_room(session, room_id, settle_frames, label, x_range, y_range, min_settle)
  settle_frames = settle_frames or 200
  min_settle = min_settle or 15
  local last
  for frame = 0, settle_frames - 1 do
    last = session:hold(1, {}, label .. "_settle")
    if last.room_id == room_id and last.game_state == 8
        and last.door_transition == 0 and frame > min_settle then
      local ok = true
      if x_range and not (x_range[1] <= last.samus_x and last.samus_x <= x_range[2]) then
        ok = false
      end
      if y_range and not (y_range[1] <= last.samus_y and last.samus_y <= y_range[2]) then
        ok = false
      end
      if ok then
        return last
      end
    end
  end
  last = session.state
  if last.room_id ~= room_id then
    error(string.format("%s: expected 0x%04X, got 0x%04X", label, room_id, last.room_id))
  end
  if x_range or y_range then
    error(string.format(
      "%s: settled in room but position window missed xy=(%d,%d)",
      label, last.samus_x, last.samus_y
    ))
  end
  return last
end

function M.ensure_morph(session, max_attempts)
  max_attempts = max_attempts or 5
  for attempt = 0, max_attempts - 1 do
    if M.is_morph(session.state.pose) then
      return session.state
    end
    local up = 4 + math.min(attempt, 3) * 2
    session:hold(up, {"UP"}, "morph_pre")
    session:hold(3 + attempt, {}, "morph_idle")
    session:hold(5 + attempt * 2, {"DOWN"}, "morph_tap1")
    session:hold(3, {}, "morph_release")
    session:hold(6 + attempt * 3, {"DOWN"}, "morph_tap2")
    local timeout = 28 + attempt * 4
    local got = false
    for _ = 1, timeout do
      if M.is_morph(session.state.pose) then
        got = true
        break
      end
      session:hold(1, {}, "morph_poll")
    end
    if got then
      return session.state
    end
  end
  error(string.format("ensure_morph failed, pose=%d", session.state.pose))
end

function M.play_script(session, runs, reason, room_id, stop_when, on_lag)
  on_lag = on_lag or "ignore"
  local st = session.state
  for i = 1, #runs do
    local n = runs[i][1]
    local btns = runs[i][2] or {}
    for _ = 1, n do
      if session.state.room_id ~= room_id then
        return session.state
      end
      if stop_when and stop_when(session.state) then
        return session.state
      end
      if on_lag == "break" then
        local pose = session.state.pose
        if pose == 137 or pose == 138 or pose == 164 then
          for k = 1, 40 do
            local p = session.state.pose
            if p ~= 137 and p ~= 138 and p ~= 164 then
              break
            end
            session:hold(1, {"A"}, reason .. "_lag")
            session:hold(2, {}, reason .. "_lag")
          end
        end
      end
      st = session:hold(1, btns, reason)
      if stop_when and stop_when(session.state) then
        return session.state
      end
    end
  end
  return session.state
end

function M.in_business(state)
  return state.room_id == M.ROOM_BUSINESS
end

function M.in_ice_gate(state)
  return state.room_id == M.ROOM_ICE_GATE
end

function M.in_ice_acid(state)
  return state.room_id == M.ROOM_ICE_ACID
end

function M.in_ice_snake(state)
  return state.room_id == M.ROOM_ICE_SNAKE
end

function M.in_ice_tutorial(state)
  return state.room_id == M.ROOM_ICE_TUTORIAL
end

function M.has_ice(state)
  return band(state.collected_beams or 0, M.ICE_BEAM_MASK) ~= 0
end

function M.on_snake_floor(state)
  if not M.in_ice_snake(state) then
    return false
  end
  local x, y = state.samus_x, state.samus_y
  if not (M.SNAKE_HANDOFF_X[1] <= x and x <= M.SNAKE_HANDOFF_X[2]) then
    return false
  end
  if not (M.SNAKE_HANDOFF_Y[1] <= y and y <= M.SNAKE_HANDOFF_Y[2]) then
    return false
  end
  return state.velocity_y == 0
end

function M.on_snake_top(state)
  if not M.in_ice_snake(state) then
    return false
  end
  local x, y = state.samus_x, state.samus_y
  if not (M.SNAKE_TOP_Y[1] <= y and y <= M.SNAKE_TOP_Y[2]) then
    return false
  end
  if not (M.SNAKE_TOP_X[1] <= x and x <= M.SNAKE_TOP_X[2] + 80) then
    return false
  end
  return state.velocity_y == 0
    and (M.STANDING_POSES[state.pose] or M.LEDGE_POSES[state.pose])
end

function M.on_snake_tunnel_band(state)
  if not M.in_ice_snake(state) then
    return false
  end
  local x, y = state.samus_x, state.samus_y
  return x >= M.SNAKE_TUNNEL_X_MIN
    and M.SNAKE_TUNNEL_Y[1] <= y and y <= M.SNAKE_TUNNEL_Y[2]
end

function M.on_snake_false_ledge(state)
  if not M.in_ice_snake(state) then
    return false
  end
  local x, y = state.samus_x, state.samus_y
  return x >= M.SNAKE_TUNNEL_X_MIN
    and M.SNAKE_FALSE_LEDGE_Y[1] <= y and y <= M.SNAKE_FALSE_LEDGE_Y[2]
end

function M.on_snake_mid_shelf(state)
  if not M.in_ice_snake(state) then
    return false
  end
  local x, y = state.samus_x, state.samus_y
  if not (M.SNAKE_MID_SHELF_X[1] <= x and x <= M.SNAKE_MID_SHELF_X[2]) then
    return false
  end
  if not (M.SNAKE_MID_SHELF_Y[1] <= y and y <= M.SNAKE_MID_SHELF_Y[2]) then
    return false
  end
  return state.velocity_y == 0
end

function M.on_acid_floor(state)
  if not M.in_ice_acid(state) then
    return false
  end
  local y = state.samus_y
  if not (M.ACID_FLOOR_Y[1] <= y and y <= M.ACID_FLOOR_Y[2]) then
    return false
  end
  if state.velocity_y ~= 0 then
    return false
  end
  return M.STANDING_POSES[state.pose] or M.LEDGE_POSES[state.pose]
end

function M.on_ice_super_lip(state)
  if not M.in_business(state) then
    return false
  end
  local y, x = state.samus_y, state.samus_x
  if not (M.ICE_SUPER_Y_MIN <= y and y <= M.ICE_SUPER_Y_MAX) then
    return false
  end
  if x > M.ICE_SUPER_LIP_X_MAX then
    return false
  end
  if state.velocity_y ~= 0 then
    return false
  end
  local p = state.pose
  return M.LEDGE_POSES[p] or p == 1 or p == 2 or p == 9 or p == 10
end

function M.in_ice_super_band(state)
  if not M.in_business(state) then
    return false
  end
  local y = state.samus_y
  return M.ICE_SUPER_Y_MIN <= y and y <= M.ICE_SUPER_Y_MAX
end

-- Silence unused import if knockback is only used by callers.
M._knockback = knockback

return M
