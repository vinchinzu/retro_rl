-- K4 Wave branch geometry: named bands, seats, and predicates.
-- Room-prefixed constants only (never a bare DOOR_X across hops).
-- Lua 5.1. Session API: step / hold / wait_until / span.

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

M.ROOM_BUBBLE = 0xACB3
M.ROOM_SINGLE_CHAMBER = 0xAD5E
M.ROOM_DOUBLE_CHAMBER = 0xADAD
M.ROOM_WAVE = 0xADDE
M.ROOM_UPPER_NORFAIR_FARM = 0xAF72
M.ROOM_FROG_SPEEDWAY = 0xB106
M.ROOM_FROG_SAVE = 0xB167
M.ROOM_BUSINESS = 0xA7DE

M.WAVE_BEAM_MASK = 0x0001
M.SPEED_BOOSTER_MASK = 0x2000

M.STANDING_POSES = geom.STANDING_POSES or {
  [1] = true, [2] = true, [9] = true, [10] = true,
  [25] = true, [26] = true, [27] = true, [28] = true,
  [37] = true, [38] = true, [137] = true, [138] = true,
}
M.MORPH_POSES = {
  [29] = true, [30] = true, [31] = true, [32] = true,
  [49] = true, [50] = true, [65] = true, [66] = true,
}

-- Wave → Double return
M.WAVE_DOOR_X = 48
M.WAVE_LEAVE_FRAMES = 420
M.WAVE_DOUBLE_SETTLE = 280

-- K4.8 Bubble → Single
M.BSC_TOP_Y_MAX = 200
M.BSC_DROP_X = {370, 400}
M.BSC_DROP_TARGET_X = 385
M.BSC_MID_Y = {220, 340}
M.BSC_FLOOR_Y = 360
M.BSC_DOOR_Y = {380, 420}
M.BSC_DOOR_X = 470
M.BSC_SINGLE_SETTLE = 320
M.BSC_TOP_WALK_FRAMES = 400
M.BSC_DROP_FRAMES = 500
M.BSC_NAV_TO_DOOR_FRAMES = 1200
M.BSC_DOOR_PUSH_FRAMES = 400

-- K4.9 Single → Double
M.SC_TOP_Y = 200
M.SC_MID_Y = {250, 290}
M.SC_FLOOR_Y = {380, 420}
M.SC_SHOT_X = {115, 135}
M.SC_DOOR_X = 220
M.SC_DOUBLE_SETTLE = 320

-- Double → Single return
M.DTS_LEDGE_Y_MAX = 170
M.DTS_HOP_LAUNCH_X = 950
M.DTS_MID_X = {760, 860}
M.DTS_MID_Y = {220, 360}
M.DTS_FLOOR_Y_MIN = 420
M.DTS_FLOOR_Y = {430, 470}
M.DTS_MORPH_TUNNEL_X = {440, 580}
M.DTS_GAP_LAUNCH_X = 190
M.DTS_DOOR_X = 40
M.DTS_DOOR_Y = {380, 415}
M.DTS_SINGLE_SETTLE = 280
M.DTS_LEDGE_FRAMES = 200
M.DTS_DROP_FRAMES = 700
M.DTS_FLOOR_FRAMES = 900
M.DTS_DOOR_FRAMES = 500

-- Single → Bubble return
M.STB_DEEP_Y_MIN = 580
M.STB_MID_LOW_Y = {500, 540}
M.STB_FLOOR_Y = {380, 420}
M.STB_MID_HI_Y = {250, 290}
M.STB_UPPER_Y = {190, 230}
M.STB_TOP_Y_MAX = 160
M.STB_WALL_X = 55
M.STB_MID_LOW_LAND_X = {45, 160}
M.STB_FLOOR_LAND_X = {80, 160}
M.STB_DOOR_X = 40
M.STB_DOOR_Y = {120, 160}
M.STB_BUBBLE_SETTLE = 280
M.STB_DEEP_FRAMES = 200
M.STB_CLIMB_FRAMES = 1800
M.STB_DOOR_FRAMES = 400

-- Bubble → Farm return
M.BTF_MID_Y = {380, 420}
M.BTF_UPPER_Y = {130, 180}
M.BTF_MID_LOW_Y = {500, 560}
M.BTF_BOTTOM_Y_MIN = 700
M.BTF_BOTTOM_FLOOR_Y = {880, 950}
M.BTF_DOOR_X = 40
M.BTF_DOOR_Y = {880, 940}
M.BTF_FARM_SETTLE = 280
M.BTF_CLIMB_FRAMES = 500
M.BTF_DROP_FRAMES = 900
M.BTF_BOTTOM_FRAMES = 1600
M.BTF_DOOR_FRAMES = 400

-- Farm → Speedway
M.FTS_PIN_X = {400, 560}
M.FTS_PIN_Y = {100, 180}
M.FTS_MID_HOP_X = 320
M.FTS_DOOR_X = 40
M.FTS_DOOR_Y = {120, 160}
M.FTS_LEAVE_FRAMES = 500
M.FTS_DOOR_FRAMES = 280
M.FTS_SPEEDWAY_SETTLE = 280

-- Speedway → Frog Save
M.STF_PIN_X = {1950, 2100}
M.STF_PIN_Y = {100, 180}
M.STF_BOOST_BLOCK_X = 800
M.STF_DOOR_X = 40
M.STF_DOOR_Y = {120, 160}
M.STF_LEAVE_FRAMES = 900
M.STF_DOOR_FRAMES = 320
M.STF_FROG_SETTLE = 280

-- Frog Save → Business
M.FTB_PIN_X = {160, 280}
M.FTB_PIN_Y = {100, 180}
M.FTB_TUBE_X = {90, 160}
M.FTB_DOOR_X = 40
M.FTB_DOOR_Y = {120, 160}
M.FTB_LEAVE_FRAMES = 500
M.FTB_DOOR_FRAMES = 280
M.FTB_BUSINESS_SETTLE = 280

-- Double Chamber → Wave
M.DC_WAVE_SETTLE = 280
M.DC_GATE_X = {360, 430}
M.DC_GATE_SEAT_X = {365, 390}
M.DC_GATE_SEAT_Y_MAX = 200
M.DC_GATE_OPEN_SEAT_X = {370, 375}
M.DC_GATE_OPEN_SEAT_Y = 139
M.DC_PAST_GATE_X = 480
M.DC_DOOR_X = 920
M.DC_DOOR_Y_MAX = 180
M.DC_LEDGE_Y_MAX = 165
M.DC_RUNWAY_X = 425
M.DC_EDGE_X = 600
M.DC_MISSILE_X = 495
M.DC_WJ = {
  into = "LEFT",
  flip = "LEFT",
  into_frames = 6,
  amid_frames = 0,
  flip_frames = 0,
  delay_into_frames = 3,
}
M.DC_WJ_LEFT_FOLLOW = 8

M.GATE_OPEN_RLE = require("wave.double_chamber_gate_open_rle")

function M.is_morph(pose)
  return M.MORPH_POSES[pose] == true
end

function M.has_wave(state)
  return band(state.collected_beams or 0, M.WAVE_BEAM_MASK) ~= 0
end

function M.has_speed(state)
  return band(state.collected_items or 0, M.SPEED_BOOSTER_MASK) ~= 0
end

function M.dc_on_missile_ledge(state)
  return state.room_id == M.ROOM_DOUBLE_CHAMBER
    and state.samus_y <= M.DC_LEDGE_Y_MAX
    and state.velocity_y == 0
end

function M.dc_on_sill(state)
  return state.room_id == M.ROOM_DOUBLE_CHAMBER
    and state.samus_x >= M.DC_DOOR_X - 20
    and state.samus_y < M.DC_DOOR_Y_MAX
    and state.velocity_y == 0
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

function M.play_rle(session, label, steps, from_room, to_room, stop_when)
  local knockback = require("skills.knockback")
  for i = 1, #steps do
    local n = steps[i][1]
    local buttons = steps[i][2] or {}
    for _ = 1, n do
      local st = session.state
      if st.room_id == to_room then
        return
      end
      if st.room_id ~= from_room then
        return
      end
      if stop_when and stop_when(st) then
        return
      end
      if knockback.is_knockback(st) then
        knockback.escape_kb(session, label, "LEFT", {stop_room_id = to_room})
      else
        session:hold(1, buttons, label)
      end
    end
  end
end

return M
