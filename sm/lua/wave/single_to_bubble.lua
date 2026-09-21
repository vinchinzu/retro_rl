-- K4 Single Chamber → Bubble return (rr-u0y8). Deep pin climb + top-left door.
-- Lua 5.1. Session: step / hold / wait_until / span.

local G = require("wave.geometry")
local knockback = require("skills.knockback")

local ROOM_SC = G.ROOM_SINGLE_CHAMBER
local ROOM_BUBBLE = G.ROOM_BUBBLE
local LEDGE = {
  [1] = true, [2] = true, [9] = true, [10] = true, [25] = true, [26] = true,
  [27] = true, [28] = true, [37] = true, [38] = true, [137] = true, [138] = true,
}

local M = {}

local function y_band(state, band)
  return band[1] <= state.samus_y and state.samus_y <= band[2]
end

local function on_top(state)
  return state.room_id == ROOM_SC and state.samus_y <= G.STB_TOP_Y_MAX and state.velocity_y == 0
end

local function settle_ground(session, label, max_frames)
  max_frames = max_frames or 40
  for _ = 1, max_frames do
    local st = session.state
    if st.room_id ~= ROOM_SC then
      return
    end
    if knockback.is_knockback(st) then
      knockback.escape_kb(session, label, "LEFT", {stop_room_id = ROOM_BUBBLE})
    elseif st.pose == 31 or st.pose == 39 or st.pose == 40 or st.pose == 41
        or st.pose == 42 or st.pose == 65 or st.pose == 137 or st.pose == 138 then
      session:hold(1, {"UP"}, label .. "_unmorph")
    elseif st.velocity_y == 0 and LEDGE[st.pose] and st.door_transition == 0 then
      return
    else
      session:hold(1, {}, label .. "_settle")
    end
  end
end

local function play_rle(session, label, steps, stop_when)
  for i = 1, #steps do
    local n, buttons = steps[i][1], steps[i][2] or {}
    for _ = 1, n do
      local st = session.state
      if st.room_id == ROOM_BUBBLE or st.room_id ~= ROOM_SC then
        return
      end
      if stop_when and stop_when(st) then
        return
      end
      if knockback.is_knockback(st) then
        knockback.escape_kb(session, label, "LEFT", {stop_room_id = ROOM_BUBBLE})
      else
        local has_up = false
        for k = 1, #buttons do
          if buttons[k] == "UP" then
            has_up = true
          end
        end
        if (st.pose == 31 or st.pose == 39 or st.pose == 40 or st.pose == 41
            or st.pose == 42 or st.pose == 65 or st.pose == 137 or st.pose == 138)
            and not has_up then
          session:hold(1, {"UP"}, label .. "_unmorph")
        else
          session:hold(1, buttons, label)
        end
      end
    end
  end
end

local function deep_to_mid_low(session, label)
  if session.state.samus_y <= G.STB_MID_LOW_Y[2] + 15 then
    return
  end
  G.unmorph(session)
  G.select_weapon(session, 0)
  settle_ground(session, label .. "_deep")
  play_rle(session, label .. "_deep_rle", {
    {10, {}}, {16, {"LEFT"}}, {36, {"LEFT", "A"}}, {34, {"LEFT"}},
    {20, {"A"}}, {13, {"RIGHT", "A"}}, {4, {"RIGHT", "A", "X"}},
    {8, {"RIGHT", "X"}}, {16, {"RIGHT"}},
  }, function(st)
    return y_band(st, G.STB_MID_LOW_Y) and st.velocity_y == 0 and LEDGE[st.pose] and st.samus_x >= 70
  end)
  settle_ground(session, label .. "_ml_land", 50)
  if session.state.samus_y >= G.STB_DEEP_Y_MIN - 30 then
    G.unmorph(session)
    play_rle(session, label .. "_deep_retry", {
      {8, {"LEFT"}}, {28, {"LEFT", "A"}}, {20, {"LEFT"}},
      {18, {"A"}}, {14, {"RIGHT", "A"}}, {12, {"RIGHT"}},
    }, function(st)
      return st.samus_y < G.STB_DEEP_Y_MIN - 40
    end)
    settle_ground(session, label .. "_deep_retry_land", 50)
  end
end

local function mid_low_to_floor(session, label)
  if session.state.samus_y <= G.STB_FLOOR_Y[2] + 10 then
    return
  end
  if session.state.samus_y > G.STB_MID_LOW_Y[2] + 40 then
    return
  end
  G.unmorph(session)
  settle_ground(session, label .. "_ml")
  for _ = 1, 55 do
    local st = session.state
    if st.room_id ~= ROOM_SC then
      return
    end
    if st.samus_y < G.STB_MID_LOW_Y[1] - 10 then
      break
    end
    if knockback.is_knockback(st) then
      knockback.escape_kb(session, label, "RIGHT", {stop_room_id = ROOM_BUBBLE})
    elseif st.pose == 31 or st.pose == 39 or st.pose == 40 or st.pose == 41
        or st.pose == 42 or st.pose == 65 or st.pose == 137 or st.pose == 138 then
      session:hold(2, {"UP"}, label .. "_ml_unmorph")
    elseif st.samus_x >= 152 and st.velocity_y == 0 then
      break
    else
      session:hold(1, {"RIGHT", "B"}, label .. "_ml_run")
    end
  end
  play_rle(session, label .. "_ml_up", {
    {17, {"LEFT", "A"}}, {18, {"A"}}, {20, {"LEFT", "A"}}, {12, {"LEFT"}},
  }, function(st)
    return y_band(st, G.STB_FLOOR_Y) and st.velocity_y == 0
  end)
  settle_ground(session, label .. "_fl_land", 40)
end

local function floor_to_mid_hi(session, label)
  local y = session.state.samus_y
  if y <= G.STB_MID_HI_Y[2] + 10 then
    return
  end
  if not (G.STB_FLOOR_Y[1] - 30 <= y and y <= G.STB_FLOOR_Y[2] + 30) then
    return
  end
  G.unmorph(session)
  play_rle(session, label .. "_fl_climb", {
    {20, {"LEFT"}}, {12, {"B", "LEFT", "A"}}, {30, {"LEFT", "A"}},
    {16, {"A"}}, {20, {"RIGHT", "A"}}, {16, {"RIGHT"}},
  }, function(st)
    return st.samus_y <= G.STB_MID_HI_Y[2] + 10
  end)
  settle_ground(session, label .. "_mh_land", 40)
end

local function mid_hi_to_top(session, label)
  local y = session.state.samus_y
  if y <= G.STB_TOP_Y_MAX then
    return
  end
  if y > G.STB_MID_HI_Y[2] + 50 then
    return
  end
  G.unmorph(session)
  play_rle(session, label .. "_top", {
    {12, {"RIGHT"}}, {8, {"A"}}, {16, {"LEFT", "A"}}, {20, {"LEFT", "A"}},
    {12, {"LEFT"}}, {10, {"A"}}, {16, {"RIGHT", "A"}}, {12, {"RIGHT"}},
  }, function(st)
    return on_top(st) or st.room_id == ROOM_BUBBLE
  end)
  settle_ground(session, label .. "_top_land", 30)
end

local function reactive_climb_budget(session, label)
  for frame = 0, G.STB_CLIMB_FRAMES - 1 do
    local st = session.state
    if st.room_id ~= ROOM_SC or on_top(st) then
      return
    end
    if knockback.is_knockback(st) then
      knockback.escape_kb(session, label, "LEFT", {stop_room_id = ROOM_BUBBLE})
    else
      local y = st.samus_y
      if y > G.STB_MID_LOW_Y[2] then
        session:hold(1, {"LEFT", "A"}, label .. "_rx_deep")
      elseif y > G.STB_FLOOR_Y[2] then
        session:hold(1, {"LEFT", "A"}, label .. "_rx_ml")
      elseif y > G.STB_MID_HI_Y[2] then
        session:hold(1, {"LEFT", "A"}, label .. "_rx_fl")
      else
        session:hold(1, {"LEFT", "A"}, label .. "_rx_up")
      end
    end
  end
end

local function top_left_door(session, label)
  G.unmorph(session)
  G.select_weapon(session, 0)
  for frame = 0, G.STB_DOOR_FRAMES - 1 do
    local st = session.state
    if st.room_id == ROOM_BUBBLE or st.room_id ~= ROOM_SC then
      return
    end
    if knockback.is_knockback(st) then
      knockback.escape_kb(session, label, "LEFT", {stop_room_id = ROOM_BUBBLE})
    else
      local x, y = st.samus_x, st.samus_y
      if y > G.STB_TOP_Y_MAX + 20 then
        session:hold(1, {"LEFT", "A"}, label .. "_door_up")
      elseif x <= G.STB_DOOR_X + 10 and G.STB_DOOR_Y[1] <= y and y <= G.STB_DOOR_Y[2] then
        local phase = frame % 16
        if phase < 4 then
          session:hold(1, {"LEFT", "X"}, label .. "_door_shot")
        elseif phase < 12 then
          session:hold(1, {"LEFT", "B"}, label .. "_door_push")
        else
          session:hold(1, {"LEFT", "B", "A"}, label .. "_door_spin")
        end
      else
        session:hold(1, {"LEFT", "B"}, label .. "_door_run")
      end
    end
  end
end

function M.play_single_to_bubble(session)
  local label = "single_to_bubble"
  G.require_room(session, ROOM_SC, label)
  local start = session.frame
  G.unmorph(session)
  G.select_weapon(session, 0)
  for _attempt = 1, 3 do
    if session.state.room_id ~= ROOM_SC or on_top(session.state) then
      break
    end
    local y = session.state.samus_y
    if y > G.STB_MID_LOW_Y[2] then
      deep_to_mid_low(session, label)
    end
    if session.state.room_id ~= ROOM_SC then
      break
    end
    y = session.state.samus_y
    if y > G.STB_FLOOR_Y[2] and y <= G.STB_MID_LOW_Y[2] + 40 then
      mid_low_to_floor(session, label)
    end
    if session.state.room_id ~= ROOM_SC then
      break
    end
    y = session.state.samus_y
    if G.STB_FLOOR_Y[1] - 30 <= y and y <= G.STB_FLOOR_Y[2] + 30 then
      floor_to_mid_hi(session, label)
    end
    if session.state.room_id ~= ROOM_SC then
      break
    end
    y = session.state.samus_y
    if y <= G.STB_MID_HI_Y[2] + 50 and y > G.STB_TOP_Y_MAX then
      mid_hi_to_top(session, label)
    end
    if on_top(session.state) or session.state.room_id == ROOM_BUBBLE then
      break
    end
  end
  if session.state.room_id == ROOM_SC and not on_top(session.state)
      and session.state.samus_y > G.STB_TOP_Y_MAX then
    reactive_climb_budget(session, label)
  end
  if session.state.room_id == ROOM_SC then
    top_left_door(session, label)
  end
  if session.state.room_id == ROOM_SC and not on_top(session.state) then
    reactive_climb_budget(session, label)
    if session.state.room_id == ROOM_SC then
      top_left_door(session, label)
    end
  end
  if session.state.room_id ~= ROOM_BUBBLE then
    local state = session.state
    error(string.format(
      "%s: Bubble door missed; room=0x%04X pose=%d xy=(%d,%d) frames=%d",
      label, state.room_id, state.pose, state.samus_x, state.samus_y,
      session.frame - start
    ))
  end
  return G.wait_ordinary_room(session, ROOM_BUBBLE, G.STB_BUBBLE_SETTLE, label)
end

return M
