-- K4 Bubble → Upper Norfair Farm return (rr-czg9).
-- Lua 5.1. Session: step / hold / wait_until / span.

local G = require("wave.geometry")
local knockback = require("skills.knockback")

local ROOM_BUBBLE = G.ROOM_BUBBLE
local ROOM_FARM = G.ROOM_UPPER_NORFAIR_FARM
local LEDGE = {
  [1] = true, [2] = true, [9] = true, [10] = true, [25] = true, [26] = true,
  [27] = true, [28] = true, [37] = true, [38] = true, [137] = true, [138] = true,
}

local M = {}

local function y_band(state, band)
  return band[1] <= state.samus_y and state.samus_y <= band[2]
end

local function on_door_sill(state)
  return state.room_id == ROOM_BUBBLE and state.samus_x <= G.BTF_DOOR_X + 12
    and y_band(state, G.BTF_DOOR_Y) and state.velocity_y == 0
end

local function play_rle(session, label, steps, stop_when)
  for i = 1, #steps do
    local n, buttons = steps[i][1], steps[i][2] or {}
    for _ = 1, n do
      local st = session.state
      if st.room_id == ROOM_FARM or st.room_id ~= ROOM_BUBBLE then
        return
      end
      if stop_when and stop_when(st) then
        return
      end
      if knockback.is_knockback(st) then
        knockback.escape_kb(session, label, "LEFT", {stop_room_id = ROOM_FARM})
      else
        session:hold(1, buttons, label)
      end
    end
  end
end

local function mid_to_upper(session, label)
  if session.state.samus_y <= G.BTF_UPPER_Y[2] + 20 then
    return
  end
  if session.state.samus_y > G.BTF_MID_Y[2] + 50 then
    return
  end
  G.unmorph(session)
  G.select_weapon(session, 0)
  for _ = 1, 20 do
    local st = session.state
    if st.velocity_y == 0 and LEDGE[st.pose] then
      break
    end
    session:hold(1, {}, label .. "_mid_settle")
  end
  play_rle(session, label .. "_climb", {
    {23, {"LEFT"}}, {40, {"LEFT", "A"}}, {52, {"LEFT", "A", "X"}},
    {3, {"LEFT"}}, {5, {}}, {3, {"A"}}, {31, {"LEFT", "A"}}, {2, {"A"}},
    {8, {}}, {8, {"X"}}, {6, {}}, {6, {"X"}}, {4, {}},
  }, function(st)
    return y_band(st, G.BTF_UPPER_Y) and st.velocity_y == 0 and LEDGE[st.pose]
  end)
  for frame = 0, math.floor(G.BTF_CLIMB_FRAMES / 3) - 1 do
    local st = session.state
    if st.room_id ~= ROOM_BUBBLE then
      return
    end
    if y_band(st, G.BTF_UPPER_Y) and st.velocity_y == 0 then
      return
    end
    if st.samus_y <= G.BTF_UPPER_Y[2] then
      return
    end
    if knockback.is_knockback(st) then
      knockback.escape_kb(session, label, "LEFT", {stop_room_id = ROOM_FARM})
    elseif G.is_morph(st.pose) then
      session:hold(3, {"UP"}, label .. "_climb_unmorph")
    elseif st.velocity_y ~= 0 then
      session:hold(1, {"LEFT", "A"}, label .. "_climb_air")
    elseif st.samus_x > 420 then
      session:hold(1, {"LEFT"}, label .. "_climb_approach")
    else
      local phase = frame % 16
      if phase < 10 then
        session:hold(1, {"LEFT", "A"}, label .. "_climb_spin")
      else
        session:hold(1, {"LEFT", "A", "X"}, label .. "_climb_shot")
      end
    end
  end
end

local function upper_to_mid_low(session, label)
  local y = session.state.samus_y
  if y >= G.BTF_MID_LOW_Y[1] - 5 then
    return
  end
  if y > G.BTF_UPPER_Y[2] + 100 then
    return
  end
  G.unmorph(session)
  for _ = 1, 15 do
    local st = session.state
    if st.velocity_y == 0 and LEDGE[st.pose] then
      break
    end
    session:hold(1, {}, label .. "_up_settle")
  end
  play_rle(session, label .. "_drop", {
    {6, {"LEFT"}}, {5, {"X"}}, {4, {}}, {1, {"LEFT"}}, {9, {"LEFT", "A"}},
    {54, {"LEFT"}}, {14, {"LEFT", "A"}}, {6, {"LEFT"}}, {3, {"DOWN"}},
    {7, {"DOWN", "X"}}, {5, {"DOWN"}}, {6, {"DOWN", "X"}}, {4, {"DOWN"}},
    {5, {"DOWN", "X"}}, {5, {"DOWN", "LEFT"}}, {5, {"LEFT"}},
    {7, {"DOWN", "LEFT"}}, {20, {"DOWN"}},
  }, function(st)
    return st.samus_y >= G.BTF_MID_LOW_Y[1] - 5
  end)
  for _ = 1, math.floor(G.BTF_DROP_FRAMES / 3) do
    local st = session.state
    if st.room_id ~= ROOM_BUBBLE or st.samus_y >= G.BTF_MID_LOW_Y[1] - 5 then
      return
    end
    if knockback.is_knockback(st) then
      knockback.escape_kb(session, label, "LEFT", {stop_room_id = ROOM_FARM})
    elseif st.samus_x > 200 then
      session:hold(1, {"LEFT"}, label .. "_drop_l")
    elseif not G.is_morph(st.pose) and st.velocity_y ~= 0 then
      session:hold(1, {"DOWN"}, label .. "_drop_morph")
    else
      session:hold(1, {"LEFT", "DOWN"}, label .. "_drop_roll")
    end
  end
end

local function mid_low_through_bottom_to_door(session, label)
  play_rle(session, label .. "_bot_rle", {
    {20, {"LEFT"}}, {12, {"DOWN"}}, {40, {"LEFT"}}, {20, {"DOWN", "LEFT"}},
    {40, {"LEFT"}}, {16, {"DOWN"}}, {60, {"LEFT"}}, {20, {"UP"}},
    {40, {"LEFT", "B"}}, {20, {"LEFT", "B", "A"}}, {80, {"LEFT", "B"}},
  }, function(st)
    return st.room_id == ROOM_FARM or on_door_sill(st)
  end)
  for frame = 0, G.BTF_BOTTOM_FRAMES - 1 do
    local st = session.state
    if st.room_id == ROOM_FARM or st.room_id ~= ROOM_BUBBLE then
      return
    end
    if on_door_sill(st) then
      return
    end
    if knockback.is_knockback(st) then
      knockback.escape_kb(session, label, "LEFT", {stop_room_id = ROOM_FARM})
    else
      local x, y = st.samus_x, st.samus_y
      if y < G.BTF_BOTTOM_Y_MIN then
        if not G.is_morph(st.pose) then
          session:hold(1, {"DOWN"}, label .. "_bot_morph")
        else
          session:hold(1, {"LEFT"}, label .. "_bot_drop")
        end
      elseif y < G.BTF_BOTTOM_FLOOR_Y[1] then
        session:hold(1, {"LEFT"}, label .. "_bot_roll")
      elseif x > G.BTF_DOOR_X + 20 then
        local phase = frame % 16
        if phase < 10 then
          session:hold(1, {"LEFT", "B"}, label .. "_bot_run")
        elseif phase < 13 then
          session:hold(1, {"LEFT", "B", "A"}, label .. "_bot_hop")
        else
          session:hold(1, {"LEFT", "X"}, label .. "_bot_shot")
        end
      else
        session:hold(1, {"LEFT", "B"}, label .. "_bot_push")
      end
    end
  end
end

local function door_budget(session, label)
  for frame = 0, G.BTF_DOOR_FRAMES - 1 do
    local st = session.state
    if st.room_id == ROOM_FARM or st.room_id ~= ROOM_BUBBLE then
      return
    end
    if knockback.is_knockback(st) then
      knockback.escape_kb(session, label, "LEFT", {stop_room_id = ROOM_FARM})
    elseif G.is_morph(st.pose) then
      session:hold(3, {"UP"}, label .. "_final_unmorph")
    elseif st.samus_x > G.BTF_DOOR_X + 20 then
      session:hold(1, {"LEFT", "B"}, label .. "_final_run")
    else
      local phase = frame % 12
      if phase < 7 then
        session:hold(1, {"B", "LEFT"}, label .. "_final_push")
      elseif phase < 10 then
        session:hold(1, {"LEFT", "X"}, label .. "_final_shot")
      else
        session:hold(1, {"LEFT"}, label .. "_final_walk")
      end
    end
  end
end

function M.play_bubble_to_farm(session)
  local label = "bubble_to_farm"
  G.require_room(session, ROOM_BUBBLE, label)
  local start = session.frame
  G.unmorph(session)
  G.select_weapon(session, 0)
  local y = session.state.samus_y
  if G.BTF_MID_Y[1] - 40 <= y and y <= G.BTF_MID_Y[2] + 50 then
    mid_to_upper(session, label)
  end
  if session.state.room_id == ROOM_BUBBLE then
    y = session.state.samus_y
    if y < G.BTF_MID_LOW_Y[1] - 5 then
      upper_to_mid_low(session, label)
    end
  end
  if session.state.room_id == ROOM_BUBBLE then
    mid_low_through_bottom_to_door(session, label)
  end
  if session.state.room_id == ROOM_BUBBLE
      and (on_door_sill(session.state) or session.state.samus_x <= 60) then
    door_budget(session, label)
  end
  if session.state.room_id ~= ROOM_FARM then
    local state = session.state
    error(string.format(
      "%s: Farm door missed; room=0x%04X pose=%d xy=(%d,%d) frames=%d",
      label, state.room_id, state.pose, state.samus_x, state.samus_y,
      session.frame - start
    ))
  end
  return G.wait_ordinary_room(session, ROOM_FARM, G.BTF_FARM_SETTLE, label)
end

return M
