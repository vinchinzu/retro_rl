-- K4 Frog Save → Business return (rr-vsjy).
-- Lua 5.1. Session: step / hold / wait_until / span.

local G = require("wave.geometry")
local knockback = require("skills.knockback")

local ROOM_FROG = G.ROOM_FROG_SAVE
local ROOM_BUSINESS = G.ROOM_BUSINESS
local LEDGE = {
  [1] = true, [2] = true, [9] = true, [10] = true, [11] = true, [12] = true,
  [25] = true, [26] = true, [27] = true, [28] = true, [37] = true, [38] = true,
  [82] = true, [137] = true, [138] = true,
}

local M = {}

local function y_band(state, band)
  return band[1] <= state.samus_y and state.samus_y <= band[2]
end

local function on_door_sill(state)
  return state.room_id == ROOM_FROG
    and state.samus_x <= G.FTB_DOOR_X + 12
    and y_band(state, G.FTB_DOOR_Y)
    and state.velocity_y == 0
end

function M.play_frog_save_to_business(session)
  local label = "frog_save_to_business"
  G.require_room(session, ROOM_FROG, label)

  local start = session.frame
  G.unmorph(session)
  G.select_weapon(session, 0)
  for _ = 1, 24 do
    local st = session.state
    if st.velocity_y == 0 and LEDGE[st.pose] then
      break
    end
    session:hold(1, {}, label .. "_settle")
  end

  G.play_rle(session, label .. "_rle", {
    {8, {"LEFT"}},
    {20, {"B", "LEFT"}},
    {12, {"B", "LEFT", "A"}},
    {8, {"B", "LEFT"}},
    {12, {"B", "LEFT", "A"}},
    {16, {"B", "LEFT"}},
    {12, {"LEFT", "X"}},
    {8, {"LEFT"}},
    {24, {"B", "LEFT"}},
    {16, {"B", "LEFT", "X"}},
  }, ROOM_FROG, ROOM_BUSINESS)

  local min_x = session.state.samus_x
  for frame = 0, G.FTB_LEAVE_FRAMES - 1 do
    local st = session.state
    if st.room_id == ROOM_BUSINESS or st.room_id ~= ROOM_FROG then
      break
    end
    if knockback.is_knockback(st) then
      knockback.escape_kb(session, label, "LEFT", {stop_room_id = ROOM_BUSINESS})
    else
      local x, y = st.samus_x, st.samus_y
      if x < min_x then
        min_x = x
      end
      if on_door_sill(st) or (x <= G.FTB_DOOR_X + 20 and y_band(st, G.FTB_DOOR_Y)) then
        local phase = frame % 14
        if phase < 4 then
          session:hold(1, {"LEFT", "X"}, label .. "_door_shot")
        elseif phase < 11 then
          session:hold(1, {"B", "LEFT"}, label .. "_door_push")
        else
          session:hold(1, {"LEFT"}, label .. "_door_walk")
        end
      elseif G.FTB_TUBE_X[1] <= x and x <= G.FTB_TUBE_X[2] + 40 then
        local phase = frame % 16
        if phase < 6 then
          session:hold(1, {"B", "LEFT", "A"}, label .. "_rx_tube_hop")
        elseif phase < 12 then
          session:hold(1, {"B", "LEFT"}, label .. "_rx_tube_run")
        else
          session:hold(1, {"LEFT", "X"}, label .. "_rx_tube_shot")
        end
      elseif st.velocity_y ~= 0 then
        session:hold(1, {"B", "LEFT"}, label .. "_rx_air")
      elseif y > G.FTB_DOOR_Y[2] + 20 then
        session:hold(1, {"LEFT", "A"}, label .. "_rx_up")
      else
        local phase = frame % 16
        if phase < 10 then
          session:hold(1, {"B", "LEFT"}, label .. "_rx_run")
        elseif phase < 13 then
          session:hold(1, {"LEFT", "X"}, label .. "_rx_shot")
        else
          session:hold(1, {"B", "LEFT", "A"}, label .. "_rx_hop")
        end
      end
    end
  end

  if session.state.room_id ~= ROOM_BUSINESS then
    for frame = 0, G.FTB_DOOR_FRAMES - 1 do
      local st = session.state
      if st.room_id == ROOM_BUSINESS or st.room_id ~= ROOM_FROG then
        break
      end
      if knockback.is_knockback(st) then
        knockback.escape_kb(session, label, "LEFT", {stop_room_id = ROOM_BUSINESS})
      else
        local phase = frame % 12
        if phase < 4 then
          session:hold(1, {"LEFT", "X"}, label .. "_final_shot")
        elseif phase < 10 then
          session:hold(1, {"B", "LEFT"}, label .. "_final_push")
        else
          session:hold(1, {"LEFT"}, label .. "_final_walk")
        end
      end
    end
  end

  if session.state.room_id ~= ROOM_BUSINESS then
    local state = session.state
    local tube = ""
    if G.FTB_TUBE_X[1] - 10 <= state.samus_x and state.samus_x <= G.FTB_TUBE_X[2] + 20 then
      tube = " (save-tube stall?)"
    end
    error(string.format(
      "%s: Business door missed; room=0x%04X pose=%d xy=(%d,%d) frames=%d min_x=%d%s",
      label, state.room_id, state.pose, state.samus_x, state.samus_y,
      session.frame - start, min_x, tube
    ))
  end

  G.wait_ordinary_room(session, ROOM_BUSINESS, G.FTB_BUSINESS_SETTLE, label)
  G.unmorph(session)
  for _ = 1, 160 do
    local st = session.state
    if st.room_id ~= ROOM_BUSINESS then
      break
    end
    local x, y = st.samus_x, st.samus_y
    if y < 1350 then
      break
    end
    if 200 <= x and x <= 240 and st.velocity_y == 0 then
      session:hold(8, {}, label .. "_business_floor_pin")
      return session.state
    end
    if x < 200 then
      session:hold(1, {"RIGHT", "B"}, label .. "_business_floor_r")
    elseif x > 240 then
      session:hold(1, {"LEFT"}, label .. "_business_floor_l")
    else
      session:hold(1, {}, label .. "_business_floor_idle")
    end
  end
  return session.state
end

return M
