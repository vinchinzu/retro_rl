-- K4 Farm → Frog Speedway return (rr-z13h). Needs Speed.
-- Lua 5.1. Session: step / hold / wait_until / span.

local G = require("wave.geometry")
local knockback = require("skills.knockback")

local ROOM_FARM = G.ROOM_UPPER_NORFAIR_FARM
local ROOM_SPEEDWAY = G.ROOM_FROG_SPEEDWAY
local LEDGE = {
  [1] = true, [2] = true, [9] = true, [10] = true, [11] = true, [12] = true,
  [25] = true, [26] = true, [27] = true, [28] = true, [37] = true, [38] = true,
  [137] = true, [138] = true,
}

local M = {}

local function y_band(state, band)
  return band[1] <= state.samus_y and state.samus_y <= band[2]
end

local function on_door_sill(state)
  return state.room_id == ROOM_FARM
    and state.samus_x <= G.FTS_DOOR_X + 12
    and y_band(state, G.FTS_DOOR_Y)
    and state.velocity_y == 0
end

function M.play_farm_to_speedway(session)
  local label = "farm_to_speedway"
  G.require_room(session, ROOM_FARM, label)
  if not G.has_speed(session.state) then
    error(string.format(
      "%s: Speed Booster not collected (items=0x%04X; need bit 0x%04X)",
      label, session.state.collected_items or 0, G.SPEED_BOOSTER_MASK
    ))
  end

  local start = session.frame
  G.unmorph(session)
  G.select_weapon(session, 0)
  for _ = 1, 20 do
    local st = session.state
    if st.velocity_y == 0 and LEDGE[st.pose] then
      break
    end
    session:hold(1, {}, label .. "_settle")
  end

  G.play_rle(session, label .. "_rle", {
    {10, {"LEFT"}},
    {35, {"B", "LEFT"}},
    {44, {"B", "LEFT", "A"}},
    {5, {"LEFT", "A"}},
    {10, {"LEFT", "A", "X"}},
    {4, {"LEFT", "X"}},
    {1, {"LEFT", "A", "X"}},
    {1, {"B", "LEFT", "A", "X"}},
    {18, {"B", "LEFT", "A"}},
    {70, {"B", "LEFT"}},
  }, ROOM_FARM, ROOM_SPEEDWAY)

  local left = false
  for frame = 0, G.FTS_LEAVE_FRAMES - 1 do
    local st = session.state
    if st.room_id == ROOM_SPEEDWAY or st.room_id ~= ROOM_FARM then
      left = true
      break
    end
    if knockback.is_knockback(st) then
      knockback.escape_kb(session, label, "LEFT", {stop_room_id = ROOM_SPEEDWAY})
    else
      local x, y, vy = st.samus_x, st.samus_y, st.velocity_y
      if on_door_sill(st) or (x <= G.FTS_DOOR_X + 15 and y_band(st, G.FTS_DOOR_Y)) then
        local phase = frame % 14
        if phase < 4 then
          session:hold(1, {"LEFT", "X"}, label .. "_door_shot")
        elseif phase < 11 then
          session:hold(1, {"B", "LEFT"}, label .. "_door_push")
        else
          session:hold(1, {"LEFT"}, label .. "_door_walk")
        end
      elseif x > G.FTS_MID_HOP_X then
        local phase = frame % 18
        if phase < 10 then
          session:hold(1, {"B", "LEFT"}, label .. "_rx_run")
        elseif phase < 15 then
          session:hold(1, {"B", "LEFT", "A"}, label .. "_rx_hop")
        else
          session:hold(1, {"LEFT", "X"}, label .. "_rx_shot")
        end
      elseif vy ~= 0 then
        session:hold(1, {"B", "LEFT"}, label .. "_rx_air")
      elseif y > G.FTS_DOOR_Y[2] + 20 then
        session:hold(1, {"LEFT", "A"}, label .. "_rx_up")
      else
        local phase = frame % 16
        if phase < 8 then
          session:hold(1, {"B", "LEFT"}, label .. "_rx_lrun")
        elseif phase < 12 then
          session:hold(1, {"LEFT", "X"}, label .. "_rx_lshot")
        else
          session:hold(1, {"B", "LEFT", "A"}, label .. "_rx_lhop")
        end
      end
    end
  end

  if session.state.room_id ~= ROOM_SPEEDWAY then
    for frame = 0, G.FTS_DOOR_FRAMES - 1 do
      local st = session.state
      if st.room_id == ROOM_SPEEDWAY or st.room_id ~= ROOM_FARM then
        break
      end
      if knockback.is_knockback(st) then
        knockback.escape_kb(session, label, "LEFT", {stop_room_id = ROOM_SPEEDWAY})
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

  if session.state.room_id ~= ROOM_SPEEDWAY then
    local state = session.state
    error(string.format(
      "%s: Speedway door missed; room=0x%04X pose=%d xy=(%d,%d) frames=%d",
      label, state.room_id, state.pose, state.samus_x, state.samus_y,
      session.frame - start
    ))
  end

  return G.wait_ordinary_room(session, ROOM_SPEEDWAY, G.FTS_SPEEDWAY_SETTLE, label)
end

return M
