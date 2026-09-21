-- K4 Frog Speedway → Frog Save return (rr-05dp). Needs Speed (Boost Blocks).
-- Lua 5.1. Session: step / hold / wait_until / span.

local G = require("wave.geometry")
local knockback = require("skills.knockback")
local shinespark = require("skills.shinespark")

local ROOM_SPEEDWAY = G.ROOM_FROG_SPEEDWAY
local ROOM_FROG = G.ROOM_FROG_SAVE
local LEDGE = {
  [1] = true, [2] = true, [9] = true, [10] = true, [11] = true, [12] = true,
  [25] = true, [26] = true, [27] = true, [28] = true, [37] = true, [38] = true,
  [137] = true, [138] = true,
}

local M = {}
M.shinespark = shinespark

local function y_band(state, band)
  return band[1] <= state.samus_y and state.samus_y <= band[2]
end

local function on_door_sill(state)
  return state.room_id == ROOM_SPEEDWAY
    and state.samus_x <= G.STF_DOOR_X + 12
    and y_band(state, G.STF_DOOR_Y)
    and state.velocity_y == 0
end

function M.play_speedway_to_frog_save(session)
  local label = "speedway_to_frog_save"
  G.require_room(session, ROOM_SPEEDWAY, label)
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
    {12, {"LEFT"}},
    {280, {"B", "LEFT"}},
    {20, {"LEFT"}},
    {16, {"LEFT", "X"}},
    {8, {"LEFT"}},
    {40, {"B", "LEFT"}},
    {20, {"LEFT"}},
    {12, {"B", "LEFT", "X"}},
  }, ROOM_SPEEDWAY, ROOM_FROG)

  local min_x = session.state.samus_x
  for frame = 0, G.STF_LEAVE_FRAMES - 1 do
    local st = session.state
    if st.room_id == ROOM_FROG or st.room_id ~= ROOM_SPEEDWAY then
      break
    end
    if knockback.is_knockback(st) then
      knockback.escape_kb(session, label, "LEFT", {stop_room_id = ROOM_FROG})
    else
      local x, y = st.samus_x, st.samus_y
      if x < min_x then
        min_x = x
      end
      if on_door_sill(st) or (x <= G.STF_DOOR_X + 20 and y_band(st, G.STF_DOOR_Y)) then
        local phase = frame % 14
        if phase < 4 then
          session:hold(1, {"LEFT", "X"}, label .. "_door_shot")
        elseif phase < 11 then
          session:hold(1, {"B", "LEFT"}, label .. "_door_push")
        else
          session:hold(1, {"LEFT"}, label .. "_door_walk")
        end
      elseif st.velocity_y ~= 0 then
        session:hold(1, {"B", "LEFT"}, label .. "_rx_air")
      elseif y > G.STF_DOOR_Y[2] + 20 then
        session:hold(1, {"LEFT", "A"}, label .. "_rx_up")
      else
        local phase = frame % 16
        if phase < 12 then
          session:hold(1, {"B", "LEFT"}, label .. "_rx_run")
        elseif phase < 14 then
          session:hold(1, {"LEFT", "X"}, label .. "_rx_shot")
        else
          session:hold(1, {"B", "LEFT", "X"}, label .. "_rx_dash_shot")
        end
      end
    end
  end

  if session.state.room_id ~= ROOM_FROG then
    for frame = 0, G.STF_DOOR_FRAMES - 1 do
      local st = session.state
      if st.room_id == ROOM_FROG or st.room_id ~= ROOM_SPEEDWAY then
        break
      end
      if knockback.is_knockback(st) then
        knockback.escape_kb(session, label, "LEFT", {stop_room_id = ROOM_FROG})
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

  if session.state.room_id ~= ROOM_FROG then
    local state = session.state
    local stall = ""
    if min_x >= 780 and state.samus_x >= 780 then
      stall = " (boost-block stall; no Speed charge?)"
    end
    error(string.format(
      "%s: Frog Save door missed; room=0x%04X pose=%d xy=(%d,%d) frames=%d min_x=%d%s",
      label, state.room_id, state.pose, state.samus_x, state.samus_y,
      session.frame - start, min_x, stall
    ))
  end

  return G.wait_ordinary_room(session, ROOM_FROG, G.STF_FROG_SETTLE, label)
end

return M
