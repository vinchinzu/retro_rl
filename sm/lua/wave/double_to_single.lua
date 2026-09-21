-- K4 Double Chamber → Single Chamber return (rr-qpkd).
-- Lua 5.1. Session: step / hold / wait_until / span.

local G = require("wave.geometry")
local knockback = require("skills.knockback")

local ROOM_DC = G.ROOM_DOUBLE_CHAMBER
local ROOM_SC = G.ROOM_SINGLE_CHAMBER
local STANDING = G.STANDING_POSES

local M = {}

local function on_floor(state)
  return state.room_id == ROOM_DC and state.samus_y >= G.DTS_FLOOR_Y_MIN and state.velocity_y == 0
end

local function on_door_sill(state)
  return state.room_id == ROOM_DC and state.samus_x <= G.DTS_DOOR_X + 15
    and G.DTS_DOOR_Y[1] <= state.samus_y and state.samus_y <= G.DTS_DOOR_Y[2]
    and state.velocity_y == 0
end

local function ledge_left_and_hop(session, label)
  G.unmorph(session)
  G.select_weapon(session, 0)
  for _ = 1, 25 do
    local state = session:hold(1, {}, label .. "_ledge_stand")
    if state.velocity_y == 0 and STANDING[state.pose] then
      break
    end
  end
  local hopping = false
  for frame = 0, G.DTS_LEDGE_FRAMES - 1 do
    local state = session.state
    if state.room_id ~= ROOM_DC then
      return
    end
    if state.samus_y > G.DTS_LEDGE_Y_MAX + 10 then
      return
    end
    if knockback.is_knockback(state) then
      knockback.escape_kb(session, label, "LEFT", {stop_room_id = ROOM_SC})
    elseif G.is_morph(state.pose) or state.pose == 137 or state.pose == 138
        or state.pose == 39 or state.pose == 40 then
      session:hold(4, {"UP"}, label .. "_ledge_unmorph")
    else
      local x, y = state.samus_x, state.samus_y
      local grounded = state.velocity_y == 0 and y <= G.DTS_LEDGE_Y_MAX
      if hopping or (grounded and x <= G.DTS_HOP_LAUNCH_X) then
        hopping = true
        session:hold(1, {"LEFT", "B", "A"}, label .. "_ledge_hop")
      else
        local phase = frame % 16
        if phase < 10 then
          session:hold(1, {"LEFT", "B"}, label .. "_ledge_run")
        elseif phase < 13 then
          session:hold(1, {"LEFT"}, label .. "_ledge_walk")
        else
          session:hold(1, {"LEFT", "X"}, label .. "_ledge_shot")
        end
      end
    end
  end
end

local function super_column_drop(session, label)
  for frame = 0, G.DTS_DROP_FRAMES - 1 do
    local state = session.state
    if state.room_id ~= ROOM_DC or on_floor(state) then
      return
    end
    if knockback.is_knockback(state) then
      knockback.escape_kb(session, label, "LEFT", {stop_room_id = ROOM_SC})
    else
      local x, y, vy = state.samus_x, state.samus_y, state.velocity_y
      if y <= G.DTS_LEDGE_Y_MAX and vy == 0 then
        if x > G.DTS_HOP_LAUNCH_X then
          session:hold(1, {"LEFT", "B"}, label .. "_reledge")
        else
          session:hold(1, {"LEFT", "B", "A"}, label .. "_rehop")
        end
      elseif G.DTS_MID_Y[1] <= y and y <= G.DTS_MID_Y[2] then
        if vy == 0 and not G.is_morph(state.pose) then
          local ok, err = pcall(G.ensure_morph, session, 3)
          if not ok then
            session:hold(1, {"DOWN"}, label .. "_mid_crouch")
          end
        elseif G.is_morph(state.pose) then
          if x > G.DTS_MID_X[2] then
            session:hold(1, {"LEFT"}, label .. "_mid_morph_l")
          elseif x < G.DTS_MID_X[1] then
            session:hold(1, {"RIGHT"}, label .. "_mid_morph_r")
          else
            local phase = frame % 10
            if phase < 4 then
              session:hold(1, {"DOWN"}, label .. "_mid_morph_d")
            elseif phase < 7 then
              session:hold(1, {"LEFT"}, label .. "_mid_morph_nudge")
            else
              session:hold(1, {}, label .. "_mid_morph_fall")
            end
          end
        elseif x > 820 then
          session:hold(1, {"LEFT"}, label .. "_mid_air_l")
        elseif frame % 8 < 3 then
          session:hold(1, {"DOWN"}, label .. "_mid_air_d")
        else
          session:hold(1, {"LEFT"}, label .. "_mid_air")
        end
      elseif y < G.DTS_FLOOR_Y_MIN then
        if not G.is_morph(state.pose) and vy ~= 0 and frame % 12 < 3 then
          session:hold(1, {"DOWN"}, label .. "_fall_morph")
        elseif x > 800 then
          session:hold(1, {"LEFT"}, label .. "_fall_l")
        elseif x < 740 then
          session:hold(1, {"RIGHT"}, label .. "_fall_r")
        else
          session:hold(1, {}, label .. "_fall")
        end
      else
        session:hold(1, {}, label .. "_land_wait")
      end
    end
  end
end

local function floor_left_to_gap(session, label)
  for frame = 0, G.DTS_FLOOR_FRAMES - 1 do
    local state = session.state
    if state.room_id == ROOM_SC or state.room_id ~= ROOM_DC then
      return
    end
    if knockback.is_knockback(state) then
      knockback.escape_kb(session, label, "LEFT", {stop_room_id = ROOM_SC})
    else
      local x, y = state.samus_x, state.samus_y
      if x <= G.DTS_GAP_LAUNCH_X and y >= G.DTS_FLOOR_Y_MIN - 40 then
        return
      end
      if on_door_sill(state) then
        return
      end
      local in_tunnel = G.DTS_MORPH_TUNNEL_X[1] <= x and x <= G.DTS_MORPH_TUNNEL_X[2]
      if in_tunnel and not G.is_morph(state.pose) and state.velocity_y == 0 then
        local ok = pcall(G.ensure_morph, session, 3)
        if not ok then
          session:hold(1, {"DOWN"}, label .. "_tun_morph")
        end
      elseif G.is_morph(state.pose) then
        session:hold(1, {"LEFT"}, label .. "_tun_roll")
      elseif y > G.DTS_FLOOR_Y[2] + 20 then
        session:hold(1, {"LEFT", "A"}, label .. "_floor_up")
      else
        local phase = frame % 18
        if phase < 10 then
          session:hold(1, {"LEFT", "B"}, label .. "_floor_run")
        elseif phase < 14 then
          session:hold(1, {"LEFT", "B", "A"}, label .. "_floor_hop")
        else
          session:hold(1, {"LEFT"}, label .. "_floor_walk")
        end
      end
    end
  end
end

local function gap_and_left_door(session, label)
  for frame = 0, G.DTS_DOOR_FRAMES - 1 do
    local state = session.state
    if state.room_id == ROOM_SC or state.room_id ~= ROOM_DC then
      return
    end
    if knockback.is_knockback(state) then
      knockback.escape_kb(session, label, "LEFT", {stop_room_id = ROOM_SC})
    else
      local x, y, vy = state.samus_x, state.samus_y, state.velocity_y
      if on_door_sill(state) or x <= G.DTS_DOOR_X then
        local phase = frame % 16
        if phase < 4 then
          session:hold(1, {"LEFT", "X"}, label .. "_door_shot")
        elseif phase < 12 then
          session:hold(1, {"LEFT", "B"}, label .. "_door_push")
        else
          session:hold(1, {"LEFT", "B", "A"}, label .. "_door_spin")
        end
      elseif x > G.DTS_DOOR_X + 20 then
        if y >= G.DTS_FLOOR_Y_MIN - 50 and vy == 0 and x <= G.DTS_GAP_LAUNCH_X + 40 then
          session:hold(1, {"LEFT", "B", "A"}, label .. "_gap_hop")
        elseif vy ~= 0 then
          session:hold(1, {"LEFT"}, label .. "_gap_air")
        else
          local phase = frame % 16
          if phase < 8 then
            session:hold(1, {"LEFT", "B"}, label .. "_gap_run")
          elseif phase < 12 then
            session:hold(1, {"LEFT", "B", "A"}, label .. "_gap_spin")
          else
            session:hold(1, {"LEFT"}, label .. "_gap_walk")
          end
        end
      else
        session:hold(1, {"LEFT"}, label .. "_door_nudge")
      end
    end
  end
end

function M.play_double_to_single_chamber(session)
  local label = "double_to_single_chamber"
  G.require_room(session, ROOM_DC, label)
  local start = session.frame
  if session.state.samus_y <= G.DTS_LEDGE_Y_MAX + 20 then
    ledge_left_and_hop(session, label)
  end
  if session.state.room_id == ROOM_DC and not on_floor(session.state)
      and session.state.samus_y > G.DTS_LEDGE_Y_MAX then
    super_column_drop(session, label)
  end
  if session.state.room_id == ROOM_DC and session.state.samus_y < G.DTS_FLOOR_Y_MIN then
    super_column_drop(session, label)
  end
  if session.state.room_id == ROOM_DC and session.state.samus_x > G.DTS_GAP_LAUNCH_X then
    floor_left_to_gap(session, label)
  end
  if session.state.room_id == ROOM_DC then
    gap_and_left_door(session, label)
  end
  if session.state.room_id ~= ROOM_SC then
    local state = session.state
    error(string.format(
      "%s: Single Chamber door missed; room=0x%04X pose=%d xy=(%d,%d) frames=%d",
      label, state.room_id, state.pose, state.samus_x, state.samus_y,
      session.frame - start
    ))
  end
  return G.wait_ordinary_room(session, ROOM_SC, G.DTS_SINGLE_SETTLE, label)
end

return M
