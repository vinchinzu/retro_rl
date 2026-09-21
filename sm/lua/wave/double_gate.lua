-- K4.10 Double Chamber blue-gate hop + human open (Kamer seat).
-- Lua 5.1. Session: step / hold / wait_until / span.

local G = require("wave.geometry")
local knockback = require("skills.knockback")

local ROOM_DC = G.ROOM_DOUBLE_CHAMBER
local ROOM_WAVE = G.ROOM_WAVE
local STANDING = G.STANDING_POSES

local M = {}

function M.dc_hop_to_gate_zone(session, label)
  if session.state.samus_y <= 170 and session.state.samus_x < 61 then
    for _ = 1, 40 do
      local state = session.state
      if state.samus_x >= 61 and state.velocity_y == 0 then
        break
      end
      if state.samus_y > 220 then
        break
      end
      session:hold(1, {}, label .. "_kamer_phase")
    end
  end

  local pose = session.state.pose
  if G.is_morph(pose) or pose == 39 or pose == 40 or pose == 137 or pose == 138 then
    G.unmorph(session)
  end
  if session.state.selected_item ~= 0 then
    G.select_weapon(session, 0)
  end

  for _ = 1, 20 do
    local state = session:hold(1, {}, label .. "_top_stand")
    if state.velocity_y == 0 and STANDING[state.pose] then
      break
    end
  end

  for frame = 0, 159 do
    local state = session.state
    if state.room_id ~= ROOM_DC then
      return
    end
    if state.samus_x >= 210 and state.velocity_y == 0 and state.samus_y < 200 then
      break
    end
    if knockback.is_knockback(state) then
      knockback.escape_kb(session, label, "RIGHT", {stop_room_id = ROOM_WAVE})
    elseif state.pose == 137 or state.pose == 138 then
      G.unmorph(session)
    else
      local phase = frame % 30
      if phase < 4 then
        session:hold(1, {"RIGHT", "X"}, label .. "_hop_shot")
      elseif phase < 12 then
        session:hold(1, {"RIGHT", "B", "A"}, label .. "_hop_spin")
      elseif phase < 22 then
        session:hold(1, {"RIGHT", "B"}, label .. "_hop_run")
      else
        session:hold(1, {"RIGHT"}, label .. "_hop_walk")
      end
    end
  end

  for frame = 0, 279 do
    local state = session.state
    if state.room_id ~= ROOM_DC then
      return
    end
    if G.DC_GATE_SEAT_X[1] <= state.samus_x and state.samus_x <= G.DC_GATE_SEAT_X[2]
        and state.samus_y < G.DC_GATE_SEAT_Y_MAX and state.velocity_y == 0 then
      return
    end
    if state.samus_x >= G.DC_PAST_GATE_X and state.samus_y < 220 then
      return
    end
    if state.samus_y > 360 and state.velocity_y == 0 then
      return
    end
    if knockback.is_knockback(state) then
      knockback.escape_kb(session, label, "RIGHT", {stop_room_id = ROOM_WAVE})
    elseif state.pose == 137 or state.pose == 138 then
      G.unmorph(session)
    elseif state.samus_x >= G.DC_GATE_SEAT_X[1] - 20
        and state.samus_y < G.DC_GATE_SEAT_Y_MAX and state.velocity_y == 0 then
      if state.samus_x < G.DC_GATE_SEAT_X[1] then
        session:hold(1, {"RIGHT"}, label .. "_seat_in")
      elseif state.samus_x > G.DC_GATE_SEAT_X[2] then
        session:hold(1, {"LEFT"}, label .. "_seat_back")
      else
        session:hold(1, {}, label .. "_seat_brake")
      end
    else
      local phase = frame % 28
      if phase < 16 then
        session:hold(1, {"RIGHT", "B", "A"}, label .. "_gate_spin")
      else
        session:hold(1, {"RIGHT", "B"}, label .. "_gate_run")
      end
    end
  end
end

local function wait_kamer_open_seat(session, label)
  local x_lo, x_hi = G.DC_GATE_OPEN_SEAT_X[1], G.DC_GATE_OPEN_SEAT_X[2]
  for _ = 1, 800 do
    local state = session.state
    if state.room_id ~= ROOM_DC then
      return false
    end
    if state.samus_x >= G.DC_PAST_GATE_X and state.samus_y < 220 then
      return true
    end
    if state.samus_y > 360 and state.velocity_y == 0 then
      return false
    end
    if knockback.is_knockback(state) then
      knockback.escape_kb(session, label, "RIGHT", {stop_room_id = ROOM_WAVE})
    elseif state.pose == 137 or state.pose == 138 then
      G.unmorph(session)
    else
      local seated = state.velocity_y == 0
        and state.samus_y <= G.DC_GATE_OPEN_SEAT_Y
        and x_lo <= state.samus_x and state.samus_x <= x_hi
        and STANDING[state.pose]
      if seated then
        return true
      end
      if state.samus_x < x_lo then
        session:hold(1, {"RIGHT"}, label .. "_seat_r")
      elseif state.samus_x > x_hi then
        session:hold(1, {"LEFT"}, label .. "_seat_l")
      else
        session:hold(1, {}, label .. "_seat_wait")
      end
    end
  end
  return false
end

local function select_missiles_for_open(session, label)
  if session.state.selected_item ~= 1 then
    G.select_weapon(session, 1)
  else
    session:hold(1, {}, label .. "_select_missiles_pad")
    session:hold(25, {}, label .. "_select_missiles_settle")
  end
end

local function floor_recover_to_gate(session, label)
  G.unmorph(session)
  if session.state.selected_item ~= 0 then
    G.select_weapon(session, 0)
  end
  for _ = 1, 900 do
    local state = session.state
    if state.room_id ~= ROOM_DC then
      return false
    end
    if state.samus_y <= 200 and 300 <= state.samus_x and state.samus_x <= 450
        and state.velocity_y == 0 then
      return true
    end
    local pose = state.pose
    if knockback.is_knockback(state) or pose == 20 or pose == 83 or pose == 84 then
      knockback.escape_kb(session, label, "RIGHT", {stop_room_id = ROOM_WAVE})
    else
      local x, y = state.samus_x, state.samus_y
      if y >= 360 and state.velocity_y == 0 then
        if x < 320 then
          session:hold(1, {"RIGHT", "B"}, label .. "_floor_r")
        elseif x > 380 then
          session:hold(1, {"LEFT", "B"}, label .. "_floor_l")
        else
          session:hold(1, {"RIGHT", "B", "A"}, label .. "_floor_hj")
        end
      elseif y > 200 then
        if x < 340 then
          session:hold(1, {"RIGHT", "B", "A"}, label .. "_climb_r")
        elseif x > 400 then
          session:hold(1, {"LEFT", "B", "A"}, label .. "_climb_l")
        else
          session:hold(1, {"RIGHT", "A"}, label .. "_climb_up")
        end
      elseif x < 370 then
        session:hold(1, {"RIGHT"}, label .. "_reseat_r")
      elseif x > 390 then
        session:hold(1, {"LEFT"}, label .. "_reseat_l")
      else
        session:hold(1, {}, label .. "_reseat")
      end
    end
  end
  return session.state.samus_y <= 200
    and 300 <= session.state.samus_x and session.state.samus_x <= 450
end

function M.dc_open_blue_gate(session, label)
  for _attempt = 1, 3 do
    G.unmorph(session)
    if session.state.samus_y > 300 then
      if not floor_recover_to_gate(session, label) then
        return
      end
    end
    if not wait_kamer_open_seat(session, label) then
      return
    end
    if session.state.samus_x >= G.DC_PAST_GATE_X then
      return
    end
    select_missiles_for_open(session, label)
    session:hold(8, {}, label .. "_seat_settle")
    if session.state.samus_x >= G.DC_PAST_GATE_X then
      return
    end

    local aborted = false
    local hit_abort = false
    local health0 = session.state.health or 0
    G.play_script(session, G.GATE_OPEN_RLE, label .. "_human_open", ROOM_DC, function(state)
      if state.samus_x >= G.DC_PAST_GATE_X and state.samus_y < 220 then
        return true
      end
      if state.samus_y > 360 and state.velocity_y == 0 then
        aborted = true
        return true
      end
      local hp = state.health or 0
      local hit = knockback.is_knockback(state)
        or state.pose == 20 or state.pose == 83 or state.pose == 84
        or hp < health0
      if hp >= health0 then
        health0 = hp
      end
      if hit then
        hit_abort = true
        aborted = true
        return true
      end
      return false
    end, "ignore")

    if session.state.room_id ~= ROOM_DC then
      return
    end
    if session.state.samus_x >= G.DC_PAST_GATE_X and session.state.samus_y < 220 then
      return
    end
    if hit_abort then
      knockback.escape_kb(session, label, "RIGHT", {stop_room_id = ROOM_WAVE})
    end
    if not aborted then
      local still = true
      for _ = 1, 40 do
        local state = session.state
        if state.room_id ~= ROOM_DC then
          return
        end
        if state.samus_x >= G.DC_PAST_GATE_X and state.samus_y < 220 then
          return
        end
        if state.samus_y > 300 then
          still = false
          break
        end
        if knockback.is_knockback(state) or state.pose == 20 or state.pose == 83 or state.pose == 84 then
          knockback.escape_kb(session, label, "RIGHT", {stop_room_id = ROOM_WAVE})
          still = false
          break
        end
        session:hold(1, {"RIGHT"}, label .. "_past_walk")
      end
      if still then
        return
      end
    end
  end
end

return M
