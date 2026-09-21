-- K4.10 Double Chamber → Wave Beam PLM collect.
-- Lua 5.1. Session: step / hold / wait_until / span.

local G = require("wave.geometry")
local gate = require("wave.double_gate")
local knockback = require("skills.knockback")
local walljump = require("skills.walljump")

local ROOM_DC = G.ROOM_DOUBLE_CHAMBER
local ROOM_WAVE = G.ROOM_WAVE
local STANDING = G.STANDING_POSES

local M = {}

local function missiles_and_runway(session, label)
  G.unmorph(session)
  G.select_weapon(session, 0)
  local mis0 = session.state.missiles or 0
  for _ = 1, 200 do
    local state = session.state
    if state.room_id ~= ROOM_DC then
      return
    end
    if state.samus_y > G.DC_LEDGE_Y_MAX + 20 then
      session:hold(1, {"LEFT", "A"}, label .. "_ledge_recover")
    elseif (state.missiles or 0) > mis0 and G.dc_on_missile_ledge(state) then
      break
    elseif state.samus_x >= G.DC_MISSILE_X and G.dc_on_missile_ledge(state) then
      break
    else
      session:hold(1, {"RIGHT"}, label .. "_to_missiles")
    end
  end
  for _ = 1, 520 do
    local state = session.state
    if state.room_id ~= ROOM_DC then
      return
    end
    if state.samus_y > G.DC_LEDGE_Y_MAX + 20 then
      session:hold(1, {"LEFT", "A"}, label .. "_plm_free_recover")
    elseif state.samus_x >= 510 and G.dc_on_missile_ledge(state) then
      break
    else
      session:hold(1, {"RIGHT", "B"}, label .. "_plm_free")
    end
  end
  for _ = 1, 280 do
    local state = session.state
    if state.room_id ~= ROOM_DC then
      return
    end
    if state.samus_y > G.DC_LEDGE_Y_MAX + 20 then
      session:hold(1, {"RIGHT", "A"}, label .. "_runway_recover")
    elseif state.samus_x <= G.DC_RUNWAY_X and G.dc_on_missile_ledge(state) then
      break
    else
      session:hold(1, {"LEFT"}, label .. "_runway_back")
    end
  end
  session:hold(10, {"RIGHT"}, label .. "_runway_face")
  session:hold(8, {}, label .. "_runway_settle")
end

local function ledge_dash_and_launch(session, label)
  for _ = 1, 220 do
    local state = session.state
    if state.room_id ~= ROOM_DC then
      return
    end
    if G.dc_on_sill(state) or (state.samus_x >= G.DC_DOOR_X and state.samus_y < G.DC_DOOR_Y_MAX) then
      return
    end
    if state.samus_y > G.DC_LEDGE_Y_MAX + 20 then
      session:hold(1, {"LEFT", "A"}, label .. "_dash_recover")
    elseif state.samus_x >= G.DC_EDGE_X and G.dc_on_missile_ledge(state) then
      break
    else
      session:hold(1, {"RIGHT", "B"}, label .. "_ledge_dash")
    end
  end

  local did_wj = false
  local left_follow = 0
  for _ = 1, 280 do
    local state = session.state
    if state.room_id ~= ROOM_DC or state.room_id == ROOM_WAVE then
      return
    end
    if G.dc_on_sill(state) or (state.samus_x >= G.DC_DOOR_X and state.samus_y < G.DC_DOOR_Y_MAX
        and state.velocity_y == 0) then
      return
    end
    if state.samus_y > 280 and state.velocity_y == 0 then
      return
    end
    if state.samus_y > 320 then
      return
    end
    local x, y = state.samus_x, state.samus_y
    local on_ledge = G.dc_on_missile_ledge(state)
    if on_ledge and x < G.DC_EDGE_X then
      session:hold(1, {"RIGHT", "B"}, label .. "_ledge_dash")
    elseif on_ledge then
      session:hold(1, {"RIGHT", "B", "A"}, label .. "_ledge_launch")
    else
      local at_wall = x >= 915 and state.velocity_x == 0 and y < 280
      if at_wall and not did_wj and y <= 260 then
        walljump.walljump_once(session, G.DC_WJ, {reason = label .. "_door_wj"})
        did_wj = true
        left_follow = G.DC_WJ_LEFT_FOLLOW
      elseif left_follow > 0 then
        left_follow = left_follow - 1
        session:hold(1, {"LEFT", "B", "A"}, label .. "_wj_left")
      elseif did_wj then
        session:hold(1, {"RIGHT", "B", "A"}, label .. "_sill_arc")
      elseif y <= 200 then
        session:hold(1, {"RIGHT", "B", "A"}, label .. "_high_air")
      else
        session:hold(1, {"RIGHT", "B", "A"}, label .. "_mid_air")
      end
    end
  end
end

local function super_door_push(session, label)
  G.select_weapon(session, 2)
  for frame = 0, 399 do
    local state = session.state
    if state.room_id == ROOM_WAVE or state.room_id ~= ROOM_DC then
      return
    end
    if knockback.is_knockback(state) then
      knockback.escape_kb(session, label, "RIGHT", {stop_room_id = ROOM_WAVE})
    elseif state.pose == 137 or state.pose == 138 then
      G.unmorph(session)
    else
      local x, y = state.samus_x, state.samus_y
      if x < G.DC_DOOR_X - 40 or y > G.DC_DOOR_Y_MAX + 40 then
        return
      end
      if state.velocity_y == 0 then
        local phase = frame % 36
        if phase < 3 then
          session:hold(1, {"RIGHT", "X"}, label .. "_super")
        elseif phase < 12 then
          session:hold(1, {}, label .. "_super_fuse")
        else
          session:hold(1, {"RIGHT", "B"}, label .. "_door_push")
        end
      else
        session:hold(1, {"RIGHT"}, label .. "_door_air")
      end
    end
  end
end

local function to_wave_door(session, label)
  if session.state.room_id ~= ROOM_DC or session.state.room_id == ROOM_WAVE then
    return
  end
  if G.dc_on_sill(session.state) or (session.state.samus_x >= G.DC_DOOR_X
      and session.state.samus_y < G.DC_DOOR_Y_MAX) then
    super_door_push(session, label)
    return
  end
  if session.state.samus_x < G.DC_PAST_GATE_X and session.state.samus_y < 220 then
    session:hold(1, {"RIGHT"}, label .. "_past_nudge")
  end
  if session.state.samus_y <= G.DC_LEDGE_Y_MAX + 40 or session.state.samus_x < 650 then
    missiles_and_runway(session, label)
    if session.state.room_id ~= ROOM_DC then
      return
    end
    ledge_dash_and_launch(session, label)
  end
  if session.state.room_id == ROOM_DC and session.state.samus_x >= G.DC_DOOR_X - 20
      and session.state.samus_y < G.DC_DOOR_Y_MAX then
    super_door_push(session, label)
    return
  end
  for _ = 1, 180 do
    local state = session.state
    if state.room_id == ROOM_WAVE or state.room_id ~= ROOM_DC then
      return
    end
    if state.samus_y > 280 then
      return
    end
    if state.samus_x >= G.DC_DOOR_X - 20 and state.samus_y < G.DC_DOOR_Y_MAX then
      super_door_push(session, label)
      return
    end
    if knockback.is_knockback(state) then
      knockback.escape_kb(session, label, "RIGHT", {stop_room_id = ROOM_WAVE})
    elseif state.pose == 137 or state.pose == 138 then
      G.unmorph(session)
    elseif state.samus_y <= 200 then
      session:hold(1, {"RIGHT", "B", "A"}, label .. "_sill_seek")
    else
      session:hold(1, {"RIGHT", "A"}, label .. "_sill_up")
    end
  end
end

local function wave_collect_plm(session, label)
  G.require_room(session, ROOM_WAVE, label)
  if G.has_wave(session.state) then
    return session.state
  end
  G.unmorph(session)
  G.select_weapon(session, 0)
  for _ = 1, 30 do
    local state = session:hold(1, {}, label .. "_stand")
    if state.velocity_y == 0 and STANDING[state.pose] then
      break
    end
    if state.pose == 137 or state.pose == 138 or state.pose == 39 or state.pose == 40 then
      session:hold(1, {"UP"}, label .. "_unmorph")
    end
  end
  local got = false
  for frame = 0, 499 do
    local state = session.state
    if G.has_wave(state) then
      got = true
      break
    end
    if state.room_id ~= ROOM_WAVE then
      error(string.format("%s: left Wave Room during collect; room=0x%04X", label, state.room_id))
    end
    if state.pose == 137 or state.pose == 138 then
      session:hold(8, {"UP"}, label .. "_unmorph")
    elseif state.samus_x < 160 then
      local phase = frame % 20
      if phase < 8 then
        session:hold(1, {"RIGHT", "B", "A"}, label .. "_chozo_hop")
      elseif phase < 14 then
        session:hold(1, {"RIGHT", "B"}, label .. "_chozo_run")
      else
        session:hold(1, {"RIGHT", "X"}, label .. "_chozo_shot")
      end
    elseif frame % 10 == 0 then
      session:hold(1, {"X"}, label .. "_plm_shot")
    else
      session:hold(1, {"RIGHT"}, label .. "_plm_walk")
    end
  end
  if not G.has_wave(session.state) then
    local state = session.state
    error(string.format(
      "%s: Wave PLM not collected; beams=0x%04X pose=%d xy=(%d,%d)",
      label, state.collected_beams or 0, state.pose, state.samus_x, state.samus_y
    ))
  end
  session:hold(80, {}, label .. "_fanfare")
  G.unmorph(session)
  for _ = 1, 40 do
    local state = session:hold(1, {}, label .. "_post_stand")
    if state.velocity_y == 0 and STANDING[state.pose] then
      break
    end
  end
  return session.state
end

function M.play_double_chamber_to_wave(session)
  local label = "double_chamber_to_wave"
  G.require_room(session, ROOM_DC, label)
  local start = session.frame
  if G.has_wave(session.state) and session.state.room_id == ROOM_WAVE then
    return session.state
  end
  if session.state.room_id == ROOM_DC then
    gate.dc_hop_to_gate_zone(session, label)
  end
  if session.state.room_id == ROOM_DC and session.state.samus_x < G.DC_PAST_GATE_X then
    gate.dc_open_blue_gate(session, label)
  end
  if session.state.room_id == ROOM_DC then
    to_wave_door(session, label)
  end
  if session.state.room_id ~= ROOM_WAVE then
    local state = session.state
    error(string.format(
      "%s: Wave door missed; room=0x%04X pose=%d xy=(%d,%d) frames=%d",
      label, state.room_id, state.pose, state.samus_x, state.samus_y,
      session.frame - start
    ))
  end
  G.wait_ordinary_room(session, ROOM_WAVE, G.DC_WAVE_SETTLE, label)
  local state = wave_collect_plm(session, label)
  if not G.has_wave(state) then
    error(string.format("%s: finished without Wave bit; beams=0x%04X", label, state.collected_beams or 0))
  end
  return state
end

M.WAVE_BEAM_MASK = G.WAVE_BEAM_MASK
return M
