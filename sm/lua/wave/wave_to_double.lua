-- K4 Wave Beam Room → Double Chamber return (rr-pd0i).
-- Lua 5.1. Session: step / hold / wait_until / span.

local G = require("wave.geometry")
local knockback = require("skills.knockback")

local ROOM_WAVE = G.ROOM_WAVE
local ROOM_DOUBLE = G.ROOM_DOUBLE_CHAMBER
local STANDING = G.STANDING_POSES

local M = {}

function M.play_wave_to_double_chamber(session)
  local label = "wave_to_double_chamber"
  G.require_room(session, ROOM_WAVE, label)
  if not G.has_wave(session.state) then
    error(string.format(
      "%s: Wave not collected (beams=0x%04X; need bit 0x%04X)",
      label, session.state.collected_beams or 0, G.WAVE_BEAM_MASK
    ))
  end

  G.unmorph(session)
  G.select_weapon(session, 0)
  for _ = 1, 40 do
    local state = session:hold(1, {}, label .. "_stand")
    if state.pose == 137 or state.pose == 138 or state.pose == 39 or state.pose == 40 then
      session:hold(1, {"UP"}, label .. "_unmorph")
    elseif state.velocity_y == 0 and STANDING[state.pose] then
      break
    end
  end

  local hit = false
  for frame = 0, G.WAVE_LEAVE_FRAMES - 1 do
    local state = session.state
    if state.room_id == ROOM_DOUBLE then
      hit = true
      break
    end
    if state.room_id ~= ROOM_WAVE then
      hit = true
      break
    end
    if knockback.is_knockback(state) then
      knockback.escape_kb(session, label, "LEFT", {stop_room_id = ROOM_DOUBLE})
    elseif state.pose == 137 or state.pose == 138 or state.pose == 39 or state.pose == 40 then
      session:hold(6, {"UP"}, label .. "_unmorph")
    else
      local x = state.samus_x
      if x <= G.WAVE_DOOR_X and state.velocity_y == 0 then
        local phase = frame % 16
        if phase < 4 then
          session:hold(1, {"LEFT", "X"}, label .. "_door_shot")
        elseif phase < 12 then
          session:hold(1, {"LEFT", "B"}, label .. "_door_push")
        else
          session:hold(1, {"LEFT", "B", "A"}, label .. "_door_spin")
        end
      else
        local phase = frame % 20
        if phase < 10 then
          session:hold(1, {"LEFT", "B"}, label .. "_run")
        elseif phase < 14 then
          session:hold(1, {"LEFT", "B", "A"}, label .. "_hop")
        elseif phase < 17 then
          session:hold(1, {"LEFT", "X"}, label .. "_shot")
        else
          session:hold(1, {"LEFT"}, label .. "_walk")
        end
      end
    end
  end
  if session.state.room_id ~= ROOM_DOUBLE then
    local state = session.state
    error(string.format(
      "%s: left Wave door missed; room=0x%04X pose=%d xy=(%d,%d) door_transition=%d",
      label, state.room_id, state.pose, state.samus_x, state.samus_y,
      state.door_transition or 0
    ))
  end

  return G.wait_ordinary_room(session, ROOM_DOUBLE, G.WAVE_DOUBLE_SETTLE, label)
end

return M
