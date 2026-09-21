-- Pure Ice Gate → Acid (left blue door). Needs Speed for Boost Blocks ~x1045.
-- Lua 5.1. Session: step / hold / wait_until / span.

local G = require("ice.geometry")
local knockback = require("skills.knockback")
local shinespark = require("skills.shinespark")
local geom = require("skills.geometry")

local ROOM_GATE = G.ROOM_ICE_GATE
local ROOM_ACID = G.ROOM_ICE_ACID
local STANDING = geom.STANDING_POSES or G.STANDING_POSES
local GATE_FLOOR_Y = {600, 720}
local MID_HOP_X = {820, 940}
local RUN_FRAMES = 1200
local ACID_SETTLE = 280
local SPEED_MASK = 0x2000

local M = {}
M.shinespark = shinespark

function M.play_ice_gate_to_acid(session)
  local label = "ice_gate_to_acid"
  G.require_room(session, ROOM_GATE, label)

  G.unmorph(session)
  if session.state.selected_item ~= 0 then
    G.select_weapon(session, 0)
  end

  local mid_hop_done = false
  local hit = false
  for frame = 0, RUN_FRAMES - 1 do
    local state = session.state
    if state.room_id == ROOM_ACID then
      hit = true
      break
    end
    if knockback.is_knockback(state) then
      knockback.escape_knockback_spin(session, {
        prefer_dir = "LEFT",
        run_frames = 3,
        spin_frames = 14,
        label = label .. "_kb",
        run_with = {"B", "X"},
        spin_with = {"B", "A"},
        ensure_beam = true,
        break_on_motion_clear = true,
      })
    else
      local x, y, pose = state.samus_x, state.samus_y, state.pose
      local grounded = state.velocity_y == 0 and (
        STANDING[pose] or pose == 37 or pose == 38 or pose == 9 or pose == 10
        or pose == 1 or pose == 2
      )
      if (pose == 31 or pose == 39 or pose == 40 or pose == 41 or pose == 42 or pose == 65)
          and (not (MID_HOP_X[1] <= x and x <= MID_HOP_X[2]) or mid_hop_done) then
        G.unmorph(session)
      elseif not mid_hop_done
          and MID_HOP_X[1] <= x and x <= MID_HOP_X[2]
          and GATE_FLOOR_Y[1] <= y and y <= GATE_FLOOR_Y[2] then
        if grounded and pose ~= 42 and pose ~= 40 then
          session:hold(4, {"DOWN"}, label .. "_mid_crouch")
        end
        for _ = 1, 16 do
          local st = session:hold(1, {"LEFT", "B", "A"}, label .. "_mid_hop")
          if st.room_id == ROOM_ACID or st.samus_x < MID_HOP_X[1] - 10 then
            break
          end
        end
        for _ = 1, 24 do
          local st = session:hold(1, {"LEFT", "B"}, label .. "_mid_coast")
          if st.room_id == ROOM_ACID or st.samus_x < 780 then
            break
          end
        end
        mid_hop_done = true
      elseif x <= 100 then
        local phase = frame % 14
        if phase < 3 then
          session:hold(1, {"LEFT", "X"}, label .. "_door_shot")
        else
          session:hold(1, {"LEFT", "B"}, label .. "_door_run")
        end
      else
        local phase = frame % 20
        if phase < 3 then
          session:hold(1, {"LEFT", "B", "X"}, label .. "_run_shot")
        else
          session:hold(1, {"LEFT", "B"}, label .. "_run")
        end
      end
    end
  end

  if session.state.room_id ~= ROOM_ACID then
    error(string.format(
      "%s: Acid Room missed: room=0x%04X xy=(%d,%d) (need Speed if stuck near x≈1045)",
      label, session.state.room_id, session.state.samus_x, session.state.samus_y
    ))
  end

  local state = G.wait_ordinary_room(session, ROOM_ACID, ACID_SETTLE, label)
  G.unmorph(session)
  for _ = 1, 40 do
    local st = session:hold(1, {}, label .. "_stand")
    if st.velocity_y == 0 and STANDING[st.pose] and st.door_transition == 0 then
      return st
    end
  end
  return state
end

return M
