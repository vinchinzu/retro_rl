-- Ice Gate → Business return (K5 hop 3). RLE drop + tunnel roll + Super door.
-- Lua 5.1. Session: step / hold / wait_until / span.

local G = require("ice.geometry")
local knockback = require("skills.knockback")

local ROOM_GATE = G.ROOM_ICE_GATE
local ROOM_BUSINESS = G.ROOM_BUSINESS
local MORPH = {
  [27] = true, [28] = true, [29] = true, [30] = true, [31] = true,
  [37] = true, [38] = true, [39] = true, [40] = true, [41] = true,
  [42] = true, [43] = true, [45] = true, [49] = true, [50] = true,
  [55] = true, [65] = true, [137] = true, [138] = true,
}
local STAND = {[1] = true, [2] = true, [9] = true, [10] = true, [11] = true}

local M = {}

local function kb(session, label, prefer)
  prefer = prefer or "RIGHT"
  if not knockback.is_knockback(session.state) then
    return false
  end
  knockback.escape_knockback_spin(session, {
    prefer_dir = prefer,
    run_frames = 2,
    spin_frames = 10,
    label = label,
    ensure_beam = true,
    break_on_motion_clear = true,
  })
  return true
end

local function ensure_beam(session)
  if session.state.selected_item ~= 0 then
    G.select_weapon(session, 0)
  end
end

local function land_mid_top(session, label)
  for _ = 1, 48 do
    local st = session.state
    if st.room_id ~= ROOM_GATE then
      return
    end
    if kb(session, label .. "_kb") then
      -- continue
    elseif MORPH[st.pose] and st.pose ~= 41 and st.pose ~= 45 and st.pose ~= 49 and st.pose ~= 55 then
      session:hold(1, {"UP"}, label .. "_up")
    elseif st.velocity_y == 0 and (STAND[st.pose] or G.LEDGE_POSES[st.pose]
        or st.pose == 81 or st.pose == 164 or st.pose == 166) then
      break
    else
      session:hold(1, {}, label .. "_land")
    end
  end
  ensure_beam(session)
end

local function rle_drop_and_roll(session, label)
  G.play_script(session, G.GATE_TO_BUSINESS_RLE, label .. "_rle", ROOM_GATE, function(state)
    if state.room_id == ROOM_BUSINESS then
      return true
    end
    local x, y = state.samus_x, state.samus_y
    if x >= G.GATE_SUPER_DOOR_X - 20 and y >= 620 then
      return true
    end
    if x >= 1720 and y >= 640 and state.velocity_y == 0 then
      return true
    end
    return false
  end, "break")
end

local function closed_loop_tunnel_roll(session, label)
  if session.state.room_id ~= ROOM_GATE then
    return
  end
  for frame = 0, 699 do
    local st = session.state
    if st.room_id == ROOM_BUSINESS or st.room_id ~= ROOM_GATE then
      return
    end
    if kb(session, label .. "_roll_kb") then
      -- continue
    else
      local x, y, pose = st.samus_x, st.samus_y, st.pose
      if x >= G.GATE_SUPER_DOOR_X - 30 and y >= 600 then
        return
      end
      if x >= 1720 and y >= 620 then
        return
      end
      if (STAND[pose] or G.LEDGE_POSES[pose])
          and G.GATE_TUNNEL_Y[1] - 30 <= y and y <= G.GATE_TUNNEL_Y[1] + 5
          and x < 1600 then
        session:hold(3, {"DOWN"}, label .. "_pipe_morph")
        session:hold(8, {"RIGHT"}, label .. "_pipe_into")
      elseif not MORPH[pose] and y < 650 and x < 1700 then
        session:hold(2, {"DOWN"}, label .. "_remorph")
      elseif y > G.GATE_TUNNEL_Y[2] + 50 and x < 1600 then
        if frame % 20 < 6 then
          session:hold(1, {"RIGHT", "A"}, label .. "_climb_hop")
        else
          session:hold(1, {"RIGHT"}, label .. "_climb_r")
        end
      else
        session:hold(1, {"RIGHT"}, label .. "_roll")
      end
    end
  end
end

local function door_to_business(session, label)
  if session.state.room_id ~= ROOM_GATE then
    return
  end
  for frame = 0, 479 do
    local st = session.state
    if st.room_id == ROOM_BUSINESS or st.room_id ~= ROOM_GATE then
      return
    end
    if kb(session, label .. "_door_kb") then
      -- continue
    else
      local pose, x, y = st.pose, st.samus_x, st.samus_y
      if MORPH[pose] and y >= 600 then
        session:hold(1, {"UP"}, label .. "_door_up")
      elseif x < G.GATE_SUPER_DOOR_X - 100 then
        session:hold(1, {"RIGHT", "B"}, label .. "_approach")
      elseif y < 620 and x >= G.GATE_SUPER_DOOR_X - 80 then
        session:hold(1, {"RIGHT"}, label .. "_floor_drop")
      else
        local phase = frame % 16
        if phase < 4 then
          session:hold(1, {"RIGHT", "X"}, label .. "_door_shot")
        elseif phase < 12 then
          session:hold(1, {"RIGHT", "B"}, label .. "_door_push")
        else
          session:hold(1, {"RIGHT", "B", "A"}, label .. "_door_spin")
        end
      end
    end
  end
end

function M.play_ice_gate_to_business(session)
  local label = "ice_gate_to_business"
  G.require_room(session, ROOM_GATE, label)
  local start = session.frame
  ensure_beam(session)
  land_mid_top(session, label .. "_pin")

  if session.state.room_id == ROOM_BUSINESS then
    return G.wait_ordinary_room(session, ROOM_BUSINESS, G.BUSINESS_RETURN_SETTLE, label)
  end

  for attempt = 0, 1 do
    if session.state.room_id == ROOM_BUSINESS or session.state.room_id ~= ROOM_GATE then
      break
    end
    if session.frame - start > G.GATE_TO_BUSINESS_FRAMES then
      break
    end
    local x, y = session.state.samus_x, session.state.samus_y
    if x < G.GATE_SUPER_DOOR_X - 50 and y < 700 then
      if attempt == 0 or y < G.GATE_TUNNEL_Y[1] + 40 then
        rle_drop_and_roll(session, label .. "_a" .. attempt)
      else
        closed_loop_tunnel_roll(session, label .. "_a" .. attempt)
      end
    end
    if session.state.room_id == ROOM_GATE and session.state.samus_x < 1720 then
      closed_loop_tunnel_roll(session, label .. "_a" .. attempt .. "_fb")
    end
    if session.state.room_id == ROOM_GATE then
      door_to_business(session, label .. "_a" .. attempt)
    end
  end

  if session.state.room_id ~= ROOM_BUSINESS then
    local st = session.state
    error(string.format(
      "%s: Business door missed; room=0x%04X pose=%d xy=(%d,%d) frames=%d",
      label, st.room_id, st.pose, st.samus_x, st.samus_y, session.frame - start
    ))
  end

  local state = G.wait_ordinary_room(session, ROOM_BUSINESS, G.BUSINESS_RETURN_SETTLE, label)
  G.unmorph(session)
  for _ = 1, 60 do
    local st = session:hold(1, {}, label .. "_stand")
    if st.room_id ~= ROOM_BUSINESS then
      break
    end
    local y = st.samus_y
    if st.velocity_y == 0 and (STAND[st.pose] or G.LEDGE_POSES[st.pose])
        and G.ICE_SUPER_Y_MIN - 40 <= y and y <= G.ICE_SUPER_Y_MAX + 40
        and st.door_transition == 0 then
      return st
    end
    if y > G.ICE_SUPER_Y_MAX + 80 and st.velocity_y == 0 then
      return st
    end
  end
  return state
end

return M
