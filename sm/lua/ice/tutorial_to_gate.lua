-- Ice Tutorial → Ice Gate return (K5 hop 2). Partial RLE + morph tunnel + freeze gap.
-- Lua 5.1. Session: step / hold / wait_until / span.

local G = require("ice.geometry")
local knockback = require("skills.knockback")

local ROOM_TUT = G.ROOM_ICE_TUTORIAL
local ROOM_GATE = G.ROOM_ICE_GATE
local CROUCH_MORPH = {
  [31] = true, [37] = true, [38] = true, [39] = true, [40] = true, [41] = true,
  [42] = true, [49] = true, [50] = true, [55] = true, [61] = true, [65] = true,
  [137] = true, [138] = true,
}
local STAND = {[1] = true, [2] = true, [9] = true, [10] = true, [11] = true}

local M = {}

local function ensure_beam(session)
  G.unmorph(session)
  if session.state.selected_item ~= 0 then
    G.select_weapon(session, 0)
  end
end

local function kb(session, label)
  if not knockback.is_knockback(session.state) then
    return false
  end
  knockback.escape_knockback_spin(session, {
    prefer_dir = "RIGHT",
    run_frames = 2,
    spin_frames = 10,
    label = label,
    ensure_beam = true,
    break_on_motion_clear = true,
  })
  return true
end

local function stand_up(session, label, frames)
  frames = frames or 28
  for _ = 1, frames do
    local st = session.state
    if st.room_id ~= ROOM_TUT then
      return
    end
    if STAND[st.pose] and st.velocity_y == 0 then
      return
    end
    if CROUCH_MORPH[st.pose] then
      session:hold(1, {"UP"}, label .. "_up")
    else
      session:hold(1, {}, label .. "_wait")
    end
  end
end

local function land_pin(session, label)
  for _ = 1, 40 do
    local st = session.state
    if st.room_id ~= ROOM_TUT then
      return
    end
    if kb(session, label .. "_kb") then
      -- continue
    elseif CROUCH_MORPH[st.pose] then
      session:hold(1, {"UP"}, label .. "_up")
    elseif st.velocity_y == 0 and (STAND[st.pose] or st.pose == 164 or st.pose == 166
        or st.pose == 75 or st.pose == 77) then
      break
    else
      session:hold(1, {}, label .. "_land")
    end
  end
  stand_up(session, label .. "_stand")
  ensure_beam(session)
end

local function rle_to_mid(session, label)
  G.play_script(session, G.TUTORIAL_MID_RLE, label .. "_mid_rle", ROOM_TUT, function(state)
    if state.room_id == ROOM_GATE then
      return true
    end
    local x, y = state.samus_x, state.samus_y
    return x >= 208 and 128 <= y and y <= 152 and state.velocity_y == 0
  end, "break")
  stand_up(session, label .. "_mid_stand")
end

local function morph_tunnel(session, label)
  if session.state.room_id ~= ROOM_TUT then
    return
  end
  stand_up(session, label .. "_pre")
  session:hold(13, {"A"}, label .. "_jump")
  session:hold(3, {"DOWN", "A"}, label .. "_da")
  session:hold(5, {"DOWN"}, label .. "_d1")
  session:hold(3, {}, label .. "_release")
  session:hold(7, {"DOWN"}, label .. "_d2")
  for _ = 1, 55 do
    local st = session.state
    if st.room_id ~= ROOM_TUT then
      return
    end
    if st.samus_x >= 295 then
      break
    end
    session:hold(1, {"RIGHT"}, label .. "_roll")
  end
  stand_up(session, label .. "_unmorph")
end

local function gap_and_door(session, label)
  if session.state.room_id ~= ROOM_TUT then
    return
  end
  for _ = 1, 4 do
    if session.state.room_id ~= ROOM_TUT then
      return
    end
    session:hold(3, {"X"}, label .. "_freeze")
    session:hold(4, {}, label .. "_freeze_wait")
  end
  session:hold(6, {}, label .. "_freeze_settle")
  session:hold(8, {"RIGHT", "B"}, label .. "_align")
  session:hold(3, {"A"}, label .. "_gap_a")
  session:hold(60, {"RIGHT", "B", "A"}, label .. "_gap_spin")

  for frame = 0, 359 do
    local st = session.state
    if st.room_id == ROOM_GATE or st.room_id ~= ROOM_TUT then
      return
    end
    if kb(session, label .. "_door_kb") then
      -- continue
    elseif CROUCH_MORPH[st.pose] then
      session:hold(1, {"UP"}, label .. "_door_up")
    elseif st.samus_y > 160 then
      session:hold(2, {"A"}, label .. "_shelf_a")
      session:hold(16, {"RIGHT", "B", "A"}, label .. "_shelf_hop")
    elseif st.samus_x < G.TUTORIAL_DOOR_X - 30 then
      local phase = frame % 18
      if phase < 10 then
        session:hold(1, {"RIGHT", "B"}, label .. "_approach")
      elseif phase < 14 then
        session:hold(1, {"RIGHT", "B", "A"}, label .. "_approach_hop")
      else
        session:hold(1, {"RIGHT", "X"}, label .. "_approach_shot")
      end
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

function M.play_ice_tutorial_to_gate(session)
  local label = "ice_tutorial_to_gate"
  G.require_room(session, ROOM_TUT, label)
  local start = session.frame
  ensure_beam(session)
  land_pin(session, label .. "_pin")

  if session.state.room_id == ROOM_GATE then
    return G.wait_ordinary_room(session, ROOM_GATE, G.GATE_RETURN_SETTLE, label)
  end

  for attempt = 0, 1 do
    if session.state.room_id == ROOM_GATE or session.state.room_id ~= ROOM_TUT then
      break
    end
    if session.frame - start > G.TUTORIAL_TO_GATE_FRAMES then
      break
    end
    local x, y = session.state.samus_x, session.state.samus_y
    if x < 200 then
      rle_to_mid(session, label .. "_a" .. attempt)
      x, y = session.state.samus_x, session.state.samus_y
    end
    if session.state.room_id == ROOM_TUT and 200 <= x and x < 290 and y < 160 then
      morph_tunnel(session, label .. "_a" .. attempt)
    end
    if session.state.room_id == ROOM_TUT and session.state.samus_x >= 270 then
      gap_and_door(session, label .. "_a" .. attempt)
    elseif session.state.room_id == ROOM_TUT and session.state.samus_x >= 200 then
      morph_tunnel(session, label .. "_a" .. attempt .. "_retry")
      if session.state.room_id == ROOM_TUT then
        gap_and_door(session, label .. "_a" .. attempt .. "_gap")
      end
    end
  end

  if session.state.room_id ~= ROOM_GATE then
    local st = session.state
    error(string.format(
      "%s: Gate door missed; room=0x%04X pose=%d xy=(%d,%d) frames=%d",
      label, st.room_id, st.pose, st.samus_x, st.samus_y, session.frame - start
    ))
  end
  return G.wait_ordinary_room(session, ROOM_GATE, G.GATE_RETURN_SETTLE, label)
end

return M
