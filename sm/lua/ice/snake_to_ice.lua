-- Pure Ice Snake → Ice Beam PLM collect (room 0xA890, beam bit 0x0002).
-- Lua 5.1. Session: step / hold / wait_until / span.

local G = require("ice.geometry")
local knockback = require("skills.knockback")

local ROOM_SNAKE, ROOM_ICE = G.ROOM_ICE_SNAKE, G.ROOM_ICE
local STANDING = G.STANDING_POSES
local LEDGE = {
  [1] = true, [2] = true, [9] = true, [10] = true, [37] = true, [38] = true,
}
for k, v in pairs(STANDING) do
  LEDGE[k] = v
end

local M = {}
M.ICE_BEAM_MASK = G.ICE_BEAM_MASK

function M.settle_ground(session, label, max_frames)
  max_frames = max_frames or 40
  for _ = 1, max_frames do
    local st = session.state
    if st.velocity_y == 0 and LEDGE[st.pose] and st.door_transition == 0 then
      return
    end
    if knockback.is_knockback(st) then
      knockback.escape_knockback_spin(session, {
        prefer_dir = "LEFT", run_frames = 2, spin_frames = 10,
        label = label .. "_kb", ensure_beam = true, break_on_motion_clear = true,
      })
    elseif st.pose == 31 or st.pose == 39 or st.pose == 40 or st.pose == 41
        or st.pose == 42 or st.pose == 65 then
      session:hold(1, {"UP"}, label .. "_unmorph")
    else
      session:hold(1, {}, label .. "_settle")
    end
  end
end

local function y_band(state, band)
  return band[1] <= state.samus_y and state.samus_y <= band[2]
end

function M.snake_platform_climb(session, label)
  G.require_room(session, ROOM_SNAKE, label)
  G.unmorph(session)
  if session.state.selected_item ~= 0 then
    G.select_weapon(session, 0)
  end
  if session.state.samus_y > G.SNAKE_L1_Y[2] then
    session:hold(8, {}, label .. "_l1_idle")
    session:hold(11, {"LEFT"}, label .. "_l1_walk")
    session:hold(11, {"LEFT", "B"}, label .. "_l1_run")
    session:hold(34, {"LEFT", "B", "A"}, label .. "_l1_jump")
    session:hold(12, {"LEFT", "B"}, label .. "_l1_coast")
    M.settle_ground(session, label .. "_l1")
  end
  if session.state.samus_y > G.SNAKE_L2_Y[2] then
    session:hold(3, {"B"}, label .. "_l2_b")
    session:hold(4, {"B", "A"}, label .. "_l2_ba")
    session:hold(23, {"RIGHT", "B", "A"}, label .. "_l2_jump")
    session:hold(27, {"RIGHT", "B"}, label .. "_l2_coast")
    M.settle_ground(session, label .. "_l2")
  end
  if session.state.samus_y > G.SNAKE_L3_Y[2] then
    session:hold(3, {"LEFT", "A"}, label .. "_l3_a")
    session:hold(21, {"LEFT", "A"}, label .. "_l3_jump")
    session:hold(12, {"LEFT"}, label .. "_l3_coast")
    session:hold(8, {"LEFT", "B"}, label .. "_l3_run")
    session:hold(12, {"LEFT"}, label .. "_l3_walk")
    M.settle_ground(session, label .. "_l3")
  end
  if session.state.samus_y > G.SNAKE_L4_Y[2] then
    session:hold(5, {"A"}, label .. "_l4_a")
    session:hold(12, {"RIGHT", "A"}, label .. "_l4_ra")
    session:hold(12, {"RIGHT", "B", "A"}, label .. "_l4_spin")
    session:hold(25, {"RIGHT"}, label .. "_l4_coast")
    for _ = 1, 50 do
      local st = session.state
      if y_band(st, G.SNAKE_L4_Y) and st.velocity_y == 0 and LEDGE[st.pose] then
        break
      end
      session:hold(1, {}, label .. "_l4_land")
    end
    M.settle_ground(session, label .. "_l4")
  end
  if session.state.samus_y > G.SNAKE_L5_Y[2] then
    session:hold(12, {"RIGHT"}, label .. "_l5_right")
    session:hold(28, {"LEFT", "A"}, label .. "_l5_jump")
    session:hold(15, {"LEFT"}, label .. "_l5_coast")
    M.settle_ground(session, label .. "_l5")
  end
  if session.state.samus_y > G.SNAKE_L6_Y[2] then
    session:hold(18, {"LEFT"}, label .. "_l6_wall")
    session:hold(4, {"A"}, label .. "_l6_a")
    session:hold(18, {"RIGHT", "A"}, label .. "_l6_ra")
    session:hold(10, {"RIGHT", "B", "A"}, label .. "_l6_spin")
    session:hold(15, {"RIGHT"}, label .. "_l6_coast")
    M.settle_ground(session, label .. "_l6")
  end
  if session.state.samus_y > G.SNAKE_L7_Y[2] then
    session:hold(20, {"RIGHT"}, label .. "_l7_right")
    session:hold(24, {"LEFT", "A"}, label .. "_l7_jump")
    session:hold(14, {"LEFT"}, label .. "_l7_coast")
    M.settle_ground(session, label .. "_l7")
  end
  if session.state.samus_y > G.SNAKE_TOP_Y[2] then
    for _ = 1, 30 do
      if session.state.samus_x <= 68 then break end
      session:hold(1, {"LEFT"}, label .. "_top_wall")
    end
    session:hold(3, {}, label .. "_top_pause")
    session:hold(5, {"A"}, label .. "_top_a")
    session:hold(20, {"RIGHT", "A"}, label .. "_top_ra")
    session:hold(12, {"RIGHT", "B", "A"}, label .. "_top_spin")
    session:hold(25, {"RIGHT"}, label .. "_top_coast")
    for _ = 1, 50 do
      if G.on_snake_top(session.state) then break end
      session:hold(1, {}, label .. "_top_land")
    end
    M.settle_ground(session, label .. "_top")
  end
  return session.state
end

local MID_SHELF_RLE = {
  {1, {"B"}}, {6, {"B", "RIGHT", "A"}}, {5, {"B", "UP", "RIGHT", "A"}},
  {5, {"B", "UP", "A"}}, {3, {"B", "UP", "LEFT", "A"}}, {3, {"B", "LEFT", "A"}},
  {2, {"B", "UP", "LEFT", "A"}}, {1, {"B", "UP", "A"}}, {1, {"B", "UP", "RIGHT", "A"}},
  {19, {"B", "RIGHT", "A"}}, {3, {"B", "A"}}, {2, {"B"}}, {1, {"B", "DOWN"}},
  {5, {"DOWN"}}, {4, {}}, {4, {"DOWN", "RIGHT"}}, {52, {"RIGHT"}},
}

local function in_tunnel_grounded(state)
  if not G.in_ice_snake(state) then return false end
  if state.samus_x < G.SNAKE_TUNNEL_X_MIN then return false end
  if not (G.SNAKE_TUNNEL_Y[1] <= state.samus_y and state.samus_y <= G.SNAKE_TUNNEL_Y[2]) then
    return false
  end
  return state.velocity_y == 0
end

local function roll_tunnel_to_ice_door(session, label)
  if G.in_ice_snake(session.state) and in_tunnel_grounded(session.state)
      and not G.is_morph(session.state.pose) then
    pcall(G.ensure_morph, session)
  end
  for i = 0, G.SNAKE_TUNNEL_FRAMES - 1 do
    local st = session.state
    if st.room_id == ROOM_ICE or not G.in_ice_snake(st) then
      return st
    end
    if knockback.is_knockback(st) then
      session:hold(1, {"DOWN", "RIGHT"}, label .. "_kb_morph")
    else
      local x, y = st.samus_x, st.samus_y
      if y > 450 and x < G.SNAKE_TUNNEL_EXIT_X then
        return st
      end
      if x < G.SNAKE_TUNNEL_EXIT_X and G.SNAKE_TUNNEL_Y[1] - 5 <= y and y <= G.SNAKE_TUNNEL_Y[2] + 5 then
        if not G.is_morph(st.pose) then
          if st.velocity_y == 0 then
            pcall(G.ensure_morph, session)
          else
            session:hold(1, {"DOWN"}, label .. "_air_remorph")
          end
        else
          session:hold(1, {"RIGHT"}, label .. "_tunnel_roll")
        end
      elseif G.is_morph(st.pose) or st.pose == 39 or st.pose == 40 or st.pose == 41
          or st.pose == 42 or st.pose == 61 then
        session:hold(1, {"UP"}, label .. "_unmorph_exit")
      elseif x >= G.SNAKE_DOOR_X then
        local phase = i % 16
        if phase < 4 then
          session:hold(1, {"RIGHT", "X"}, label .. "_door_shot")
        elseif phase < 11 then
          session:hold(1, {"RIGHT", "B"}, label .. "_door_run")
        else
          session:hold(1, {"RIGHT", "B", "A"}, label .. "_door_hop")
        end
      else
        local phase = i % 18
        if phase < 10 then
          session:hold(1, {"RIGHT", "B", "A"}, label .. "_right_hop")
        elseif phase < 14 then
          session:hold(1, {"RIGHT", "B"}, label .. "_right_run")
        else
          session:hold(1, {"RIGHT", "X"}, label .. "_right_shot")
        end
      end
    end
  end
  return session.state
end

local function snake_top_to_right_column(session, label)
  for _ = 1, 50 do
    local st = session.state
    if not G.in_ice_snake(st) then return end
    if st.samus_x >= 200 or st.samus_y > 180 then break end
    session:hold(1, {"RIGHT", "B"}, label .. "_top_cross")
  end
  session:hold(10, {"RIGHT", "A"}, label .. "_top_jump")
  local shelf_shots = 0
  for _ = 1, 120 do
    local st = session.state
    if not G.in_ice_snake(st) then return end
    local y = st.samus_y
    if y >= 250 and st.velocity_y == 0 then break end
    if 145 <= y and y <= 180 and st.velocity_y == 0 then
      if G.is_morph(st.pose) or st.pose == 39 or st.pose == 40 or st.pose == 41 or st.pose == 42 then
        session:hold(1, {"UP"}, label .. "_shelf_up")
      elseif shelf_shots < 6 then
        session:hold(1, {"DOWN", "X"}, label .. "_shelf_shot")
        session:hold(2, {}, label .. "_shelf_wait")
        shelf_shots = shelf_shots + 1
      else
        session:hold(1, {"RIGHT"}, label .. "_shelf_off")
      end
    elseif y < 250 then
      session:hold(1, {}, label .. "_fall")
    else
      session:hold(1, {}, label .. "_right_land")
    end
  end
  G.unmorph(session)
  M.settle_ground(session, label .. "_right_plat")
end

local function snake_top_to_tunnel(session, label)
  snake_top_to_right_column(session, label)
  if session.state.room_id == ROOM_ICE then
    return session.state
  end
  for attempt = 0, 3 do
    if session.state.room_id == ROOM_ICE or not G.in_ice_snake(session.state) then
      return session.state
    end
    if knockback.is_knockback(session.state) then
      knockback.escape_knockback_spin(session, {
        prefer_dir = "RIGHT", run_frames = 2, spin_frames = 10,
        label = label .. "_kb_a" .. attempt, ensure_beam = true, break_on_motion_clear = true,
      })
    end
    local y, x = session.state.samus_y, session.state.samus_x
    if in_tunnel_grounded(session.state) then
      local st = roll_tunnel_to_ice_door(session, label .. "_a" .. attempt)
      if st.room_id == ROOM_ICE then return st end
    elseif 140 <= y and y <= 200 and x >= 190 and session.state.velocity_y == 0 then
      G.unmorph(session)
      for _ = 1, 8 do
        session:hold(1, {"DOWN", "X"}, label .. "_shelf_clear")
        session:hold(1, {}, label .. "_shelf_clear")
      end
      session:hold(6, {"RIGHT"}, label .. "_shelf_off")
      for _ = 1, 40 do
        if session.state.samus_y >= 250 then break end
        session:hold(1, {}, label .. "_shelf_fall")
      end
      G.unmorph(session)
      M.settle_ground(session, label .. "_shelf_land")
    elseif 240 <= y and y <= 320 and x >= 180 then
      G.unmorph(session)
      M.settle_ground(session, label .. "_plat")
      for _ = 1, 40 do
        local st = session.state
        if not G.in_ice_snake(st) then return st end
        if st.samus_y > 300 then break end
        if st.samus_x <= 208 and st.velocity_y == 0 then break end
        session:hold(1, {"LEFT"}, label .. "_edge")
      end
      if session.state.samus_y <= 300 and session.state.velocity_y == 0 then
        pcall(G.ensure_morph, session)
      end
      for _ = 1, 20 do
        local st = session.state
        if not G.in_ice_snake(st) or st.samus_y > 280 then break end
        if G.is_morph(st.pose) then
          session:hold(1, {"LEFT"}, label .. "_roll_off")
        else
          session:hold(1, {"LEFT", "DOWN"}, label .. "_nudge_off")
        end
      end
      for _ = 1, 100 do
        local st = session.state
        if st.room_id == ROOM_ICE or not G.in_ice_snake(st) then return st end
        if in_tunnel_grounded(st) then
          if not G.is_morph(st.pose) then pcall(G.ensure_morph, session) end
          break
        end
        if st.samus_y > 430 and st.velocity_y == 0 then break end
        if st.samus_x < 200 then
          session:hold(1, {"RIGHT"}, label .. "_fall_r")
        elseif st.samus_x > 210 then
          session:hold(1, {"LEFT"}, label .. "_fall_l")
        else
          session:hold(1, {(st.samus_y >= 340) and "RIGHT" or "LEFT"}, label .. "_fall_in")
        end
      end
      if in_tunnel_grounded(session.state) then
        local st = roll_tunnel_to_ice_door(session, label .. "_a" .. attempt)
        if st.room_id == ROOM_ICE then return st end
      end
    elseif G.on_snake_mid_shelf(session.state)
        or (G.SNAKE_MID_SHELF_Y[1] - 20 <= y and y <= G.SNAKE_MID_SHELF_Y[2] + 30 and x >= 180) then
      G.unmorph(session)
      for _ = 1, 50 do
        local st = session.state
        if not G.in_ice_snake(st) then return st end
        if st.samus_y < G.SNAKE_MID_SHELF_Y[1] - 30 then break end
        if 194 <= st.samus_x and st.samus_x <= 200 and st.velocity_y == 0 then break end
        session:hold(1, {(st.samus_x > 200) and "LEFT" or "RIGHT"}, label .. "_align")
      end
      M.settle_ground(session, label .. "_pre_jump")
      session:hold(4, {}, label .. "_pre_idle")
      for i = 1, #MID_SHELF_RLE do
        local st = session.state
        if st.room_id == ROOM_ICE or not G.in_ice_snake(st) then return st end
        session:hold(MID_SHELF_RLE[i][1], MID_SHELF_RLE[i][2], label .. "_rle")
      end
      if in_tunnel_grounded(session.state) then
        local st = roll_tunnel_to_ice_door(session, label .. "_a" .. attempt)
        if st.room_id == ROOM_ICE then return st end
      end
    elseif y > G.SNAKE_MID_SHELF_Y[2] then
      G.unmorph(session)
      session:hold(25, {(x > 210) and "LEFT" or "RIGHT", "B", "A"}, label .. "_floor_up")
      session:hold(15, {"A"}, label .. "_floor_up2")
      M.settle_ground(session, label .. "_floor")
    else
      session:hold(1, {"RIGHT", "A"}, label .. "_to_plat")
    end
  end
  return session.state
end

local function ice_collect_plm(session, label)
  G.require_room(session, ROOM_ICE, label)
  if G.has_ice(session.state) then
    return session.state
  end
  G.unmorph(session)
  if session.state.selected_item ~= 0 then
    G.select_weapon(session, 0)
  end
  for _ = 1, 30 do
    local st = session:hold(1, {}, label .. "_stand")
    if st.velocity_y == 0 and STANDING[st.pose] then break end
  end
  local got = false
  for frame = 0, G.SNAKE_ICE_COLLECT_FRAMES - 1 do
    local st = session.state
    if G.has_ice(st) then
      got = true
      break
    end
    if st.room_id ~= ROOM_ICE then
      error(string.format("%s: left Ice during collect; room=0x%04X", label, st.room_id))
    end
    local p = st.pose
    if p == 137 or p == 138 or p == 39 or p == 40 or p == 41 or p == 42 then
      session:hold(1, {"UP"}, label .. "_unmorph")
    elseif st.samus_x < G.ICE_PLM_X - 10 then
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
  if not G.has_ice(session.state) then
    error(string.format("%s: Ice PLM not collected; beams=0x%04X xy=(%d,%d)",
      label, session.state.collected_beams or 0, session.state.samus_x, session.state.samus_y))
  end
  session:hold(80, {}, label .. "_fanfare")
  G.unmorph(session)
  for _ = 1, 40 do
    local st = session:hold(1, {}, label .. "_post_stand")
    if st.velocity_y == 0 and STANDING[st.pose] then break end
  end
  return session.state
end

function M.play_ice_snake_to_ice(session)
  local label = "ice_snake_to_ice"
  G.require_room(session, ROOM_SNAKE, label)
  local start = session.frame
  if G.has_ice(session.state) and session.state.room_id == ROOM_ICE then
    return session.state
  end
  if knockback.is_knockback(session.state) then
    knockback.escape_knockback_spin(session, {
      prefer_dir = "LEFT", run_frames = 3, spin_frames = 12,
      label = label .. "_kb0", ensure_beam = true, break_on_motion_clear = true,
    })
  end
  if session.state.room_id == ROOM_SNAKE then
    M.snake_platform_climb(session, label)
  end
  if session.state.room_id == ROOM_SNAKE and not G.has_ice(session.state) then
    if not G.on_snake_top(session.state) and session.state.samus_y > G.SNAKE_TOP_Y[2] then
      for _ = 1, math.min(400, math.floor(G.SNAKE_CLIMB_FRAMES / 4)) do
        if G.on_snake_top(session.state) or session.state.room_id ~= ROOM_SNAKE then
          break
        end
        session:hold(1, {"LEFT", "A"}, label .. "_climb_push")
      end
    end
    snake_top_to_tunnel(session, label)
  end
  if session.state.room_id ~= ROOM_ICE then
    local st = session.state
    error(string.format(
      "%s: Ice door missed; room=0x%04X pose=%d xy=(%d,%d) frames=%d (prefer 2WJ climb + right-column tunnel)",
      label, st.room_id, st.pose, st.samus_x, st.samus_y, session.frame - start
    ))
  end
  G.wait_ordinary_room(session, ROOM_ICE, G.ICE_ROOM_SETTLE, label)
  local state = ice_collect_plm(session, label)
  if not G.has_ice(state) then
    error(string.format("%s: finished without Ice bit; beams=0x%04X", label, state.collected_beams or 0))
  end
  return state
end

-- Historical Python names used by snake_to_tutorial.
M._settle_ground = M.settle_ground
M._snake_platform_climb = M.snake_platform_climb

return M
