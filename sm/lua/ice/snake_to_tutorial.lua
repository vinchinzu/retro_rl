-- Ice Snake mid-right → Ice Tutorial return (K5 hop 1).
-- Lua 5.1. Session: step / hold / wait_until / span.

local G = require("ice.geometry")
local snake = require("ice.snake_to_ice")
local knockback = require("skills.knockback")

local ROOM_SNAKE, ROOM_TUT = G.ROOM_ICE_SNAKE, G.ROOM_ICE_TUTORIAL
local FLOOR_ALIGN_X = 216

local M = {}

local function on_floor_band(state)
  if not G.in_ice_snake(state) then return false end
  local y = state.samus_y
  return G.SNAKE_HANDOFF_Y[1] <= y and y <= G.SNAKE_HANDOFF_Y[2] + 40
end

local function mid_right_to_floor(session, label)
  G.unmorph(session)
  if session.state.selected_item ~= 0 then
    G.select_weapon(session, 0)
  end
  for _ = 1, 120 do
    local st = session.state
    if not G.in_ice_snake(st) then return st end
    if st.samus_y >= 420 and st.samus_x <= 335 then break end
    if knockback.is_knockback(st) then
      knockback.escape_knockback_spin(session, {
        prefer_dir = "LEFT", run_frames = 2, spin_frames = 8,
        label = label .. "_shelf_kb", ensure_beam = true, break_on_motion_clear = true,
      })
    else
      session:hold(1, {"LEFT"}, label .. "_shelf_left")
    end
  end
  session:hold(3, {"A"}, label .. "_hop_a")
  session:hold(22, {"LEFT", "A"}, label .. "_hop_la")
  for _ = 1, 40 do
    local st = session.state
    if st.velocity_y == 0 and st.samus_y < 400 then break end
    session:hold(1, {"LEFT"}, label .. "_hop_coast")
  end
  for _ = 1, 6 do session:hold(1, {"DOWN"}, label .. "_morph1") end
  for _ = 1, 4 do session:hold(1, {}, label .. "_morph_pause") end
  for _ = 1, 6 do session:hold(1, {"DOWN"}, label .. "_morph2") end
  for _ = 1, 8 do session:hold(1, {}, label .. "_morph_settle") end
  for _ = 1, 80 do
    local st = session.state
    if not G.in_ice_snake(st) then return st end
    if st.samus_x < 200 or st.samus_y > 450 then break end
    session:hold(1, {"LEFT"}, label .. "_tunnel_roll")
  end
  local dropped = false
  for frame = 0, G.SNAKE_TO_TUTORIAL_DROP_FRAMES - 1 do
    local st = session.state
    if not G.in_ice_snake(st) then return st end
    if on_floor_band(st) and st.velocity_y == 0 then
      dropped = true
      break
    end
    if knockback.is_knockback(st) then
      knockback.escape_knockback_spin(session, {
        prefer_dir = "LEFT", run_frames = 2, spin_frames = 8,
        label = label .. "_fall_kb", ensure_beam = true, break_on_motion_clear = true,
      })
    else
      local y, pose = st.samus_y, st.pose
      if 500 <= y and y <= 560 and st.velocity_y == 0 then
        if not (G.is_morph(pose) or pose == 39 or pose == 40 or pose == 41 or pose == 42 or pose == 49 or pose == 50) then
          session:hold(1, {"DOWN"}, label .. "_midshelf_morph")
        elseif math.floor(frame / 8) % 2 == 0 then
          session:hold(1, {"RIGHT"}, label .. "_midshelf_r")
        else
          session:hold(1, {"LEFT"}, label .. "_midshelf_l")
        end
      elseif 400 <= y and y <= 500 and st.velocity_y == 0 then
        session:hold(1, {"DOWN"}, label .. "_mid_morph")
      else
        session:hold(1, {}, label .. "_fall")
      end
    end
  end
  if not on_floor_band(session.state) then
    error(string.format("%s: floor drop missed; xy=(%d,%d)", label, session.state.samus_x, session.state.samus_y))
  end
  G.unmorph(session)
  for _ = 1, 20 do session:hold(1, {"UP"}, label .. "_floor_unmorph") end
  snake.settle_ground(session, label .. "_floor")
  for _ = 1, 30 do
    local st = session.state
    if math.abs(st.samus_x - FLOOR_ALIGN_X) < 10 then break end
    if st.samus_x < FLOOR_ALIGN_X then
      session:hold(1, {"RIGHT"}, label .. "_floor_align")
    else
      session:hold(1, {"LEFT"}, label .. "_floor_align")
    end
  end
  snake.settle_ground(session, label .. "_floor_align")
  return session.state
end

local function climb_to_top(session, label)
  if session.state.selected_item ~= 0 then
    G.select_weapon(session, 0)
  end
  for attempt = 0, 3 do
    local st = session.state
    if not G.in_ice_snake(st) or G.on_snake_top(st) then
      return st
    end
    if st.samus_y <= G.SNAKE_TOP_Y[2] + 10 and st.velocity_y == 0 then
      snake.settle_ground(session, label .. "_near_top")
      if G.on_snake_top(session.state) or session.state.samus_y <= G.SNAKE_TOP_Y[2] + 20 then
        return session.state
      end
    end
    if session.state.samus_y > G.SNAKE_TOP_Y[2] then
      snake.snake_platform_climb(session, label .. "_a" .. attempt)
    end
    if G.on_snake_top(session.state) then
      return session.state
    end
  end
  if not G.on_snake_top(session.state) and session.state.samus_y > G.SNAKE_TOP_Y[2] then
    error(string.format("%s: climb missed top; xy=(%d,%d)", label, session.state.samus_x, session.state.samus_y))
  end
  return session.state
end

local function top_to_tutorial(session, label)
  G.unmorph(session)
  if session.state.selected_item ~= 0 then
    G.select_weapon(session, 0)
  end
  local hit = false
  for frame = 0, G.SNAKE_TOP_TO_TUTORIAL_FRAMES - 1 do
    local st = session.state
    if st.room_id == ROOM_TUT or not G.in_ice_snake(st) then
      hit = true
      break
    end
    if knockback.is_knockback(st) then
      knockback.escape_knockback_spin(session, {
        prefer_dir = "RIGHT", run_frames = 2, spin_frames = 10,
        label = label .. "_door_kb", ensure_beam = true, break_on_motion_clear = true,
      })
    elseif G.is_morph(st.pose) or st.pose == 39 or st.pose == 40 or st.pose == 41 or st.pose == 42 then
      session:hold(1, {"UP"}, label .. "_door_unmorph")
    elseif st.samus_y > G.SNAKE_TOP_Y[2] + 30 then
      session:hold(1, {"RIGHT", "A"}, label .. "_reclimb")
    elseif st.samus_x >= G.SNAKE_TUTORIAL_DOOR_X
        and G.SNAKE_TUTORIAL_DOOR_Y[1] <= st.samus_y and st.samus_y <= G.SNAKE_TUTORIAL_DOOR_Y[2] then
      local phase = frame % 16
      if phase < 4 then
        session:hold(1, {"RIGHT", "X"}, label .. "_door_shot")
      elseif phase < 11 then
        session:hold(1, {"RIGHT", "B"}, label .. "_door_push")
      else
        session:hold(1, {"RIGHT", "B", "A"}, label .. "_door_spin")
      end
    elseif st.samus_x < 100 then
      local phase = frame % 18
      if phase < 10 then
        session:hold(1, {"RIGHT", "B"}, label .. "_top_run")
      elseif phase < 14 then
        session:hold(1, {"RIGHT", "B", "A"}, label .. "_top_hop")
      else
        session:hold(1, {"RIGHT"}, label .. "_top_walk")
      end
    elseif st.samus_x < G.SNAKE_TUTORIAL_DOOR_X then
      local phase = frame % 18
      if phase < 8 then
        session:hold(1, {"RIGHT", "B"}, label .. "_approach")
      elseif phase < 12 then
        session:hold(1, {"RIGHT", "A"}, label .. "_approach_jump")
      elseif phase < 15 then
        session:hold(1, {"RIGHT", "X"}, label .. "_approach_shot")
      else
        session:hold(1, {"RIGHT"}, label .. "_approach_walk")
      end
    else
      session:hold(1, {"RIGHT"}, label .. "_door_nudge")
    end
  end
  if session.state.room_id ~= ROOM_TUT then
    error(string.format("%s: Tutorial door missed; room=0x%04X xy=(%d,%d)",
      label, session.state.room_id, session.state.samus_x, session.state.samus_y))
  end
  return G.wait_ordinary_room(session, ROOM_TUT, G.TUTORIAL_RETURN_SETTLE, label)
end

function M.play_ice_snake_to_tutorial(session)
  local label = "ice_snake_to_tutorial"
  G.require_room(session, ROOM_SNAKE, label)
  local start = session.frame
  if knockback.is_knockback(session.state) then
    knockback.escape_knockback_spin(session, {
      prefer_dir = "LEFT", run_frames = 3, spin_frames = 12,
      label = label .. "_kb0", ensure_beam = true, break_on_motion_clear = true,
    })
  end
  if G.on_snake_top(session.state) then
    return top_to_tutorial(session, label)
  end
  if session.state.samus_y < G.SNAKE_HANDOFF_Y[1] - 20 then
    mid_right_to_floor(session, label)
  end
  if session.state.room_id == ROOM_SNAKE and not G.on_snake_top(session.state) then
    climb_to_top(session, label)
  end
  if session.state.room_id == ROOM_TUT then
    return G.wait_ordinary_room(session, ROOM_TUT, G.TUTORIAL_RETURN_SETTLE, label)
  end
  if session.state.room_id ~= ROOM_SNAKE then
    error(string.format("%s: left Snake without Tutorial; room=0x%04X frames=%d",
      label, session.state.room_id, session.frame - start))
  end
  local state = top_to_tutorial(session, label)
  if state.room_id ~= ROOM_TUT then
    error(string.format("%s: finished without Tutorial; room=0x%04X", label, state.room_id))
  end
  return state
end

return M
