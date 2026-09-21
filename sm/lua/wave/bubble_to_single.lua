-- K4.8 Bubble Mountain → Single Chamber.
-- Lua 5.1. Session: step / hold / wait_until / span.

local G = require("wave.geometry")
local knockback = require("skills.knockback")

local ROOM_BUBBLE = G.ROOM_BUBBLE
local ROOM_SC = G.ROOM_SINGLE_CHAMBER
local STANDING = G.STANDING_POSES

local M = {}

local function top_walk_to_drop(session, label)
  G.unmorph(session)
  G.select_weapon(session, 0)
  for _ = 1, 30 do
    local state = session:hold(1, {}, label .. "_top_stand")
    if state.velocity_y == 0 and STANDING[state.pose] then
      break
    end
  end
  for frame = 0, G.BSC_TOP_WALK_FRAMES - 1 do
    local state = session.state
    if state.room_id ~= ROOM_BUBBLE then
      return
    end
    if state.samus_y > G.BSC_TOP_Y_MAX then
      return
    end
    if G.BSC_DROP_X[1] <= state.samus_x and state.samus_x <= G.BSC_DROP_X[2]
        and state.velocity_y == 0 then
      return
    end
    if knockback.is_knockback(state) then
      knockback.escape_kb(session, label, "LEFT", {stop_room_id = ROOM_SC})
    elseif state.samus_x < G.BSC_DROP_X[1] then
      session:hold(1, {"RIGHT"}, label .. "_top_r")
    elseif state.samus_x > G.BSC_DROP_X[2] then
      local phase = frame % 14
      if phase < 10 then
        session:hold(1, {"LEFT", "B"}, label .. "_top_run")
      else
        session:hold(1, {"LEFT"}, label .. "_top_walk")
      end
    else
      session:hold(1, {}, label .. "_top_seat")
    end
  end
end

local function drop_shaft(session, label)
  for frame = 0, G.BSC_DROP_FRAMES - 1 do
    local state = session.state
    if state.room_id ~= ROOM_BUBBLE then
      return
    end
    if state.samus_y >= G.BSC_FLOOR_Y and state.velocity_y == 0 then
      return
    end
    if knockback.is_knockback(state) then
      knockback.escape_kb(session, label, "LEFT", {stop_room_id = ROOM_SC})
    elseif state.velocity_y == 0 and state.samus_y <= G.BSC_TOP_Y_MAX then
      if state.samus_x > G.BSC_DROP_TARGET_X + 8 then
        session:hold(1, {"LEFT"}, label .. "_lip_left")
      elseif state.samus_x < G.BSC_DROP_X[1] then
        session:hold(1, {"RIGHT"}, label .. "_lip_right")
      else
        local phase = frame % 12
        if phase < 4 then
          session:hold(1, {"LEFT"}, label .. "_step_off")
        elseif phase < 7 then
          session:hold(1, {"A"}, label .. "_lip_hop")
        else
          session:hold(1, {}, label .. "_lip_wait")
        end
      end
    elseif state.samus_x > 400 then
      session:hold(1, {"LEFT"}, label .. "_fall_l")
    elseif state.samus_x < 350 then
      session:hold(1, {"RIGHT"}, label .. "_fall_r")
    else
      session:hold(1, {}, label .. "_fall")
    end
  end
end

local function nav_floor_to_door(session, label)
  G.unmorph(session)
  G.select_weapon(session, 0)
  for frame = 0, G.BSC_NAV_TO_DOOR_FRAMES - 1 do
    local state = session.state
    if state.room_id == ROOM_SC or state.room_id ~= ROOM_BUBBLE then
      return
    end
    if knockback.is_knockback(state) then
      knockback.escape_kb(session, label, "RIGHT", {stop_room_id = ROOM_SC})
    else
      local on_sill = state.samus_x >= G.BSC_DOOR_X
        and G.BSC_DOOR_Y[1] <= state.samus_y and state.samus_y <= G.BSC_DOOR_Y[2]
        and state.velocity_y == 0
      if on_sill then
        return
      end
      if state.samus_y < G.BSC_FLOOR_Y then
        if state.velocity_y == 0 and STANDING[state.pose] then
          if state.samus_x < 360 and state.samus_y < 300 then
            session:hold(1, {"LEFT", "A"}, label .. "_mid_hop")
          elseif state.samus_x < 420 then
            local phase = frame % 18
            if phase < 6 then
              session:hold(1, {"RIGHT", "B", "A"}, label .. "_mid_spin")
            elseif phase < 12 then
              session:hold(1, {"RIGHT", "B"}, label .. "_mid_run")
            else
              session:hold(1, {"RIGHT"}, label .. "_mid_walk")
            end
          else
            session:hold(1, {"RIGHT"}, label .. "_mid_drop")
          end
        elseif state.samus_x < 450 then
          session:hold(1, {"RIGHT"}, label .. "_air_r")
        else
          session:hold(1, {}, label .. "_air")
        end
      elseif state.samus_x < G.BSC_DOOR_X then
        local phase = frame % 22
        if phase < 6 then
          session:hold(1, {"RIGHT", "B", "A"}, label .. "_floor_hop")
        elseif phase < 14 then
          session:hold(1, {"RIGHT", "B"}, label .. "_floor_run")
        else
          session:hold(1, {"RIGHT"}, label .. "_floor_walk")
        end
      else
        session:hold(1, {"RIGHT"}, label .. "_sill_nudge")
      end
    end
  end
end

local function push_right_blue_door(session, label)
  G.select_weapon(session, 0)
  for frame = 0, G.BSC_DOOR_PUSH_FRAMES - 1 do
    local state = session.state
    if state.room_id == ROOM_SC or state.room_id ~= ROOM_BUBBLE then
      return
    end
    if knockback.is_knockback(state) then
      knockback.escape_kb(session, label, "RIGHT", {stop_room_id = ROOM_SC})
    elseif state.samus_y > 430 then
      session:hold(1, {"LEFT", "B", "A"}, label .. "_under_recover")
    elseif state.samus_x < G.BSC_DOOR_X - 20 then
      session:hold(1, {"RIGHT", "B"}, label .. "_reapproach")
    else
      local phase = frame % 16
      if phase < 5 then
        session:hold(1, {"RIGHT", "X"}, label .. "_door_shot")
      elseif phase < 11 then
        session:hold(1, {"RIGHT", "B"}, label .. "_door_push")
      else
        session:hold(1, {"RIGHT", "B", "A"}, label .. "_door_spin")
      end
    end
  end
end

function M.play_bubble_to_single_chamber(session)
  local label = "bubble_to_single_chamber"
  G.require_room(session, ROOM_BUBBLE, label)
  local start = session.frame

  if session.state.samus_y <= G.BSC_TOP_Y_MAX then
    top_walk_to_drop(session, label)
    if session.state.room_id == ROOM_BUBBLE and session.state.samus_y < G.BSC_FLOOR_Y then
      drop_shaft(session, label)
    end
  end
  if session.state.room_id == ROOM_BUBBLE then
    nav_floor_to_door(session, label)
  end
  if session.state.room_id == ROOM_BUBBLE then
    push_right_blue_door(session, label)
  end
  if session.state.room_id ~= ROOM_SC then
    local state = session.state
    error(string.format(
      "%s: Single Chamber door missed; room=0x%04X pose=%d xy=(%d,%d) frames=%d",
      label, state.room_id, state.pose, state.samus_x, state.samus_y,
      session.frame - start
    ))
  end
  return G.wait_ordinary_room(session, ROOM_SC, G.BSC_SINGLE_SETTLE, label)
end

return M
