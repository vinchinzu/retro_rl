-- Pure Ice Acid Room → Ice Snake (left blue door). Horizontal hop, not freeze.
-- Lua 5.1. Session: step / hold / wait_until / span.

local G = require("ice.geometry")
local knockback = require("skills.knockback")

local ROOM_ACID = G.ROOM_ICE_ACID
local ROOM_SNAKE = G.ROOM_ICE_SNAKE
local STANDING = G.STANDING_POSES
local DOOR_PUSH_FRAMES = 280
local LEFT_DOOR_X = 70

local M = {}

local function acid_to_snake_rle(session, label)
  G.play_script(session, G.ACID_TO_SNAKE_RLE, label .. "_rle", ROOM_ACID, function(state)
    return state.room_id == ROOM_SNAKE
  end, "break")
end

local function door_push_if_needed(session, label)
  for frame = 0, DOOR_PUSH_FRAMES - 1 do
    local state = session.state
    if state.room_id == ROOM_SNAKE or state.room_id ~= ROOM_ACID then
      return
    end
    if knockback.is_knockback(state) then
      knockback.escape_knockback_spin(session, {
        prefer_dir = "LEFT",
        run_frames = 3,
        spin_frames = 12,
        label = label .. "_kb",
        run_with = {"B", "X"},
        spin_with = {"B", "A"},
        ensure_beam = true,
        break_on_motion_clear = true,
      })
    else
      local x, y = state.samus_x, state.samus_y
      if y > G.ACID_FLOOR_Y_MAX + 40 then
        session:hold(1, {"LEFT", "B", "A"}, label .. "_recover")
      elseif x <= LEFT_DOOR_X then
        local phase = frame % 16
        if phase < 4 then
          session:hold(1, {"LEFT", "X"}, label .. "_door_shot")
        elseif phase < 10 then
          session:hold(1, {"LEFT", "B"}, label .. "_door_run")
        else
          session:hold(1, {"LEFT", "B", "A"}, label .. "_door_hop")
        end
      else
        local phase = frame % 20
        if phase < 10 then
          session:hold(1, {"LEFT", "B", "A"}, label .. "_push_hop")
        elseif phase < 13 then
          session:hold(1, {"LEFT", "B", "X"}, label .. "_push_shot")
        else
          session:hold(1, {"LEFT", "B"}, label .. "_push_run")
        end
      end
    end
  end
end

function M.play_ice_acid_to_snake(session)
  local label = "ice_acid_to_snake"
  G.require_room(session, ROOM_ACID, label)

  G.unmorph(session)
  if session.state.selected_item ~= 0 then
    G.select_weapon(session, 0)
  end

  if knockback.is_knockback(session.state) then
    knockback.escape_knockback_spin(session, {
      prefer_dir = "LEFT",
      run_frames = 3,
      spin_frames = 12,
      label = label .. "_kb0",
      ensure_beam = true,
      break_on_motion_clear = true,
    })
  end

  acid_to_snake_rle(session, label)
  if session.state.room_id ~= ROOM_SNAKE then
    door_push_if_needed(session, label)
  end
  if session.state.room_id ~= ROOM_SNAKE then
    error(string.format(
      "%s: Ice Snake missed: room=0x%04X xy=(%d,%d) (Acid floor ~y139 x~470)",
      label, session.state.room_id, session.state.samus_x, session.state.samus_y
    ))
  end

  local state = G.wait_ordinary_room(session, ROOM_SNAKE, G.ACID_SNAKE_SETTLE_FRAMES, label)
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
