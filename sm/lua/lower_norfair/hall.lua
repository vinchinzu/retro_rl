-- Lower Norfair entry. Room ids come from finish.rooms.
-- No sm/lua door edge measures Space Jump (0xD9AA) into the main hall
-- (0xB236). Do not invent rooms or hold a path.

local rooms = require("finish.rooms")

local M = {}

M.proven = false
M.ROOM_SPACE_JUMP = rooms.ROOM_SPACE_JUMP
M.ROOM_LN_MAIN_HALL = rooms.ROOM_LN_MAIN_HALL
M.ROOM_GOLDEN_TORIZO = rooms.ROOM_GOLDEN_TORIZO
M.ROOM_RIDLEY = rooms.ROOM_RIDLEY
M.ROOM_RIDLEY_TANK = rooms.ROOM_RIDLEY_TANK
M.ROOM_LN_FARMING = rooms.ROOM_LN_FARMING
M.ROOM_SCREW = rooms.ROOM_SCREW
M.ROOM_METAL_PIRATES = rooms.ROOM_METAL_PIRATES
M.MASK_SCREW = rooms.MASK_SCREW

-- Measured Space Jump -> LN main hall edges. sm/lua has none.
M.DOOR_EDGES = {}

function M.screw_collected(state)
  state = state or {}
  return rooms.bit(state.collected_items, M.MASK_SCREW)
end

-- Plasma is a collected_beams ($09A8) bit, not an item bit. sm/lua does
-- not name that mask. Do not test collected_items for it and do not poke.
function M.beam_word(state)
  state = state or {}
  return tonumber(state.collected_beams) or 0
end

function M.observe(session)
  if type(session) ~= "table" then
    return nil
  end
  if session.state == nil and type(session.refresh) == "function" then
    pcall(function()
      session:refresh()
    end)
  end
  local st = session.state
  if type(st) ~= "table" then
    return nil
  end
  M.screw_collected(st)
  M.beam_word(st)
  return tonumber(st.room_id) or 0
end

function M.entry_measured(room_id)
  room_id = tonumber(room_id)
  if room_id == nil then
    return false
  end
  local i
  for i = 1, #M.DOOR_EDGES do
    local edge = M.DOOR_EDGES[i]
    if edge.from == M.ROOM_SPACE_JUMP and edge.to == M.ROOM_LN_MAIN_HALL then
      return true
    end
  end
  return false
end

function M.play_entry(session)
  local room_id = M.observe(session)
  if not M.entry_measured(room_id) then
    error("UNPROVEN: LN entry")
  end
  return room_id
end

return M
