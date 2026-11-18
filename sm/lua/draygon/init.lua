-- Botwoon 0xD95E to Space Jump 0xD9AA. proven stays false.
-- No energy, ammo, item, boss, door, pose, or xy writes.
-- Verified rooms only: Botwoon, Precious, Draygon, Space Jump.

local fight = require("draygon.fight")
local space_jump = require("draygon.space_jump")

local M = {}

M.proven = false
M.ROOM_BOTWOON = 0xD95E
M.ROOM_PRECIOUS = 0xD78F
M.ROOM_DRAYGON = 0xDA60
M.ROOM_SPACE_JUMP = 0xD9AA
M.SPACE_JUMP_MASK = space_jump.SPACE_JUMP_MASK
M.GRAPPLE_MASK = fight.GRAPPLE_MASK
M.GRAVITY_MASK = space_jump.GRAVITY_MASK
M.POST_SJ_ITEMS = space_jump.POST_SJ_ITEMS
M.fight = fight
M.space_jump = space_jump

local VERIFIED = {
  BOTWOON = M.ROOM_BOTWOON,
  PRECIOUS = M.ROOM_PRECIOUS,
  DRAYGON = M.ROOM_DRAYGON,
  SPACE_JUMP = M.ROOM_SPACE_JUMP,
}

local FINISH_KEYS = {
  BOTWOON = { "BOTWOON", "ROOM_BOTWOON", "botwoon" },
  PRECIOUS = { "PRECIOUS", "ROOM_PRECIOUS", "precious" },
  DRAYGON = { "DRAYGON", "ROOM_DRAYGON", "draygon" },
  SPACE_JUMP = { "SPACE_JUMP", "ROOM_SPACE_JUMP", "space_jump" },
}

-- finish.rooms is optional. Only the four verified ids are accepted.
local function apply_finish_rooms(ids)
  local ok, mod = pcall(require, "finish.rooms")
  if not ok or type(mod) ~= "table" then
    return ids
  end
  local name, keys
  for name, keys in pairs(FINISH_KEYS) do
    local i
    for i = 1, #keys do
      local v = tonumber(mod[keys[i]])
      if v == VERIFIED[name] then
        ids[name] = v
      end
    end
  end
  return ids
end

function M.ids()
  local ids = {
    BOTWOON = VERIFIED.BOTWOON,
    PRECIOUS = VERIFIED.PRECIOUS,
    DRAYGON = VERIFIED.DRAYGON,
    SPACE_JUMP = VERIFIED.SPACE_JUMP,
  }
  return apply_finish_rooms(ids)
end

local function state_of(session)
  if session == nil or session.state == nil then
    error("UNPROVEN: room id stopped in 0x0000")
  end
  return session.state
end

function M.play_botwoon_to_space_jump(session)
  local ids = M.ids()
  local state = state_of(session)
  if space_jump.succeeded(state) then
    return state
  end
  local rid = tonumber(state.room_id) or 0
  if rid == ids.SPACE_JUMP then
    return space_jump.play_collect(session)
  end
  if rid == ids.DRAYGON then
    return fight.play_draygon_room(session)
  end
  if rid == ids.PRECIOUS then
    return fight.play_precious(session)
  end
  -- Botwoon right door 0xA918 does not land in Precious, Draygon, or
  -- Space Jump. The next room id is not verified. Do not cross.
  error(string.format("UNPROVEN: room id stopped in 0x%04X", rid))
end

return M
