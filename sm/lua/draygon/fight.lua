-- Draygon (0xDA60) and Precious (0xD78F). Clean: boss byte is read-only.
-- Spike kill needs grapple. input.lua has no grapple name; do not guess
-- a shoulder or a shot. Super spray is not a measured path.

local ram = require("ram")
local input_mod = require("input")

local M = {}

M.proven = false
M.ROOM_DRAYGON = 0xDA60
M.ROOM_PRECIOUS = 0xD78F
M.ROOM_SPACE_JUMP = 0xD9AA
-- $D828 + Maridia area index 4. Draygon is bit 0. Botwoon is bit 1.
M.ADDR_MARIDIA_BOSS_BITS = 0xD82C
M.DRAYGON_BOSS_BIT = 0x01
M.GRAPPLE_MASK = 0x4000
-- Precious bottom-left door. Destination room is Draygon, not an
-- intermediate. The shaft inputs are not measured.
M.DOOR_PRECIOUS_TO_DRAYGON = 0xA840
-- Draygon left door. Destination room is Space Jump after the boss bit.
M.DOOR_DRAYGON_TO_SPACE_JUMP = 0xA978
-- Draygon right door, back into Precious.
M.DOOR_DRAYGON_TO_PRECIOUS = 0xA96C

local function grapple_bound()
  local order = input_mod.ORDER
  if type(order) ~= "table" then
    return false
  end
  local i
  for i = 1, #order do
    if tostring(order[i]):lower() == "grapple" then
      return true
    end
  end
  return false
end

function M.draygon_defeated(session)
  local ok, bits = pcall(ram.u8, M.ADDR_MARIDIA_BOSS_BITS)
  if not ok or bits == nil then
    return false
  end
  return ram.band(bits, M.DRAYGON_BOSS_BIT) ~= 0
end

local function room_of(session)
  local state = session and session.state
  if state == nil then
    return 0
  end
  return tonumber(state.room_id) or 0
end

function M.play_draygon_fight(session)
  local rid = room_of(session)
  if rid ~= M.ROOM_DRAYGON then
    error(string.format("UNPROVEN: room id stopped in 0x%04X", rid))
  end
  if M.draygon_defeated(session) then
    return session.state
  end
  if not grapple_bound() then
    error("UNPROVEN: grapple button")
  end
  error(string.format("UNPROVEN: draygon fight stopped in 0x%04X", rid))
end

function M.play_precious(session)
  local rid = room_of(session)
  if rid ~= M.ROOM_PRECIOUS then
    error(string.format("UNPROVEN: room id stopped in 0x%04X", rid))
  end
  -- Door 0xA840 is Draygon. The other Precious door's room id is not
  -- verified; do not cross it and do not name it.
  error(string.format("UNPROVEN: precious to draygon stopped in 0x%04X", rid))
end

function M.play_draygon_room(session)
  local rid = room_of(session)
  if rid ~= M.ROOM_DRAYGON then
    error(string.format("UNPROVEN: room id stopped in 0x%04X", rid))
  end
  if not M.draygon_defeated(session) then
    return M.play_draygon_fight(session)
  end
  -- Boss bit is set. Left door 0xA978 targets 0xD9AA. No measured exit.
  error(string.format("UNPROVEN: draygon to space jump stopped in 0x%04X", rid))
end

return M
