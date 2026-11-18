-- These ids are for the scripted finish after Gravity. Not a live proof.
-- Header addresses already verified (human_tape/stitch.py, SOURCE_STATES.md). Do not invent ids.

local rooms = {}

rooms.ROOM_GRAVITY = 0xCE40
rooms.ROOM_GRAPPLE_BEAM = 0xAC2B
rooms.ROOM_GRAPPLE_TUTORIAL = 0xAC00
rooms.ROOM_MAIN_STREET = 0xCFC9
rooms.ROOM_BOTWOON = 0xD95E
rooms.ROOM_DRAYGON = 0xDA60
rooms.ROOM_SPACE_JUMP = 0xD9AA
rooms.ROOM_PLASMA = 0xD2AA
rooms.ROOM_PRECIOUS = 0xD78F
rooms.ROOM_CRAB_HOLE = 0xD21C
rooms.ROOM_GOLDEN_TORIZO = 0xB283
rooms.ROOM_SCREW = 0xB6C1
rooms.ROOM_METAL_PIRATES = 0xB62B
rooms.ROOM_RIDLEY = 0xB32E
rooms.ROOM_RIDLEY_TANK = 0xB698
rooms.ROOM_LN_MAIN_HALL = 0xB236
rooms.ROOM_LN_FARMING = 0xB37A
rooms.ROOM_STATUES = 0xA66A
rooms.ROOM_TOURIAN_FIRST = 0xDAAE
rooms.ROOM_MOTHER_BRAIN = 0xDD58
rooms.ROOM_ESCAPE_1 = 0xDE4D
rooms.ROOM_ESCAPE_4 = 0xDEDE
rooms.ROOM_BUSINESS = 0xA7DE
rooms.ROOM_CATERPILLAR = 0xA322

-- Item masks (collected_items $09A4), also verified against gravity leave
-- items 0x3125 and post-SJ items 0x7325.
rooms.MASK_VARIA = 0x0001
rooms.MASK_MORPH = 0x0004
rooms.MASK_SCREW = 0x0008
rooms.MASK_GRAVITY = 0x0020
rooms.MASK_HIJUMP = 0x0100
rooms.MASK_SPACE_JUMP = 0x0200
rooms.MASK_BOMBS = 0x1000
rooms.MASK_SPEED = 0x2000
rooms.MASK_GRAPPLE = 0x4000

-- lsnes provides global bit. Stock Lua 5.1 does not, including lua5.1 -p hosts.
local function band(a, b)
  if bit and bit.band then
    return bit.band(a, b)
  end
  a = tonumber(a) or 0
  b = tonumber(b) or 0
  if a < 0 then a = a + 65536 end
  if b < 0 then b = b + 65536 end
  a = a % 65536
  b = b % 65536
  local r, p = 0, 1
  for _ = 1, 16 do
    local aa, bb = a % 2, b % 2
    if aa == 1 and bb == 1 then
      r = r + p
    end
    a = (a - aa) / 2
    b = (b - bb) / 2
    p = p * 2
  end
  return r
end

function rooms.bit(items, mask)
  return band(items or 0, mask) ~= 0
end

return rooms
