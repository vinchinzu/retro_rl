-- Morph room → Bomb Torizo defeat → Parlor settle (Clean, no ammo refill).
--
-- Hash-pinned early_game policy RLE + structured BT fight. Never writes
-- missiles / energy / items. Leave is Python bombs tip: Parlor 0x92FD gs=8
-- items include BOMBS_MASK 0x1000 (clean 0x1004 morph+bombs), xy~(968,651).

local ram = require("ram")
local rooms = require("rooms")
local morph_roll = require("bombs.morph_roll")
local bt = require("combat.bomb_torizo")

local M = {}

M.ROOM_MORPH = rooms.ROOM_MORPH or 0x9E9F
M.ROOM_CONSTRUCTION = rooms.ROOM_CONSTRUCTION or 0x9F11
M.ROOM_BLUE_BRINSTAR_ELEVATOR = rooms.ROOM_BLUE_BRINSTAR_ELEVATOR or 0x97B5
M.ROOM_PIT = rooms.ROOM_PIT or 0x975C
M.ROOM_BOMB_TORIZO = rooms.ROOM_BOMB_TORIZO or 0x9804
M.ROOM_FLYWAY = rooms.ROOM_FLYWAY or 0x9879
M.ROOM_PARLOR = rooms.ROOM_PARLOR or 0x92FD
M.MORPH_BALL_MASK = ram.MORPH_BALL_MASK or 0x0004
M.BOMBS_MASK = ram.BOMBS_MASK or 0x1000
M.GS_ORDINARY = ram.GS_ORDINARY or 8
M.PIT_TO_POST_TORIZO_FRAMES = 13143
M.PIT_TO_POST_TORIZO_EXIT_TAIL_START = -2000
M.LEAVE_ROOM = 0x92FD
M.LEAVE_ITEMS_MASK = 0x1000
M.LEAVE_ITEMS_CLEAN = 0x1004
M.LEAVE_X = 968
M.LEAVE_Y = 651

local function num(v, default)
  v = tonumber(v)
  if v == nil then
    return default or 0
  end
  return v
end

local function has_mask(value, mask)
  if ram.band then
    return ram.band(value, mask) ~= 0
  end
  return math.floor(num(value) / mask) % 2 == 1
end

local function timeout(msg)
  local ok, runtime = pcall(require, "runtime")
  if ok and runtime and runtime.timeout then
    runtime.timeout(msg)
  end
  error("TimeoutError: " .. msg)
end

local function brief(state)
  if type(state) ~= "table" then
    return tostring(state)
  end
  return string.format(
    "room=0x%04X gs=%s xy=%s,%s pose=%s items=0x%04X missiles=%s/%s",
    num(state.room_id),
    tostring(state.game_state),
    tostring(state.samus_x or state.x),
    tostring(state.samus_y or state.y),
    tostring(state.pose),
    num(state.collected_items),
    tostring(state.missiles),
    tostring(state.max_missiles)
  )
end

local function hold(session, frames, names, reason)
  names = names or {}
  if frames <= 0 then
    return session.state
  end
  if session.hold then
    session:hold(frames, names, reason)
    return session.state
  end
  local i
  for i = 1, frames do
    session:step(names, reason)
  end
  return session.state
end

local function wait_until(session, pred, timeout_frames, reason)
  timeout_frames = timeout_frames or 120
  if session.wait_until then
    return session:wait_until(pred, timeout_frames, reason)
  end
  local waited
  for waited = 0, timeout_frames do
    if pred(session.state) then
      return waited
    end
    session:step({}, reason)
  end
  timeout(reason .. " timed out: " .. brief(session.state))
end

local function span(session, names, frames, reason)
  if session.span then
    session:span(names or {}, frames, reason)
    return session.state
  end
  return hold(session, frames, names, reason)
end

-- Play {n, names} RLE. opts.skip drops that many tape frames; stop_when
-- ends early after a stepped frame. Returns true if stop_when matched.
local function play_rle(session, runs, reason, opts)
  opts = opts or {}
  local skip = opts.skip or 0
  local stop_when = opts.stop_when
  if skip < 0 then
    local total = 0
    local i
    for i = 1, #runs do
      total = total + runs[i][1]
    end
    skip = total + skip
    if skip < 0 then
      skip = 0
    end
  end
  local played = 0
  local i
  for i = 1, #runs do
    local run = runs[i]
    local n = run[1]
    local names = run[2] or {}
    local f
    for f = 1, n do
      if played >= skip then
        session:step(names, reason)
        if stop_when and stop_when(session.state) then
          return true
        end
      end
      played = played + 1
    end
  end
  return false
end

local TWO_MISSILE_RLE = {
  {122, {}}, {4, {"B"}}, {11, {"B", "RIGHT"}}, {53, {"B", "RIGHT", "A"}}, {3, {"B", "RIGHT"}},
  {8, {"RIGHT"}}, {3, {"RIGHT", "A"}}, {7, {"DOWN", "RIGHT", "A"}}, {1, {"DOWN", "A"}}, {3, {"DOWN"}},
  {9, {"DOWN", "X"}}, {6, {"DOWN"}}, {5, {"DOWN", "X"}}, {7, {"DOWN"}}, {4, {"DOWN", "X"}},
  {6, {"DOWN"}}, {6, {"DOWN", "X"}}, {5, {"DOWN"}}, {6, {"DOWN", "X"}}, {5, {"DOWN"}},
  {7, {"DOWN", "X"}}, {9, {"DOWN"}}, {7, {"DOWN", "X"}}, {19, {"DOWN"}}, {7, {"DOWN", "A"}},
  {8, {"DOWN"}}, {8, {"DOWN", "X"}}, {6, {"DOWN"}}, {8, {"DOWN", "X"}}, {18, {"DOWN"}},
  {2, {"DOWN", "RIGHT"}}, {14, {"RIGHT"}}, {3, {"DOWN", "RIGHT"}}, {6, {"DOWN"}}, {5, {"DOWN", "LEFT"}},
  {13, {}}, {6, {"DOWN"}}, {6, {}}, {6, {"DOWN"}}, {13, {}},
  {31, {"LEFT"}}, {3, {"UP", "LEFT"}}, {5, {"UP"}}, {5, {"UP", "LEFT"}}, {8, {"UP", "LEFT", "X"}},
  {1, {"UP", "LEFT"}}, {17, {"LEFT"}}, {34, {"LEFT", "L"}}, {3, {"LEFT"}}, {134, {}},
  {8, {"LEFT"}}, {13, {"B", "LEFT"}}, {1, {"B", "LEFT", "A"}}, {13, {"LEFT", "A"}}, {23, {"LEFT"}},
  {9, {"LEFT", "X"}}, {14, {"LEFT"}}, {6, {}}, {8, {"X"}}, {3, {}},
  {7, {"A"}}, {2, {"A", "X"}}, {4, {"X"}}, {4, {}}, {6, {"X"}},
  {6, {}}, {4, {"X"}}, {23, {}}, {1, {"LEFT"}}, {8, {"LEFT", "A"}},
  {4, {"LEFT"}}, {23, {}}, {3, {"X"}}, {7, {"RIGHT", "X"}}, {13, {"RIGHT"}},
  {12, {}}, {7, {"A"}}, {3, {"UP", "RIGHT", "A", "L"}}, {3, {"UP", "RIGHT", "L"}}, {1, {"UP", "L"}},
  {5, {"L"}}, {3, {}}, {12, {"A"}}, {47, {}}, {12, {"A"}},
  {22, {}}, {10, {"A", "X"}}, {20, {}}, {12, {"B"}}, {2, {}},
  {14, {"A"}}, {24, {}}, {11, {"A", "X"}}, {2, {"A"}}, {21, {}},
  {10, {"A"}}, {13, {}}, {4, {"RIGHT"}}, {14, {"B", "RIGHT"}}, {1, {"B"}},
  {8, {}}, {10, {"X"}}, {19, {}}, {12, {"A"}}, {13, {}},
  {11, {"RIGHT"}}, {12, {"B", "RIGHT"}}, {7, {"RIGHT"}}, {9, {"RIGHT", "X"}}, {11, {"RIGHT"}},
  {13, {"RIGHT", "L"}}, {40, {"RIGHT"}}, {20, {}}, {8, {"L"}}, {22, {}},
  {13, {"B"}}, {64, {"B", "RIGHT"}}, {4, {"B", "DOWN", "RIGHT"}}, {2, {"DOWN", "RIGHT"}}, {2, {"DOWN"}},
  {10, {}}, {7, {"DOWN"}}, {10, {}}, {6, {"DOWN"}}, {13, {}},
  {28, {"RIGHT"}}, {3, {"UP", "RIGHT"}}, {12, {"UP"}}, {6, {"UP", "A"}}, {7, {"UP", "RIGHT", "A"}},
  {13, {"RIGHT", "A"}}, {16, {"RIGHT"}}, {7, {}}, {4, {"A"}}, {30, {"LEFT", "A"}},
  {9, {"LEFT"}}, {14, {}}, {10, {"UP"}}, {9, {"UP", "X"}}, {6, {"UP"}},
  {5, {"UP", "X"}}, {6, {"UP"}}, {6, {"UP", "X"}}, {6, {"UP"}}, {6, {"UP", "X"}},
  {23, {"UP"}}, {15, {}}, {5, {"A"}}, {36, {"RIGHT", "A"}}, {2, {"UP", "RIGHT", "A"}},
  {2, {"UP", "A"}}, {3, {"UP", "LEFT", "A"}}, {7, {"LEFT", "A"}}, {5, {"LEFT", "A", "X"}}, {6, {"LEFT", "X"}},
  {6, {}}, {7, {"X"}}, {23, {}}, {16, {"A"}}, {12, {"A", "X"}},
  {1, {"A"}}, {6, {}}, {7, {"X"}}, {28, {}}, {12, {"A"}},
  {13, {"LEFT", "A"}}, {18, {"LEFT"}}, {4, {}}, {3, {"RIGHT"}}, {32, {"B", "RIGHT", "A"}},
  {19, {"RIGHT"}}, {14, {"SELECT", "RIGHT"}}, {17, {"RIGHT"}}, {7, {"RIGHT", "X"}}, {5, {"RIGHT"}},
  {6, {"RIGHT", "X"}}, {4, {"RIGHT"}}, {6, {"RIGHT", "X"}}, {5, {"RIGHT"}}, {6, {"RIGHT", "X"}},
  {5, {"RIGHT"}}, {6, {"RIGHT", "X"}}, {22, {"RIGHT"}}, {12, {"RIGHT", "L"}}, {16, {"RIGHT"}},
  {8, {"RIGHT", "A"}}, {1, {"RIGHT", "A", "L"}}, {6, {"RIGHT", "L"}}, {12, {"RIGHT"}}, {39, {}},
  {11, {"L"}}, {68, {}}, {3, {"RIGHT"}}, {157, {"B", "RIGHT"}}, {3, {"B", "DOWN", "RIGHT"}},
  {7, {"B", "DOWN"}}, {11, {"B"}}, {7, {"B", "DOWN"}}, {14, {"B"}}, {48, {"B", "RIGHT"}},
  {8, {"B", "UP", "RIGHT"}}, {4, {"B", "UP", "RIGHT", "A"}}, {5, {"UP", "RIGHT", "A"}}, {5, {"RIGHT", "A"}}, {35, {"RIGHT"}},
  {19, {}}, {1, {"UP"}}, {2, {"UP", "LEFT"}}, {8, {"UP", "LEFT", "L"}}, {2, {"UP", "LEFT"}},
  {1, {"UP", "LEFT", "A"}}, {16, {"A"}}, {14, {}}, {6, {"X"}}, {1, {"A", "X"}},
  {2, {"A", "X", "L"}}, {5, {"A", "L"}}, {6, {"L"}}, {9, {}}, {12, {"A"}},
  {19, {}}, {9, {"X"}}, {14, {}}, {4, {"LEFT"}}, {4, {"UP", "LEFT"}},
  {17, {"LEFT"}}, {9, {"UP", "LEFT", "L"}}, {4, {"UP", "LEFT"}}, {2, {"LEFT"}}, {2, {"X"}},
  {9, {"A", "X"}}, {2, {"A"}}, {18, {}}, {13, {"LEFT"}}, {10, {"LEFT", "A"}},
  {13, {"LEFT"}}, {10, {"LEFT", "L"}}, {5, {"LEFT"}}, {2, {"LEFT", "A"}}, {8, {"LEFT", "A", "X"}},
  {1, {"LEFT", "A"}}, {23, {"LEFT"}}, {14, {}}, {11, {"A"}}, {6, {}},
  {9, {"X"}}, {23, {}}, {9, {"A"}}, {22, {}}, {12, {"DOWN"}},
  {8, {}}, {8, {"DOWN"}}, {14, {}}, {94, {"LEFT"}}, {16, {"UP", "LEFT"}},
  {3, {"LEFT"}}, {47, {"B", "LEFT"}}, {36, {"B", "UP", "LEFT"}}, {17, {"B", "UP", "LEFT", "A"}}, {3, {"B", "LEFT", "A"}},
  {2, {"B", "LEFT"}}, {10, {"LEFT"}}, {8, {"LEFT", "X"}}, {15, {"LEFT"}}, {19, {"LEFT", "L"}},
  {23, {"LEFT"}}, {136, {}}, {2, {"LEFT"}}, {9, {"B", "LEFT"}}, {20, {"B", "LEFT", "A"}},
  {42, {"B", "LEFT"}}, {36, {"LEFT"}}, {105, {}},
}

local CONSTRUCTION_RETURN_RLE = {
  {77, {"LEFT"}}, {1, {"UP", "LEFT"}}, {14, {"UP"}}, {8, {"UP", "LEFT"}}, {20, {"UP", "LEFT", "A"}},
  {30, {"UP", "LEFT"}}, {9, {"UP", "LEFT", "X"}}, {4, {"UP", "LEFT"}}, {45, {"LEFT"}}, {25, {}},
  {93, {"LEFT"}}, {23, {"B", "LEFT"}}, {21, {"B", "LEFT", "A"}}, {1, {"LEFT", "A"}}, {52, {"LEFT"}},
  {10, {"LEFT", "X"}}, {85, {"LEFT"}}, {3, {}}, {6, {"A"}}, {116, {}},
  {1, {"LEFT"}}, {144, {"B", "LEFT"}}, {16, {"LEFT"}}, {12, {"LEFT", "A"}}, {34, {"LEFT"}},
  {9, {}}, {3, {"UP"}}, {3, {"UP", "RIGHT"}}, {5, {"RIGHT"}}, {20, {}},
  {7, {"RIGHT"}}, {17, {}}, {18, {"UP"}}, {424, {}},
}

local ELEVATOR_RETURN_RLE = {
  {261, {}}, {16, {"LEFT"}}, {5, {"LEFT", "X"}}, {8, {"LEFT"}}, {7, {"LEFT", "X"}},
  {55, {"LEFT"}},
}

local PIT_TO_POST_TORIZO_RLE = {
  {146, {}}, {3, {"UP"}}, {11, {"UP", "RIGHT"}}, {1, {"UP"}}, {18, {}},
  {36, {"UP"}}, {28, {}}, {8, {"UP"}}, {64, {}}, {5, {"RIGHT"}},
  {90, {}}, {5, {"UP"}}, {2, {"UP", "X"}}, {7, {"UP", "RIGHT", "X"}}, {12, {"UP", "RIGHT"}},
  {1, {"RIGHT"}}, {3, {"RIGHT", "X"}}, {5, {"X"}}, {10, {}}, {1, {"UP"}},
  {8, {"UP", "LEFT"}}, {3, {"UP"}}, {2, {}}, {13, {"R"}}, {10, {"X", "R"}},
  {5, {"R"}}, {7, {"X", "R"}}, {5, {"R"}}, {5, {"X", "R"}}, {1, {"RIGHT", "X", "R"}},
  {6, {"RIGHT", "R"}}, {9, {"RIGHT", "A", "R"}}, {13, {"RIGHT", "R"}}, {3, {"UP", "RIGHT", "R"}}, {2, {"UP", "R"}},
  {7, {"UP", "LEFT", "R"}}, {13, {"R"}}, {14, {"A", "R"}}, {7, {"R"}}, {7, {"X", "R"}},
  {1, {"X"}}, {7, {}}, {6, {"X"}}, {21, {}}, {14, {"A"}},
  {4, {"A", "X"}}, {6, {"X"}}, {4, {}}, {3, {"RIGHT"}}, {8, {"RIGHT", "X"}},
  {15, {"RIGHT"}}, {1, {}}, {13, {"LEFT"}}, {6, {"B", "LEFT"}}, {1, {"LEFT"}},
  {15, {"LEFT", "A"}}, {26, {"LEFT"}}, {7, {}}, {10, {"LEFT", "A"}}, {24, {"LEFT"}},
  {8, {"LEFT", "X"}}, {1, {"X"}}, {6, {}}, {6, {"X"}}, {2, {}},
  {29, {"LEFT"}}, {9, {}}, {14, {"LEFT"}}, {25, {"LEFT", "A"}}, {9, {"LEFT"}},
  {27, {}}, {16, {"LEFT"}}, {11, {"LEFT", "A"}}, {11, {"A"}}, {3, {}},
  {10, {"RIGHT"}}, {2, {}}, {10, {"A"}}, {8, {"LEFT", "A"}}, {7, {"LEFT"}},
  {7, {"LEFT", "X"}}, {3, {"LEFT"}}, {15, {}}, {41, {"DOWN"}}, {3, {"DOWN", "LEFT"}},
  {1, {"LEFT"}}, {2, {}}, {3, {"A"}}, {38, {"LEFT", "A"}}, {32, {"LEFT"}},
  {21, {"LEFT", "A"}}, {31, {"LEFT"}}, {20, {"LEFT", "A"}}, {49, {"LEFT"}}, {14, {"LEFT", "A"}},
  {43, {"LEFT"}}, {12, {"UP", "LEFT"}}, {1, {"LEFT"}}, {3, {"LEFT", "X"}}, {9, {"LEFT", "A", "X"}},
  {1, {"LEFT", "X"}}, {9, {"LEFT"}}, {6, {"LEFT", "X"}}, {4, {"LEFT"}}, {12, {}},
  {3, {"A"}}, {13, {"LEFT", "A"}}, {42, {"LEFT"}}, {2, {"B", "LEFT"}}, {63, {"B", "LEFT", "A"}},
  {2, {"B", "LEFT"}}, {5, {"LEFT"}}, {3, {"LEFT", "A"}}, {24, {"B", "LEFT", "A"}}, {2, {"LEFT", "A"}},
  {26, {"LEFT"}}, {8, {"LEFT", "X"}}, {6, {"LEFT"}}, {7, {"LEFT", "X"}}, {5, {"LEFT"}},
  {9, {"LEFT", "X"}}, {31, {"LEFT"}}, {4, {"LEFT", "L"}}, {3, {"UP", "LEFT", "L"}}, {6, {"LEFT", "L"}},
  {4, {"LEFT"}}, {108, {}}, {10, {"UP"}}, {9, {"UP", "X"}}, {6, {"UP"}},
  {16, {"UP", "X"}}, {6, {"UP"}}, {9, {"UP", "LEFT"}}, {1, {"UP"}}, {4, {}},
  {8, {"A"}}, {13, {"LEFT", "A"}}, {5, {"LEFT"}}, {6, {}}, {14, {"LEFT"}},
  {16, {}}, {42, {"LEFT"}}, {12, {}}, {24, {"A"}}, {1, {"UP", "LEFT", "A"}},
  {2, {"LEFT", "A"}}, {7, {"LEFT"}}, {31, {}}, {9, {"R"}}, {21, {"A", "R"}},
  {4, {"A", "X", "R"}}, {8, {"X", "R"}}, {6, {"R"}}, {7, {"X", "R"}}, {5, {"R"}},
  {7, {"X", "R"}}, {11, {"R"}}, {19, {}}, {4, {"A"}}, {18, {"LEFT", "A"}},
  {9, {"LEFT"}}, {19, {}}, {2, {"B"}}, {11, {"B", "A"}}, {17, {"B", "RIGHT", "A"}},
  {9, {"B", "RIGHT"}}, {19, {"RIGHT"}}, {1, {"RIGHT", "A"}}, {19, {"B", "A"}}, {1, {"B", "UP", "LEFT", "A"}},
  {17, {"B", "LEFT", "A"}}, {1, {"B", "LEFT"}}, {24, {}}, {13, {"A"}}, {22, {"LEFT", "A"}},
  {1, {"A"}}, {1, {"RIGHT", "A"}}, {7, {"RIGHT"}}, {8, {"X"}}, {6, {}},
  {6, {"X"}}, {4, {}}, {7, {"X"}}, {1, {}}, {4, {"DOWN"}},
  {6, {"DOWN", "X"}}, {6, {"DOWN"}}, {45, {}}, {14, {"A"}}, {4, {"UP", "A"}},
  {3, {"UP"}}, {3, {"UP", "RIGHT"}}, {2, {"RIGHT"}}, {14, {}}, {1, {"LEFT"}},
  {2, {"UP", "LEFT"}}, {2, {"LEFT"}}, {23, {}}, {11, {"A"}}, {7, {}},
  {8, {"X"}}, {16, {"X", "R"}}, {6, {"R"}}, {8, {"X", "R"}}, {16, {"R"}},
  {27, {"A"}}, {7, {"LEFT", "A"}}, {8, {"A"}}, {46, {}}, {3, {"B", "A"}},
  {27, {"B", "RIGHT", "A"}}, {3, {"B", "RIGHT"}}, {11, {"RIGHT"}}, {9, {}}, {25, {"A"}},
  {16, {"LEFT", "A"}}, {7, {"LEFT"}}, {7, {}}, {16, {"A"}}, {8, {"LEFT", "A"}},
  {12, {"LEFT"}}, {12, {}}, {2, {"B"}}, {1, {"B", "A"}}, {45, {"B", "RIGHT", "A"}},
  {1, {"B", "RIGHT"}}, {4, {"RIGHT"}}, {4, {}}, {8, {"X"}}, {5, {}},
  {7, {"X"}}, {4, {}}, {20, {"DOWN"}}, {14, {}}, {6, {"RIGHT"}},
  {41, {}}, {3, {"A"}}, {12, {"B", "A"}}, {2, {"B", "LEFT", "A"}}, {2, {"B", "UP", "LEFT", "A"}},
  {21, {"B", "LEFT", "A"}}, {1, {"LEFT", "A"}}, {15, {}}, {17, {"A"}}, {11, {"LEFT", "A"}},
  {1, {"LEFT"}}, {56, {}}, {18, {"A"}}, {1, {"A", "R"}}, {10, {"A", "X", "R"}},
  {7, {"X", "R"}}, {3, {"R"}}, {39, {}}, {2, {"B", "LEFT"}}, {32, {"B", "LEFT", "A"}},
  {6, {"LEFT"}}, {1, {"DOWN", "LEFT", "A"}}, {1, {"DOWN", "A"}}, {1, {"DOWN", "RIGHT", "A"}}, {27, {"RIGHT", "A"}},
  {2, {}}, {10, {"LEFT"}}, {20, {}}, {2, {"B"}}, {1, {"B", "LEFT"}},
  {24, {"B", "LEFT", "A"}}, {1, {"B", "LEFT"}}, {4, {"LEFT"}}, {1, {}}, {1, {"DOWN", "A"}},
  {1, {"DOWN", "RIGHT", "A"}}, {22, {"RIGHT", "A"}}, {1, {"A"}}, {19, {}}, {7, {"A"}},
  {44, {"RIGHT", "A"}}, {1, {"RIGHT"}}, {12, {}}, {27, {"A"}}, {11, {"LEFT", "A"}},
  {9, {"LEFT"}}, {13, {}}, {8, {"A"}}, {18, {"LEFT", "A"}}, {3, {"LEFT"}},
  {21, {}}, {6, {"RIGHT"}}, {6, {}}, {24, {"A"}}, {4, {"A", "X"}},
  {6, {"X"}}, {15, {}}, {8, {"X"}}, {78, {}}, {19, {"B", "A"}},
  {30, {"B", "RIGHT", "A"}}, {1, {"B", "RIGHT"}}, {5, {"RIGHT"}}, {6, {}}, {26, {"A"}},
  {11, {"LEFT", "A"}}, {12, {"LEFT"}}, {23, {"A"}}, {3, {"A", "X"}}, {5, {"X"}},
  {6, {}}, {7, {"X"}}, {24, {}}, {4, {"B"}}, {4, {"B", "A"}},
  {18, {"B", "LEFT", "A"}}, {4, {"B", "LEFT"}}, {1, {"LEFT"}}, {19, {}}, {1, {"A"}},
  {11, {"B", "A"}}, {35, {"B", "RIGHT", "A"}}, {1, {"B", "RIGHT"}}, {2, {"RIGHT"}}, {9, {}},
  {20, {"B", "A"}}, {4, {"A"}}, {2, {"A", "X"}}, {6, {"X"}}, {6, {}},
  {7, {"X"}}, {4, {}}, {9, {"X"}}, {4, {}}, {9, {"X"}},
  {12, {}}, {16, {"A"}}, {21, {"A", "X"}}, {1, {"LEFT", "A", "X"}}, {1, {"UP", "LEFT", "A", "X"}},
  {5, {"LEFT", "A", "X"}}, {2, {"UP", "LEFT", "X"}}, {3, {"UP", "LEFT"}}, {4, {"LEFT"}}, {19, {}},
  {4, {"A"}}, {17, {"LEFT", "A"}}, {1, {"LEFT"}}, {9, {}}, {12, {"X"}},
  {22, {}}, {11, {"A"}}, {2, {}}, {13, {"X"}}, {56, {}},
  {1, {"B"}}, {1, {"B", "A"}}, {25, {"B", "LEFT", "A"}}, {2, {"LEFT", "A"}}, {5, {"LEFT"}},
  {2, {}}, {26, {"RIGHT"}}, {1, {"DOWN", "RIGHT"}}, {1, {"RIGHT"}}, {27, {}},
  {17, {"B", "A"}}, {30, {"B", "RIGHT", "A"}}, {6, {"B", "RIGHT"}}, {7, {}}, {35, {"A"}},
  {10, {"LEFT", "A"}}, {9, {"LEFT"}}, {19, {}}, {21, {"A"}}, {13, {"LEFT", "A"}},
  {6, {"LEFT"}}, {11, {}}, {4, {"X"}}, {20, {"X", "R"}}, {5, {"R"}},
  {9, {"X", "R"}}, {11, {"R"}}, {10, {"X", "R"}}, {6, {"R"}}, {6, {"X", "R"}},
  {5, {"R"}}, {8, {"X", "R"}}, {5, {"R"}}, {6, {"X", "R"}}, {9, {"R"}},
  {6, {"X", "R"}}, {7, {"R"}}, {8, {"X", "R"}}, {14, {"R"}}, {16, {}},
  {31, {"A"}}, {16, {"RIGHT", "A"}}, {3, {"RIGHT"}}, {1, {"DOWN", "RIGHT"}}, {10, {}},
  {25, {"A"}}, {6, {"LEFT", "A"}}, {4, {"LEFT"}}, {25, {}}, {1, {"A"}},
  {26, {"RIGHT", "A"}}, {21, {"RIGHT"}}, {4, {"UP", "RIGHT"}}, {11, {"UP"}}, {10, {"UP", "A"}},
  {3, {"UP", "A", "X"}}, {22, {"UP", "X"}}, {19, {"UP"}}, {42, {"UP", "A"}}, {15, {"UP"}},
  {169, {}}, {40, {"LEFT"}}, {7, {}}, {5, {"RIGHT"}}, {42, {}},
  {20, {"B", "A"}}, {14, {"B", "RIGHT", "A"}}, {1, {"RIGHT", "A"}}, {2, {}}, {46, {"R"}},
  {6, {}}, {13, {"B", "A"}}, {10, {"B", "A", "R"}}, {3, {"B", "A"}}, {5, {"B", "RIGHT", "A"}},
  {15, {"B", "RIGHT", "A", "R"}}, {16, {"B", "A", "R"}}, {1, {"B", "A"}}, {3, {"A"}}, {15, {}},
  {1, {"B", "A"}}, {26, {"B", "RIGHT", "A"}}, {4, {"B", "RIGHT", "A", "R"}}, {16, {"RIGHT", "A", "R"}}, {4, {"RIGHT", "R"}},
  {8, {"RIGHT"}}, {3, {"RIGHT", "R"}}, {18, {"R"}}, {1, {}}, {1, {"B"}},
  {1, {"B", "A"}}, {16, {"B", "LEFT", "A"}}, {1, {"LEFT", "A"}}, {3, {"LEFT"}}, {9, {"LEFT", "R"}},
  {28, {"R"}}, {1, {"LEFT", "R"}}, {1, {"LEFT", "A", "R"}}, {19, {"B", "LEFT", "A", "R"}}, {1, {"B", "LEFT", "A"}},
  {2, {"LEFT", "A"}}, {11, {"LEFT"}}, {3, {"LEFT", "R"}}, {25, {"R"}}, {1, {"RIGHT", "R"}},
  {15, {"RIGHT", "A", "R"}}, {6, {"RIGHT", "A"}}, {15, {"RIGHT"}}, {3, {"RIGHT", "A"}}, {9, {"B", "RIGHT", "A"}},
  {2, {"RIGHT", "A"}}, {3, {"A"}}, {1, {"A", "R"}}, {6, {"R"}}, {3, {}},
  {7, {"LEFT"}}, {8, {}}, {5, {"A"}}, {13, {"LEFT", "A"}}, {10, {"LEFT"}},
  {10, {"LEFT", "R"}}, {13, {"R"}}, {1, {"A", "R"}}, {34, {"RIGHT", "A", "R"}}, {2, {"RIGHT", "A"}},
  {12, {"RIGHT"}}, {11, {"RIGHT", "R"}}, {13, {"R"}}, {6, {}}, {9, {"LEFT"}},
  {2, {"LEFT", "A"}}, {13, {"A"}}, {20, {"RIGHT", "A"}}, {11, {"RIGHT", "A", "R"}}, {2, {"RIGHT", "R"}},
  {29, {}}, {1, {"LEFT"}}, {1, {"UP", "LEFT"}}, {22, {"B", "LEFT", "A"}}, {4, {"LEFT", "A"}},
  {6, {"LEFT"}}, {20, {"LEFT", "R"}}, {12, {"R"}}, {15, {"RIGHT", "R"}}, {11, {"RIGHT", "A", "R"}},
  {1, {"RIGHT", "A"}}, {24, {"B", "RIGHT", "A"}}, {3, {"RIGHT", "A"}}, {2, {"RIGHT", "A", "R"}}, {38, {"RIGHT", "R"}},
  {20, {}}, {12, {"A"}}, {11, {}}, {2, {"A"}}, {1, {"UP", "LEFT", "A"}},
  {9, {"LEFT", "A"}}, {2, {"UP", "LEFT", "A"}}, {22, {"A"}}, {11, {}}, {20, {"A"}},
  {24, {"RIGHT", "A"}}, {1, {"A"}}, {14, {}}, {11, {"RIGHT"}}, {16, {}},
  {8, {"LEFT"}}, {27, {}}, {1, {"LEFT"}}, {2, {"UP", "LEFT"}}, {3, {"UP", "LEFT", "A"}},
  {37, {"LEFT", "A"}}, {7, {"LEFT"}}, {14, {}}, {28, {"A"}}, {1, {"LEFT", "A"}},
  {2, {"UP", "LEFT", "A"}}, {2, {"LEFT", "A"}}, {24, {"LEFT"}}, {16, {}}, {9, {"LEFT"}},
  {10, {}}, {16, {"RIGHT"}}, {10, {}}, {7, {"B", "A"}}, {1, {"B", "LEFT", "A"}},
  {1, {"B", "UP", "LEFT", "A"}}, {24, {"B", "LEFT", "A"}}, {2, {"LEFT"}}, {4, {}}, {17, {"RIGHT"}},
  {16, {}}, {1, {"RIGHT"}}, {25, {"RIGHT", "A"}}, {7, {"RIGHT"}}, {19, {}},
  {2, {"UP", "LEFT"}}, {6, {"LEFT"}}, {25, {}}, {18, {"A"}}, {12, {"LEFT", "A"}},
  {5, {"LEFT"}}, {22, {}}, {1, {"B"}}, {1, {"B", "LEFT"}}, {1, {"B", "UP", "LEFT"}},
  {17, {"B", "LEFT", "A"}}, {2, {"LEFT", "A"}}, {5, {"LEFT"}}, {21, {"LEFT", "A"}}, {2, {"A"}},
  {3, {"RIGHT", "A"}}, {10, {"RIGHT"}}, {26, {}}, {8, {"RIGHT"}}, {32, {}},
  {23, {"LEFT", "A"}}, {3, {}}, {1, {"RIGHT"}}, {11, {"RIGHT", "A"}}, {5, {"A"}},
  {3, {}}, {1, {"RIGHT"}}, {21, {}}, {6, {"A"}}, {23, {"LEFT", "A"}},
  {5, {"LEFT"}}, {5, {"UP", "LEFT"}}, {14, {"LEFT"}}, {27, {}}, {17, {"A"}},
  {1, {"LEFT", "A"}}, {4, {"UP", "LEFT", "A"}}, {19, {"LEFT", "A"}}, {3, {"A"}}, {42, {}},
  {4, {"A"}}, {26, {"RIGHT", "A"}}, {1, {"RIGHT", "A", "R"}}, {28, {"RIGHT", "R"}}, {3, {"R"}},
  {1, {"A", "R"}}, {7, {"B", "A", "R"}}, {21, {"B", "A"}}, {1, {"B", "LEFT", "A"}}, {4, {"B", "UP", "LEFT", "A"}},
  {2, {"UP", "LEFT", "A"}}, {4, {"LEFT"}}, {4, {}}, {8, {"A"}}, {14, {"RIGHT", "A"}},
  {3, {"A"}}, {14, {}}, {18, {"A"}}, {31, {"RIGHT", "A"}}, {6, {"RIGHT"}},
  {16, {}}, {16, {"RIGHT"}}, {4, {}}, {7, {"A"}}, {6, {"RIGHT", "A"}},
  {7, {"RIGHT"}}, {1, {"DOWN", "RIGHT"}}, {18, {}}, {6, {"DOWN"}}, {5, {}},
  {4, {"DOWN"}}, {18, {}}, {136, {"RIGHT"}}, {24, {}}, {9, {"RIGHT"}},
  {100, {}}, {13, {"RIGHT"}}, {12, {"RIGHT", "X"}}, {17, {"RIGHT"}}, {11, {"UP", "RIGHT"}},
  {3, {"UP", "RIGHT", "X"}}, {8, {"RIGHT", "X"}}, {27, {"RIGHT"}}, {10, {"RIGHT", "X"}}, {96, {"RIGHT"}},
  {71, {}}, {5, {"RIGHT"}}, {87, {"B", "RIGHT"}}, {15, {"B", "RIGHT", "R"}}, {45, {"B", "RIGHT", "X", "R"}},
  {1, {"B", "RIGHT", "R"}}, {16, {"B", "RIGHT"}}, {21, {"B", "RIGHT", "A"}}, {1, {"B", "RIGHT"}}, {14, {"RIGHT"}},
  {7, {}}, {16, {"SELECT"}}, {8, {}}, {5, {"X"}}, {7, {}},
  {5, {"X"}}, {5, {}}, {6, {"X"}}, {5, {}}, {4, {"X"}},
  {5, {}}, {5, {"X"}}, {20, {}}, {38, {"RIGHT"}}, {100, {}},
  {11, {"SELECT"}}, {78, {}}, {45, {"RIGHT"}}, {11, {"RIGHT", "A"}}, {38, {"RIGHT"}},
  {7, {"RIGHT", "A"}}, {9, {"RIGHT"}}, {7, {"RIGHT", "X"}}, {6, {"RIGHT"}}, {8, {"RIGHT", "X"}},
  {10, {"RIGHT"}}, {37, {}}, {11, {"SELECT"}}, {10, {}}, {7, {"A"}},
  {2, {}}, {9, {"RIGHT"}}, {2, {"RIGHT", "X"}}, {5, {"X"}}, {6, {}},
  {5, {"X"}}, {7, {}}, {7, {"X"}}, {11, {}}, {1, {"A"}},
  {8, {"RIGHT", "A"}}, {18, {"RIGHT"}}, {12, {}}, {11, {"A"}}, {7, {}},
  {12, {"SELECT"}}, {5, {}}, {10, {"A"}}, {13, {}}, {5, {"LEFT", "A"}},
  {10, {"LEFT"}}, {3, {}}, {9, {"B"}}, {4, {}}, {12, {"A"}},
  {38, {}}, {2, {"A"}}, {5, {"LEFT", "A"}}, {8, {"LEFT"}}, {4, {"LEFT", "A"}},
  {1, {"LEFT"}}, {9, {}}, {7, {"A"}}, {80, {}}, {6, {"X"}},
  {6, {"B", "A", "X"}}, {1, {"B", "A"}}, {9, {"A"}}, {12, {}}, {9, {"A"}},
  {16, {}}, {11, {"X"}}, {15, {}}, {10, {"A"}}, {9, {}},
  {15, {"RIGHT"}}, {7, {}}, {23, {"LEFT"}}, {17, {"B", "LEFT"}}, {7, {"B", "LEFT", "A"}},
  {35, {"LEFT", "A"}}, {7, {"LEFT"}}, {7, {}}, {54, {"RIGHT"}}, {18, {}},
  {11, {"RIGHT"}}, {42, {}}, {22, {"LEFT"}}, {2, {}}, {13, {"RIGHT"}},
  {41, {}}, {10, {"DOWN"}}, {31, {"DOWN", "X"}}, {13, {"DOWN"}}, {13, {"DOWN", "R"}},
  {15, {"DOWN", "X", "R"}}, {43, {"DOWN", "R"}}, {9, {"DOWN", "RIGHT", "R"}}, {2, {"DOWN", "R"}}, {25, {"R"}},
  {70, {"DOWN", "R"}}, {24, {"DOWN", "X", "R"}}, {6, {"DOWN", "R"}}, {6, {"DOWN", "X", "R"}}, {6, {"DOWN", "R"}},
  {6, {"DOWN", "X", "R"}}, {5, {"DOWN", "R"}}, {6, {"DOWN", "X", "R"}}, {6, {"DOWN", "R"}}, {5, {"DOWN", "X", "R"}},
  {2, {"DOWN", "LEFT", "R"}}, {33, {"LEFT", "R"}}, {2, {"R"}}, {13, {"RIGHT", "R"}}, {1, {"DOWN", "RIGHT", "R"}},
  {13, {"R"}}, {15, {"DOWN", "R"}}, {7, {"DOWN", "X", "R"}}, {7, {"DOWN", "R"}}, {3, {"DOWN", "X", "R"}},
  {4, {"DOWN", "R"}}, {2, {"R"}}, {9, {"UP", "R"}}, {6, {"UP", "X", "R"}}, {2, {"UP", "RIGHT", "R"}},
  {3, {"RIGHT", "R"}}, {2, {"RIGHT", "X", "R"}}, {4, {"DOWN", "RIGHT", "X", "R"}}, {4, {"DOWN", "RIGHT", "R"}}, {6, {"DOWN", "RIGHT", "X", "R"}},
  {4, {"DOWN", "RIGHT", "R"}}, {6, {"DOWN", "RIGHT", "X", "R"}}, {5, {"RIGHT", "R"}}, {5, {"RIGHT", "X", "R"}}, {1, {"RIGHT", "R"}},
  {5, {"R"}}, {5, {"X", "R"}}, {5, {"R"}}, {4, {"X", "R"}}, {3, {"R"}},
  {2, {"UP", "R"}}, {7, {"UP", "X", "R"}}, {1, {"UP", "R"}}, {1, {"UP"}}, {3, {"UP", "RIGHT"}},
  {1, {"RIGHT"}}, {13, {}}, {10, {"SELECT"}}, {16, {}}, {10, {"R"}},
  {5, {"X", "R"}}, {4, {"UP", "X", "R"}}, {2, {"X", "R"}}, {3, {"R"}}, {15, {"X", "R"}},
  {4, {"R"}}, {7, {"X", "R"}}, {5, {"UP", "RIGHT", "R"}}, {5, {"RIGHT", "R"}}, {1, {"RIGHT", "X", "R"}},
  {5, {"X", "R"}}, {5, {"R"}}, {7, {"X", "R"}}, {3, {"R"}}, {3, {"X", "R"}},
  {3, {"RIGHT", "X", "R"}}, {4, {"RIGHT", "R"}}, {1, {"R"}}, {4, {"X", "R"}}, {3, {"LEFT", "X", "R"}},
  {2, {"LEFT", "R"}}, {2, {"UP", "LEFT", "R"}}, {1, {"UP", "LEFT", "X", "R"}}, {3, {"UP", "X", "R"}}, {4, {"UP", "R"}},
  {6, {"UP", "X", "R"}}, {7, {"UP", "X"}}, {4, {"UP"}}, {5, {"UP", "X"}}, {5, {"UP", "LEFT"}},
  {2, {"UP", "LEFT", "X"}}, {4, {"LEFT", "X"}}, {4, {"LEFT"}}, {3, {"LEFT", "X"}}, {2, {"UP", "LEFT", "X"}},
  {4, {}}, {6, {"X"}}, {3, {}}, {8, {"X"}}, {4, {}},
  {5, {"X"}}, {1, {"UP", "RIGHT", "X"}}, {6, {"UP", "RIGHT"}}, {3, {"UP", "RIGHT", "X"}}, {3, {"UP", "X"}},
  {5, {"UP"}}, {2, {"UP", "A"}}, {14, {"A"}}, {7, {}}, {5, {"X"}},
  {5, {}}, {5, {"X"}}, {3, {}}, {2, {"X"}}, {4, {"LEFT", "X"}},
  {2, {"LEFT"}}, {9, {"LEFT", "X"}}, {28, {"LEFT"}}, {1, {}}, {8, {"RIGHT"}},
  {2, {"RIGHT", "A"}}, {3, {"A"}}, {5, {"A", "X"}}, {5, {}}, {6, {"X"}},
  {4, {}}, {5, {"X"}}, {5, {}}, {5, {"X"}}, {8, {}},
  {1, {"B"}}, {2, {"B", "A"}}, {10, {"B", "A", "X"}}, {38, {"B", "RIGHT", "A", "X"}}, {1, {"RIGHT", "X"}},
  {13, {"RIGHT"}}, {11, {"RIGHT", "X"}}, {4, {"RIGHT"}}, {8, {"RIGHT", "X"}}, {13, {"RIGHT"}},
  {1, {}}, {14, {"LEFT"}}, {1, {"LEFT", "R"}}, {8, {"R"}}, {9, {"X", "R"}},
  {4, {"R"}}, {5, {"X", "R"}}, {1, {"UP", "LEFT", "X", "R"}}, {1, {"UP", "LEFT", "R"}}, {4, {"LEFT", "R"}},
  {5, {"LEFT", "X", "R"}}, {4, {"R"}}, {5, {"X", "R"}}, {4, {"R"}}, {3, {"X", "R"}},
  {3, {"RIGHT", "X", "R"}}, {3, {"RIGHT", "R"}}, {2, {"RIGHT", "X", "R"}}, {4, {"X", "R"}}, {1, {"LEFT", "X", "R"}},
  {5, {"LEFT", "R"}}, {3, {"LEFT", "X", "R"}}, {2, {"LEFT", "R"}}, {3, {"R"}}, {5, {"X", "R"}},
  {1, {"R"}}, {2, {"RIGHT", "R"}}, {7, {"RIGHT", "X", "R"}}, {4, {"RIGHT", "R"}}, {11, {"RIGHT", "X", "R"}},
  {9, {"RIGHT", "R"}}, {1, {"UP", "RIGHT", "R"}}, {1, {"UP", "R"}}, {1, {"UP", "LEFT", "R"}}, {7, {"LEFT", "R"}},
  {5, {"LEFT", "X", "R"}}, {4, {"LEFT", "R"}}, {3, {"R"}}, {5, {"X", "R"}}, {3, {"R"}},
  {4, {"X", "R"}}, {5, {"R"}}, {5, {"X", "R"}}, {6, {"R"}}, {2, {"X", "R"}},
  {5, {"R"}}, {7, {"X", "R"}}, {4, {"R"}}, {6, {"X", "R"}}, {5, {"R"}},
  {5, {"X", "R"}}, {5, {"R"}}, {5, {"X", "R"}}, {5, {"R"}}, {5, {"X", "R"}},
  {2, {"R"}}, {9, {"X", "R"}}, {1, {"R"}}, {7, {"X", "R"}}, {5, {"R"}},
  {7, {"X", "R"}}, {2, {"R"}}, {1, {"UP", "LEFT", "R"}}, {7, {"LEFT", "X", "R"}}, {4, {"LEFT", "R"}},
  {1, {"R"}}, {7, {"X", "R"}}, {1, {"R"}}, {4, {"LEFT", "R"}}, {5, {"LEFT", "X", "R"}},
  {6, {"LEFT", "R"}}, {8, {"LEFT", "X", "R"}}, {6, {"LEFT", "R"}}, {5, {"LEFT", "X", "R"}}, {5, {"R"}},
  {5, {"X", "R"}}, {4, {"R"}}, {6, {"X", "R"}}, {1, {"R"}}, {4, {"UP", "RIGHT", "R"}},
  {1, {"UP", "RIGHT", "X", "R"}}, {1, {"RIGHT", "X", "R"}}, {3, {"X", "R"}}, {5, {"R"}}, {7, {"X", "R"}},
  {1, {"R"}}, {10, {"RIGHT", "R"}}, {2, {"UP", "RIGHT", "R"}}, {6, {"UP", "RIGHT", "X", "R"}}, {6, {"UP", "RIGHT", "R"}},
  {7, {"UP", "RIGHT", "X", "R"}}, {1, {"UP", "RIGHT", "R"}}, {3, {"UP", "RIGHT"}}, {2, {"UP"}}, {6, {"UP", "X"}},
  {5, {"UP"}}, {1, {"UP", "LEFT"}}, {2, {"UP", "LEFT", "X"}}, {6, {"LEFT", "X"}}, {2, {"LEFT"}},
  {1, {"UP", "LEFT"}}, {4, {"UP", "LEFT", "X"}}, {1, {"UP", "X"}}, {6, {"UP"}}, {5, {"UP", "X"}},
  {5, {"UP"}}, {6, {"UP", "X"}}, {5, {"UP"}}, {7, {"UP", "X"}}, {5, {"UP"}},
  {1, {"UP", "LEFT", "R"}}, {1, {"UP", "LEFT", "X", "R"}}, {5, {"X", "R"}}, {3, {"R"}}, {2, {}},
  {2, {"X"}}, {5, {"X", "R"}}, {10, {"UP", "LEFT", "X", "R"}}, {1, {"UP", "LEFT", "R"}}, {4, {"R"}},
  {7, {"X", "R"}}, {1, {"UP", "LEFT", "X", "R"}}, {4, {"LEFT", "R"}}, {5, {"LEFT", "X", "R"}}, {6, {"R"}},
  {5, {"X", "R"}}, {4, {"R"}}, {5, {"X", "R"}}, {4, {"R"}}, {4, {"X", "R"}},
  {2, {"RIGHT", "X", "R"}}, {4, {"RIGHT", "R"}}, {6, {"RIGHT", "X", "R"}}, {3, {"RIGHT", "R"}}, {11, {"RIGHT", "X", "R"}},
  {10, {"RIGHT", "R"}}, {2, {"UP", "RIGHT", "R"}}, {1, {"UP", "R"}}, {2, {"UP", "LEFT", "R"}}, {8, {"LEFT", "R"}},
  {4, {"LEFT", "X", "R"}}, {3, {"X", "R"}}, {6, {"R"}}, {6, {"X", "R"}}, {4, {"R"}},
  {8, {"X", "R"}}, {1, {"R"}}, {3, {"X", "R"}}, {4, {"RIGHT", "X", "R"}}, {36, {"RIGHT", "R"}},
  {1, {"UP", "RIGHT", "R"}}, {1, {"UP", "R"}}, {1, {"UP", "LEFT", "R"}}, {7, {"LEFT", "R"}}, {8, {"LEFT", "X", "R"}},
  {2, {"LEFT", "R"}}, {3, {"R"}}, {5, {"X", "R"}}, {5, {"R"}}, {2, {"X", "R"}},
  {5, {"LEFT", "X", "R"}}, {10, {"LEFT", "R"}}, {8, {"LEFT", "X", "R"}}, {1, {"LEFT", "R"}}, {1, {"LEFT"}},
  {3, {"LEFT", "R"}}, {12, {"LEFT", "X", "R"}}, {20, {"LEFT", "R"}}, {1, {"R"}}, {7, {"RIGHT", "R"}},
  {4, {"RIGHT", "X", "R"}}, {4, {"X", "R"}}, {5, {"R"}}, {8, {"X", "R"}}, {9, {"R"}},
  {7, {"X", "R"}}, {8, {"R"}}, {5, {"X", "R"}}, {9, {"R"}}, {3, {"X", "R"}},
  {6, {"RIGHT", "X", "R"}}, {1, {"RIGHT", "R"}}, {3, {"R"}}, {7, {"X", "R"}}, {5, {"R"}},
  {4, {"X", "R"}}, {4, {"R"}}, {7, {"LEFT", "X", "R"}}, {27, {"LEFT", "R"}}, {1, {"DOWN", "LEFT", "R"}},
  {1, {"DOWN", "R"}}, {1, {"DOWN", "RIGHT", "R"}}, {13, {"RIGHT", "R"}}, {5, {"RIGHT", "X", "R"}}, {3, {"X", "R"}},
  {10, {"R"}}, {8, {"RIGHT", "R"}}, {5, {"RIGHT"}}, {2, {}}, {1, {"X"}},
  {17, {"X", "R"}}, {5, {"R"}}, {6, {"X", "R"}}, {3, {"R"}}, {6, {"LEFT", "X", "R"}},
  {18, {"LEFT", "R"}}, {7, {"LEFT"}}, {3, {}}, {5, {"RIGHT"}}, {1, {}},
  {11, {"B", "A"}}, {39, {"B", "RIGHT", "A"}}, {1, {"RIGHT", "A"}}, {31, {"RIGHT"}}, {1, {"UP"}},
  {4, {"LEFT"}}, {5, {"LEFT", "X"}}, {4, {"X"}}, {4, {}}, {6, {"X"}},
  {4, {}}, {9, {"X"}}, {3, {}}, {1, {"UP", "LEFT"}}, {5, {"LEFT"}},
  {2, {"LEFT", "X"}}, {7, {"X"}}, {2, {}}, {5, {"X"}}, {4, {"RIGHT", "X"}},
  {16, {"RIGHT"}}, {26, {"B", "RIGHT"}}, {1, {}}, {1, {"UP", "LEFT"}}, {3, {"LEFT"}},
  {5, {"LEFT", "A"}}, {8, {"A"}}, {4, {"A", "X"}}, {3, {"X"}}, {6, {}},
  {5, {"X"}}, {5, {}}, {5, {"X"}}, {4, {}}, {7, {"X"}},
  {8, {}}, {2, {"B", "A"}}, {15, {"B", "A", "X"}}, {1, {"A", "X"}}, {5, {}},
  {7, {"X"}}, {3, {}}, {5, {"X"}}, {5, {}}, {6, {"X"}},
  {4, {}}, {6, {"X"}}, {4, {}}, {6, {"X"}}, {2, {}},
  {1, {"X"}}, {4, {"A", "X"}}, {3, {"B", "A", "X"}}, {2, {"A", "X"}}, {2, {"A"}},
  {1, {}}, {4, {"X"}}, {6, {}}, {4, {"X"}}, {4, {}},
  {6, {"X"}}, {3, {}}, {6, {"X"}}, {4, {}}, {6, {"X"}},
  {4, {}}, {6, {"X"}}, {3, {}}, {1, {"LEFT"}}, {5, {"UP", "LEFT"}},
  {9, {"LEFT"}}, {3, {"LEFT", "A"}}, {8, {"B", "LEFT", "A"}}, {2, {"LEFT", "A"}}, {3, {"LEFT"}},
  {8, {}}, {11, {"SELECT"}}, {3, {}}, {6, {"X"}}, {5, {}},
  {5, {"X"}}, {4, {}}, {6, {"X"}}, {4, {}}, {7, {"X"}},
  {3, {}}, {6, {"X"}}, {5, {}}, {5, {"X"}}, {5, {}},
  {5, {"X"}}, {6, {}}, {5, {"X"}}, {4, {}}, {8, {"X"}},
  {3, {}}, {6, {"X"}}, {1, {"X", "R"}}, {3, {"LEFT", "X", "R"}}, {2, {"X", "R"}},
  {5, {"R"}}, {7, {"X", "R"}}, {1, {"UP", "LEFT", "X", "R"}}, {4, {"UP", "LEFT", "R"}}, {5, {"LEFT", "X", "R"}},
  {5, {"LEFT", "R"}}, {2, {"LEFT", "X", "R"}}, {5, {"X", "R"}}, {3, {"R"}}, {1, {"LEFT", "R"}},
  {6, {"LEFT", "X", "R"}}, {5, {"LEFT", "R"}}, {3, {"LEFT", "X", "R"}}, {2, {"X", "R"}}, {11, {"R"}},
  {5, {}}, {20, {"B", "A"}}, {2, {"B", "A", "X"}}, {7, {"B", "LEFT", "A", "X"}}, {2, {"B", "LEFT", "A"}},
  {1, {"LEFT"}}, {59, {}}, {10, {"LEFT"}}, {115, {}}, {32, {"A"}},
  {1, {"RIGHT", "A"}}, {22, {"RIGHT"}}, {13, {"B", "RIGHT"}}, {19, {"RIGHT"}}, {12, {}},
  {1, {"LEFT"}}, {1, {"B", "LEFT"}}, {13, {"B", "LEFT", "A"}}, {17, {"LEFT", "A"}}, {4, {"LEFT"}},
  {10, {"LEFT", "X"}}, {4, {"LEFT"}}, {7, {"LEFT", "X"}}, {4, {"LEFT"}}, {6, {"LEFT", "X"}},
  {3, {"LEFT"}}, {7, {"LEFT", "X"}}, {5, {"LEFT"}}, {2, {"LEFT", "A"}}, {7, {"B", "LEFT", "A"}},
  {20, {"LEFT", "A"}}, {41, {"LEFT"}}, {10, {"LEFT", "L"}}, {16, {"LEFT"}}, {158, {}},
  {4, {"LEFT"}}, {66, {"B", "LEFT"}}, {14, {"B", "LEFT", "R"}}, {87, {"B", "LEFT", "X", "R"}}, {3, {"B", "LEFT", "R"}},
  {8, {"B", "LEFT", "A", "R"}}, {4, {"B", "LEFT", "A"}}, {4, {"LEFT", "A"}}, {14, {"LEFT"}}, {11, {"LEFT", "X"}},
  {28, {"LEFT"}}, {221, {}},
}


function M.play_morph_to_construction(session)
  local st = session.state
  if num(st.room_id) == M.ROOM_CONSTRUCTION then
    return
  end
  if num(st.room_id) == M.ROOM_MORPH and num(st.game_state) == M.GS_ORDINARY then
    local i
    for i = 1, 400 do
      st = session.state
      if num(st.room_id) ~= M.ROOM_MORPH then
        break
      end
      local x = num(st.samus_x or st.x)
      if x < 1900 then
        session:step({"RIGHT", "B"}, "morph_to_construction")
      else
        session:step({"RIGHT", "X"}, "morph_to_construction_door")
      end
    end
  end
  wait_until(session, function(s)
    return num(s.room_id) == M.ROOM_CONSTRUCTION
  end, 180, "morph_to_construction_transition_settle")
end

function M.play_two_missile_detour(session)
  M.play_morph_to_construction(session)
  if not has_mask(session.state.collected_items, M.MORPH_BALL_MASK) then
    timeout("two_missile_detour missing morph: " .. brief(session.state))
  end
  play_rle(session, TWO_MISSILE_RLE, "policy_two_missile_detour")
  if num(session.state.max_missiles) < 10 then
    timeout("two_missile_detour missed 10-pack: " .. brief(session.state))
  end
end

function M.play_construction_return(session)
  play_rle(session, CONSTRUCTION_RETURN_RLE, "policy_construction_and_morph_return")
end

function M.play_elevator_return(session)
  play_rle(session, ELEVATOR_RETURN_RLE, "policy_elevator_return")
end

function M.play_pit_natural_entry(session)
  wait_until(session, function(s)
    local ordinary = s.phase == "ordinary_gameplay"
      or (num(s.game_state) == M.GS_ORDINARY and num(s.door_transition) == 0)
    return num(s.room_id) == M.ROOM_PIT
      and ordinary
      and num(s.samus_x or s.x) == 693
      and num(s.samus_y or s.y) == 187
  end, 600, "pit_natural_entry_alignment")
  if num(session.state.selected_item) ~= 0 then
    session:step({"SELECT"}, "pit_weapon_selection_normalize")
  else
    session:step({}, "pit_weapon_already_beam")
  end
  span(session, {}, 9, "pit_grounded_settle")
  if num(session.state.selected_item) ~= 0 then
    bt.select_weapon(session, 0)
    span(session, {}, 9, "pit_grounded_settle")
  end
  if num(session.state.selected_item) ~= 0 then
    timeout("Pit weapon selection did not return to beam: " .. brief(session.state))
  end
end

function M.play_pit_to_post_torizo(session)
  -- Clean hybrid: policy prefix until natural BT activation, kite fight,
  -- then hash-pinned exit tail (last 2000f of pit_to_post_torizo).
  play_rle(
    session,
    PIT_TO_POST_TORIZO_RLE,
    "policy_pit_to_torizo_replay",
    { stop_when = bt.fight_ready }
  )
  if not bt.fight_ready(session.state) then
    timeout(
      "clean pit_to_post_torizo: policy prefix ended without BT activation; "
        .. brief(session.state)
        .. " hp=" .. tostring(session.state.enemy0_hp)
        .. " sm=0x" .. string.format("%04X", num(session.state.enemy0_spritemap))
    )
  end
  local fight = bt.play_fight(session, nil, true, true)
  if fight.outcome ~= "bomb_torizo_defeated" then
    timeout(
      "clean Bomb Torizo fight failed: " .. tostring(fight.outcome)
        .. " min_hp=" .. tostring(fight.min_enemy_hp)
        .. " final_hp=" .. tostring(fight.final_enemy_hp)
        .. " frames=" .. tostring(fight.action_frames)
    )
  end
  play_rle(
    session,
    PIT_TO_POST_TORIZO_RLE,
    "policy_pit_to_torizo_replay_tail",
    { skip = M.PIT_TO_POST_TORIZO_EXIT_TAIL_START }
  )
  return fight
end

function M.leave_ok(state)
  local ordinary = state.phase == "ordinary_gameplay"
    or (num(state.game_state) == M.GS_ORDINARY and num(state.door_transition) == 0)
  return num(state.room_id) == M.LEAVE_ROOM
    and ordinary
    and has_mask(state.collected_items, M.LEAVE_ITEMS_MASK)
    and num(state.max_missiles) >= 10
end

function M.play_morph_to_bombs_exit(session)
  if not has_mask(session.state.collected_items, M.MORPH_BALL_MASK) then
    timeout("play_morph_to_bombs_exit expected morph: " .. brief(session.state))
  end
  M.play_two_missile_detour(session)
  M.play_construction_return(session)
  M.play_elevator_return(session)
  M.play_pit_natural_entry(session)
  local fight = M.play_pit_to_post_torizo(session)
  local st = session.state
  if not M.leave_ok(st) then
    timeout(
      "bombs leave missed Parlor 0x92FD gs=8 items&0x1000: " .. brief(st)
    )
  end
  return {
    room = num(st.room_id),
    gs = num(st.game_state),
    x = num(st.samus_x or st.x),
    y = num(st.samus_y or st.y),
    pose = num(st.pose),
    items = num(st.collected_items),
    health = num(st.health),
    max_missiles = num(st.max_missiles),
    fight = fight,
  }
end

M.play = M.play_morph_to_bombs_exit
M.play_rle = play_rle
M.hold = hold
M.wait_until = wait_until
M.span = span
M.ensure_morph = morph_roll.ensure_morph
M.bomb_roll_left_safe = morph_roll.bomb_roll_left_safe

return M
