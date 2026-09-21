-- Morph hash-pinned room seeds (policies/morph/seg00..seg05).
-- SNES-12 order B Y select start up down left right A X L R.
-- Stored as RLE {count, {12 ints}}; play() expands and feeds session:raw_actions.
-- Lua 5.1.

local M = {}

M.TAPE = {"B","Y","select","start","up","down","left","right","A","X","L","R"}

-- Elevator standing-flag phase: 17 idle frames before the BB elev seed.
M.ELEVATOR_ALIGN_FRAMES = 17
M.LANDING_ADAPTER_SHOOT = 90
M.LANDING_ADAPTER_ENTER = 180
M.TRANSITION_SETTLE = 180

-- Landing Site 0x91F8: 401 frames, 8 runs from seg00_landing_site.json
M.landing_site = {
  {48, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
  {25, {0,0,0,0,0,0,1,0,0,0,0,0}}, -- LEFT
  {249, {1,0,0,0,0,0,1,0,0,0,0,0}}, -- B+LEFT
  {2, {1,0,0,0,0,0,1,0,0,1,0,0}}, -- B+LEFT+X
  {6, {0,0,0,0,0,0,1,0,0,1,0,0}}, -- LEFT+X
  {3, {1,0,0,0,0,0,1,0,0,1,0,0}}, -- B+LEFT+X
  {66, {1,0,0,0,0,0,1,0,0,0,0,0}}, -- B+LEFT
  {2, {0,0,0,0,0,0,1,0,0,0,0,0}}, -- LEFT
}

-- Parlor 0x92FD: 1019 frames, 51 runs from seg01_parlor.json
-- Extended 50 frames past boundary for room transition detection
M.parlor = {
  {19, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
  {2, {1,0,0,0,0,0,0,0,0,0,0,0}}, -- B
  {141, {1,0,0,0,0,0,1,0,0,0,0,0}}, -- B+LEFT
  {33, {1,0,0,0,0,0,1,0,1,0,0,0}}, -- B+LEFT+A
  {139, {1,0,0,0,0,0,1,0,0,0,0,0}}, -- B+LEFT
  {3, {1,0,0,0,0,0,1,0,1,0,0,0}}, -- B+LEFT+A
  {9, {0,0,0,0,0,0,1,0,1,0,0,0}}, -- LEFT+A
  {3, {1,0,0,0,0,0,1,0,1,0,0,0}}, -- B+LEFT+A
  {27, {1,0,0,0,0,0,1,0,0,0,0,0}}, -- B+LEFT
  {2, {1,0,0,0,0,1,1,0,0,0,0,0}}, -- B+DOWN+LEFT
  {5, {1,0,0,0,0,1,0,0,0,0,0,0}}, -- B+DOWN
  {10, {1,0,0,0,0,1,0,1,0,0,0,0}}, -- B+DOWN+RIGHT
  {14, {1,0,0,0,0,0,0,1,0,0,0,0}}, -- B+RIGHT
  {1, {0,0,0,0,0,0,0,1,0,0,0,0}}, -- RIGHT
  {24, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
  {22, {0,0,0,0,0,0,0,1,0,0,0,0}}, -- RIGHT
  {4, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
  {20, {0,0,0,0,0,0,1,0,0,0,0,0}}, -- LEFT
  {5, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
  {6, {0,0,0,0,0,0,0,1,0,0,0,0}}, -- RIGHT
  {29, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
  {15, {0,0,0,0,0,0,0,1,0,0,0,0}}, -- RIGHT
  {4, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
  {41, {0,0,0,0,0,0,1,0,0,0,0,0}}, -- LEFT
  {3, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
  {26, {0,0,0,0,0,0,0,1,0,0,0,0}}, -- RIGHT
  {5, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
  {29, {0,0,0,0,0,0,1,0,0,0,0,0}}, -- LEFT
  {3, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
  {14, {0,0,0,0,0,0,0,1,0,0,0,0}}, -- RIGHT
  {4, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
  {9, {0,0,0,0,0,0,1,0,0,0,0,0}}, -- LEFT
  {3, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
  {32, {0,0,0,0,0,0,0,1,0,0,0,0}}, -- RIGHT
  {3, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
  {41, {0,0,0,0,0,0,1,0,0,0,0,0}}, -- LEFT
  {3, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
  {26, {0,0,0,0,0,0,0,1,0,0,0,0}}, -- RIGHT
  {22, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
  {9, {0,0,0,0,0,0,1,0,0,0,0,0}}, -- LEFT
  {9, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
  {12, {0,0,0,0,0,0,1,0,0,0,0,1}}, -- LEFT+R
  {10, {0,0,0,0,0,0,1,0,0,1,0,1}}, -- LEFT+X+R
  {5, {0,0,0,0,0,1,1,0,0,1,0,1}}, -- DOWN+LEFT+X+R
  {10, {0,0,0,0,0,1,0,0,0,0,0,1}}, -- DOWN+R
  {5, {0,0,0,0,0,1,0,0,0,0,0,0}}, -- DOWN
  {8, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
  {8, {0,0,0,0,0,0,0,0,0,0,1,0}}, -- L
  {10, {0,0,0,0,0,0,0,0,0,1,1,0}}, -- X+L
  {15, {0,0,0,0,0,0,0,0,0,0,1,0}}, -- L
  {117, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
}

-- Climb 0x96BA: 823 frames, 17 runs from seg02_climb.json
-- Extended 50 frames past boundary for room transition detection
M.climb = {
  {147, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
  {4, {0,0,0,0,0,0,0,1,0,0,0,0}}, -- RIGHT
  {10, {1,0,0,0,0,0,0,1,0,0,0,0}}, -- B+RIGHT
  {17, {0,0,0,0,0,0,0,1,0,0,0,0}}, -- RIGHT
  {9, {0,0,0,0,0,0,0,1,1,0,0,0}}, -- RIGHT+A
  {68, {0,0,0,0,0,0,0,1,0,0,0,0}}, -- RIGHT
  {358, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
  {21, {0,0,0,0,0,0,1,0,0,0,0,0}}, -- LEFT
  {19, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
  {23, {0,0,0,0,0,0,0,1,0,0,0,0}}, -- RIGHT
  {6, {0,0,0,0,0,1,0,1,0,0,0,0}}, -- DOWN+RIGHT
  {3, {0,0,0,0,0,1,0,0,0,0,0,0}}, -- DOWN
  {7, {0,0,0,0,0,1,0,0,0,1,0,0}}, -- DOWN+X
  {27, {0,0,0,0,0,1,0,0,0,0,0,0}}, -- DOWN
  {4, {0,0,0,0,0,1,0,1,0,0,0,0}}, -- DOWN+RIGHT
  {50, {0,0,0,0,0,0,0,1,0,0,0,0}}, -- RIGHT
  {50, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
}

-- Pit Room 0x975C: 810 frames, 30 runs from seg03_pit_room.json
M.pit_room = {
  {121, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
  {12, {0,0,0,0,0,0,0,1,0,0,0,0}}, -- RIGHT
  {17, {1,0,0,0,0,0,0,1,0,0,0,0}}, -- B+RIGHT
  {15, {1,0,0,0,0,0,0,1,1,0,0,0}}, -- B+RIGHT+A
  {33, {1,0,0,0,0,0,0,1,0,0,0,0}}, -- B+RIGHT
  {16, {1,0,0,0,0,0,0,1,1,0,0,0}}, -- B+RIGHT+A
  {26, {1,0,0,0,0,0,0,1,0,0,0,0}}, -- B+RIGHT
  {12, {1,0,0,0,0,0,0,1,1,0,0,0}}, -- B+RIGHT+A
  {36, {1,0,0,0,0,0,0,1,0,0,0,0}}, -- B+RIGHT
  {15, {1,0,0,0,0,0,0,1,1,0,0,0}}, -- B+RIGHT+A
  {9, {1,0,0,0,0,0,0,1,0,0,0,0}}, -- B+RIGHT
  {18, {1,0,0,0,0,0,0,1,1,0,0,0}}, -- B+RIGHT+A
  {6, {1,0,0,0,0,0,0,1,0,0,0,0}}, -- B+RIGHT
  {26, {1,0,0,0,0,0,0,1,1,0,0,0}}, -- B+RIGHT+A
  {60, {1,0,0,0,0,0,0,1,0,0,0,0}}, -- B+RIGHT
  {42, {1,0,0,0,0,0,0,1,1,0,0,0}}, -- B+RIGHT+A
  {22, {1,0,0,0,0,0,0,1,0,0,0,0}}, -- B+RIGHT
  {24, {1,0,0,0,0,0,0,1,1,0,0,0}}, -- B+RIGHT+A
  {47, {1,0,0,0,0,0,0,1,0,0,0,0}}, -- B+RIGHT
  {15, {1,0,0,0,0,0,0,1,1,0,0,0}}, -- B+RIGHT+A
  {50, {1,0,0,0,0,0,0,1,0,0,0,0}}, -- B+RIGHT
  {43, {1,0,0,0,0,0,0,1,1,0,0,0}}, -- B+RIGHT+A
  {3, {1,0,0,0,0,0,0,1,1,1,0,0}}, -- B+RIGHT+A+X
  {7, {0,0,0,0,0,0,0,1,1,1,0,0}}, -- RIGHT+A+X
  {7, {0,0,0,0,0,0,0,1,0,1,0,0}}, -- RIGHT+X
  {23, {0,0,0,0,0,0,0,1,0,0,0,0}}, -- RIGHT
  {1, {1,0,0,0,0,0,0,1,0,0,0,0}}, -- B+RIGHT
  {28, {1,0,0,0,0,0,0,1,1,0,0,0}}, -- B+RIGHT+A
  {2, {1,0,0,0,0,0,0,1,0,0,0,0}}, -- B+RIGHT
  {74, {0,0,0,0,0,0,0,1,0,0,0,0}}, -- RIGHT
}

-- BB Elev Hallway 0x97B5: 285 frames, 8 runs from seg04_bb_elev_hallway.json
-- Uses seg04_0x97B5_fixed.state (proper elevator init from door transition replay)
M.bb_elev_hallway = {
  {71, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
  {47, {1,0,0,0,0,0,0,1,0,0,0,0}}, -- B+RIGHT
  {9, {0,0,0,0,0,0,0,1,0,0,0,0}}, -- RIGHT
  {15, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
  {7, {0,0,0,0,0,1,0,0,0,0,0,0}}, -- DOWN
  {3, {0,0,0,0,0,1,1,0,0,0,0,0}}, -- DOWN+LEFT
  {7, {0,0,0,0,0,0,1,0,0,0,0,0}}, -- LEFT
  {126, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
}

-- Morph Ball Room 0x9E9F: 1843 frames, 84 runs from seg05_morph_ball_room.json
M.morph_ball_room = {
  {565, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
  {2, {0,0,0,0,0,0,1,0,0,0,0,0}}, -- LEFT
  {40, {1,0,0,0,0,0,1,0,0,0,0,0}}, -- B+LEFT
  {68, {1,0,0,0,0,0,1,0,1,0,0,0}}, -- B+LEFT+A
  {2, {0,0,0,0,0,0,1,0,1,0,0,0}}, -- LEFT+A
  {41, {0,0,0,0,0,0,1,0,0,0,0,0}}, -- LEFT
  {28, {1,0,0,0,0,0,1,0,0,0,0,0}}, -- B+LEFT
  {10, {0,0,0,0,0,0,1,0,0,0,0,0}}, -- LEFT
  {13, {0,0,0,0,0,0,1,0,1,0,0,0}}, -- LEFT+A
  {9, {0,0,0,0,0,0,1,0,0,0,0,0}}, -- LEFT
  {10, {0,0,0,0,0,0,1,0,0,1,0,0}}, -- LEFT+X
  {22, {0,0,0,0,0,0,1,0,0,0,0,0}}, -- LEFT
  {12, {0,0,0,0,0,0,1,0,1,0,0,0}}, -- LEFT+A
  {4, {0,0,0,0,0,0,1,0,1,1,0,0}}, -- LEFT+A+X
  {8, {0,0,0,0,0,0,1,0,0,1,0,0}}, -- LEFT+X
  {8, {0,0,0,0,0,0,1,0,0,0,0,0}}, -- LEFT
  {9, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
  {14, {0,0,0,0,0,0,0,0,1,0,0,0}}, -- A
  {31, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
  {7, {0,0,0,0,0,0,0,0,1,0,0,0}}, -- A
  {32, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
  {15, {0,0,0,0,0,0,0,0,1,0,0,0}}, -- A
  {33, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
  {12, {0,0,0,0,0,0,0,0,1,0,0,0}}, -- A
  {9, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
  {17, {0,0,0,0,0,0,0,1,0,0,0,0}}, -- RIGHT
  {17, {1,0,0,0,0,0,0,1,0,0,0,0}}, -- B+RIGHT
  {5, {0,0,0,0,0,0,0,1,0,0,0,0}}, -- RIGHT
  {14, {0,0,0,0,0,0,0,1,1,0,0,0}}, -- RIGHT+A
  {15, {0,0,0,0,0,0,0,1,0,0,0,0}}, -- RIGHT
  {15, {0,0,0,0,0,0,0,1,1,0,0,0}}, -- RIGHT+A
  {7, {0,0,0,0,0,0,0,1,0,0,0,0}}, -- RIGHT
  {11, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
  {15, {0,0,0,0,0,0,0,0,0,1,0,0}}, -- X
  {5, {0,0,0,0,0,0,0,0,1,1,0,0}}, -- A+X
  {6, {0,0,0,0,0,0,0,0,1,0,0,0}}, -- A
  {5, {0,0,0,0,0,0,0,1,1,0,0,0}}, -- RIGHT+A
  {13, {0,0,0,0,0,0,0,1,0,0,0,0}}, -- RIGHT
  {4, {1,0,0,0,0,0,0,1,0,0,0,0}}, -- B+RIGHT
  {9, {1,0,0,0,0,0,0,0,0,0,0,0}}, -- B
  {1, {1,0,0,0,0,0,0,0,1,0,0,0}}, -- B+A
  {17, {0,0,0,0,0,0,0,0,1,0,0,0}}, -- A
  {13, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
  {9, {0,0,0,0,0,0,0,0,1,0,0,0}}, -- A
  {9, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
  {20, {0,0,0,0,0,0,0,1,0,0,0,0}}, -- RIGHT
  {5, {1,0,0,0,0,0,0,1,0,0,0,0}}, -- B+RIGHT
  {16, {1,0,0,0,0,0,0,1,1,0,0,0}}, -- B+RIGHT+A
  {2, {1,0,0,0,0,0,0,1,0,0,0,0}}, -- B+RIGHT
  {15, {0,0,0,0,0,0,0,1,0,0,0,0}}, -- RIGHT
  {14, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
  {10, {0,0,0,0,0,1,0,0,0,0,0,0}}, -- DOWN
  {15, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
  {2, {0,0,0,0,0,0,0,1,0,0,0,0}}, -- RIGHT
  {28, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
  {8, {0,0,0,0,0,0,0,1,0,0,0,0}}, -- RIGHT
  {17, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
  {18, {0,0,0,0,0,1,0,0,0,0,0,0}}, -- DOWN
  {11, {0,0,0,0,0,1,0,0,0,1,0,0}}, -- DOWN+X
  {12, {0,0,0,0,0,1,0,0,0,0,0,0}}, -- DOWN
  {11, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
  {13, {0,0,0,0,0,1,0,0,0,0,0,0}}, -- DOWN
  {11, {0,0,0,0,0,1,0,0,0,1,0,0}}, -- DOWN+X
  {1, {0,0,0,0,0,1,0,0,0,0,0,0}}, -- DOWN
  {13, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
  {9, {0,0,0,0,0,1,0,0,0,0,0,0}}, -- DOWN
  {5, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
  {8, {0,0,0,0,0,1,0,0,0,0,0,0}}, -- DOWN
  {11, {0,0,0,0,0,0,0,0,0,0,0,0}}, -- idle
  {38, {0,0,0,0,0,0,0,1,0,0,0,0}}, -- RIGHT
  {16, {0,0,0,0,1,0,0,1,0,0,0,0}}, -- UP+RIGHT
  {1, {1,0,0,0,1,0,0,1,0,0,0,0}}, -- B+UP+RIGHT
  {5, {1,0,0,0,0,0,0,1,0,0,0,0}}, -- B+RIGHT
  {16, {1,0,0,0,0,0,0,1,1,0,0,0}}, -- B+RIGHT+A
  {155, {1,0,0,0,0,0,0,1,0,0,0,0}}, -- B+RIGHT
  {32, {1,0,0,0,0,0,0,1,1,0,0,0}}, -- B+RIGHT+A
  {6, {1,0,0,0,0,0,0,1,1,1,0,0}}, -- B+RIGHT+A+X
  {4, {0,0,0,0,0,0,0,1,1,1,0,0}}, -- RIGHT+A+X
  {3, {0,0,0,0,0,0,0,1,1,0,0,0}}, -- RIGHT+A
  {7, {0,0,0,0,0,0,0,1,0,0,0,0}}, -- RIGHT
  {6, {0,0,0,0,0,0,0,1,0,1,0,0}}, -- RIGHT+X
  {6, {0,0,0,0,0,0,0,1,0,0,0,0}}, -- RIGHT
  {7, {0,0,0,0,0,0,0,1,0,1,0,0}}, -- RIGHT+X
  {55, {0,0,0,0,0,0,0,1,0,0,0,0}}, -- RIGHT
}

M.FRAMES = {
  landing_site = 401,
  parlor = 1019,
  climb = 823,
  pit_room = 810,
  bb_elev_hallway = 285,
  morph_ball_room = 1843,
}

local function format_state(st)
  if type(st) ~= "table" then
    return tostring(st)
  end
  return string.format(
    "room=0x%04X gs=%s xy=(%s,%s) pose=%s items=0x%04X",
    tonumber(st.room_id) or 0,
    tostring(st.game_state),
    tostring(st.samus_x or st.x),
    tostring(st.samus_y or st.y),
    tostring(st.pose),
    tonumber(st.collected_items) or 0
  )
end

-- Expand RLE {count, {12 ints}} into a dense SNES-12 tape.
function M.expand(tape)
  if type(tape) == "string" then
    tape = M[tape]
  end
  if type(tape) ~= "table" then
    error("morph seed expand: unknown tape")
  end
  local out = {}
  local i, j
  for i = 1, #tape do
    local run = tape[i]
    local n = run[1]
    local frame = run[2]
    for j = 1, n do
      out[#out + 1] = frame
    end
  end
  return out
end

function M.frame_count(tape)
  if type(tape) == "string" then
    return M.FRAMES[tape] or 0
  end
  local n = 0
  local i
  for i = 1, #tape do
    n = n + (tape[i][1] or 0)
  end
  return n
end

local function span_idle(session, frames, reason)
  if session.span then
    session:span({}, frames, reason)
    return
  end
  if session.hold then
    session:hold(frames, {}, reason)
    return
  end
  local i
  for i = 1, frames do
    session:step({}, reason)
  end
end

-- Play one named seed. opts.align_elevator / opts.landing_door_adapter
-- match early_spine._play_seed_to_room.
function M.play(session, name, target_room, opts)
  opts = opts or {}
  local tape = M[name]
  if type(tape) ~= "table" or M.FRAMES[name] == nil then
    error("unknown morph seed: " .. tostring(name))
  end
  if opts.align_elevator then
    span_idle(session, M.ELEVATOR_ALIGN_FRAMES, "elevator_seed_alignment")
  end
  local source_room = session.state.room_id
  local actions = M.expand(tape)
  session:raw_actions(actions, "seed_" .. name)

  if opts.landing_door_adapter and session.state.room_id == source_room then
    local i
    for i = 1, M.LANDING_ADAPTER_SHOOT do
      session:step({"LEFT", "X"}, "landing_door_adapter_shoot")
      if session.state.game_state == 11 then
        break
      end
    end
    for i = 1, M.LANDING_ADAPTER_ENTER do
      session:step({"LEFT"}, "landing_door_adapter_enter")
      if session.state.room_id ~= source_room then
        break
      end
    end
  end

  if target_room ~= nil and session.state.room_id ~= target_room then
    if session.state.game_state ~= 11 then
      error(string.format(
        "seed %s missed 0x%04X: %s",
        tostring(name),
        tonumber(target_room) or 0,
        format_state(session.state)
      ))
    end
    session:wait_until(function(s)
      return s.room_id == target_room
    end, M.TRANSITION_SETTLE, "seed_" .. name .. "_transition_settle")
  end
end

M.format_state = format_state

return M

