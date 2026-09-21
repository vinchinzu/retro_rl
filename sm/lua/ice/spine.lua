-- Compose Business → Ice Beam collect following POST_SUPERS_SPINE hop order
-- for bat_cave / speed / wave / ice (frog sibling omitted).
-- Lua 5.1. Session: step / hold / wait_until / span.
-- Speed-block rooms require the sibling shinespark module.

local shinespark = require("skills.shinespark")

local cathedral = require("norfair.cathedral")
local rising_tide = require("norfair.rising_tide")
local to_bat_cave = require("norfair.to_bat_cave")
local to_speed = require("norfair.to_speed")
local speed_return = require("norfair.speed_return")

local bubble_to_single = require("wave.bubble_to_single")
local single_to_double = require("wave.single_to_double")
local double_to_wave = require("wave.double_to_wave")
local wave_to_double = require("wave.wave_to_double")
local double_to_single = require("wave.double_to_single")
local single_to_bubble = require("wave.single_to_bubble")
local bubble_to_farm = require("wave.bubble_to_farm")
local farm_to_speedway = require("wave.farm_to_speedway")
local speedway_to_frog = require("wave.speedway_to_frog")
local frog_to_business = require("wave.frog_to_business")

local business_to_gate = require("ice.business_to_gate")
local gate_to_acid = require("ice.gate_to_acid")
local acid_to_snake = require("ice.acid_to_snake")
local snake_to_ice = require("ice.snake_to_ice")
local G = require("ice.geometry")

local M = {}
M.shinespark = shinespark

-- POST_SUPERS_SPINE hop ids for bat_cave → speed → wave → ice.
M.HOP_ORDER = {
  -- bat_cave (K4.4; sibling of frog under business)
  "business_to_cathedral_entrance",
  "cathedral_entrance_to_cathedral",
  "cathedral_to_rising_tide",
  "rising_tide_to_bubble",
  "bubble_to_bat_cave",
  -- speed (K4.5)
  "bat_cave_to_speed_hall",
  "speed_hall_to_speed",
  -- wave (K4.7–K4.10)
  "speed_return_to_bubble",
  "bubble_to_single_chamber",
  "single_to_double_chamber",
  "double_chamber_to_wave",
  -- ice (K4.11; Wave return → Business Super → Gate → Acid → Snake → Ice)
  "wave_to_double_chamber",
  "double_to_single_chamber",
  "single_to_bubble",
  "bubble_to_farm",
  "farm_to_speedway",
  "speedway_to_frog_save",
  "frog_save_to_business",
  "business_to_ice_gate",
  "ice_gate_to_acid",
  "ice_acid_to_snake",
  "ice_snake_to_ice",
}

-- Ice collect leave RAM (verified Python: post_ice_snake_to_ice_pure /
-- STATUS `--to ice` 148167f). Room 0xA890, gs=8, Ice bit 0x0002 on beams.
-- Typical post-collect pin ~(187,120) pose 81, beams 0x1007 (Charge+Spazer+Wave+Ice),
-- items 0x3105 (Morph+Varia+Hi-Jump+Bombs+Speed), timer_type 0.
M.ICE_COLLECT_LEAVE = {
  room_id = 0xA890,
  room_id_hex = "0xA890",
  game_state = 8,
  door_transition = 0,
  samus_x = 187,
  samus_y = 120,
  pose = 81,
  collected_beams = 0x1007,
  ice_mask = G.ICE_BEAM_MASK, -- 0x0002
  collected_items = 0x3105,
  timer_type = 0,
  health_min = 1,
}

local function require_ice_leave(session)
  local st = session.state
  if st.room_id ~= 0xA890 then
    error(string.format(
      "play_business_to_ice: expected Ice room 0xA890, got 0x%04X xy=(%d,%d)",
      st.room_id, st.samus_x, st.samus_y
    ))
  end
  if st.game_state ~= 8 then
    error(string.format("play_business_to_ice: expected gs=8, got %d", st.game_state))
  end
  if not G.has_ice(st) then
    error(string.format(
      "play_business_to_ice: Ice bit missing; beams=0x%04X (want mask 0x%04X)",
      st.collected_beams or 0, G.ICE_BEAM_MASK
    ))
  end
  return st
end

function M.play_business_to_ice(session)
  -- bat_cave
  cathedral.play_business_to_cathedral_entrance(session)
  cathedral.play_cathedral_entrance_to_cathedral(session)
  cathedral.play_cathedral_to_rising_tide(session)
  rising_tide.play_rising_tide_to_bubble(session)
  to_bat_cave.play_bubble_to_bat_cave(session)
  -- speed
  to_speed.play_bat_cave_to_speed_hall(session)
  to_speed.play_speed_hall_to_speed(session)
  -- wave
  speed_return.play_speed_return_to_bubble(session)
  bubble_to_single.play_bubble_to_single_chamber(session)
  single_to_double.play_single_to_double_chamber(session)
  double_to_wave.play_double_chamber_to_wave(session)
  -- ice (Wave return → Business floor → Super → Gate → Acid → Snake → Ice PLM)
  wave_to_double.play_wave_to_double_chamber(session)
  double_to_single.play_double_to_single_chamber(session)
  single_to_bubble.play_single_to_bubble(session)
  bubble_to_farm.play_bubble_to_farm(session)
  farm_to_speedway.play_farm_to_speedway(session)
  speedway_to_frog.play_speedway_to_frog_save(session)
  frog_to_business.play_frog_save_to_business(session)
  business_to_gate.play_business_to_ice_gate(session)
  gate_to_acid.play_ice_gate_to_acid(session)
  acid_to_snake.play_ice_acid_to_snake(session)
  snake_to_ice.play_ice_snake_to_ice(session)
  return require_ice_leave(session)
end

local ice_to_snake = require("ice.ice_to_snake")
local snake_to_tutorial = require("ice.snake_to_tutorial")
local tutorial_to_gate = require("ice.tutorial_to_gate")
local gate_to_business = require("ice.gate_to_business")

-- Ice room 0xA890 → Business 0xA7DE (POST_ICE_SPINE prefix).
function M.play_ice_return(session)
  ice_to_snake.play_ice_to_snake(session)
  snake_to_tutorial.play_ice_snake_to_tutorial(session)
  tutorial_to_gate.play_ice_tutorial_to_gate(session)
  gate_to_business.play_ice_gate_to_business(session)
  return session.state
end

M.play = M.play_business_to_ice

return M
