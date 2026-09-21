-- Morph spine: Landing Site → Morph Ball collect.
--
-- Ceres prefix lives in require("ceres.spine") / require("ceres.outbound");
-- do not re-port it here. play_ship_to_morph is the vanilla suffix
-- (Python early_spine.play_ship_to_morph).
-- Lua 5.1.

local ram = require("ram")
local rooms = require("rooms")
local seeds = require("morph.seeds")
local parlor = require("morph.parlor")
local climb = require("morph.climb")

local M = {}

local ROOM_LANDING_SITE = rooms.ROOM_LANDING_SITE
local ROOM_PARLOR = rooms.ROOM_PARLOR
local ROOM_CLIMB = rooms.ROOM_CLIMB
local ROOM_PIT = rooms.ROOM_PIT
local ROOM_BLUE_BRINSTAR_ELEVATOR = rooms.ROOM_BLUE_BRINSTAR_ELEVATOR
local ROOM_MORPH = rooms.ROOM_MORPH

local MORPH_BALL_MASK = ram.MORPH_BALL_MASK
local GS_ORDINARY = ram.GS_ORDINARY
local ADDR_ELEV_STATUS = 0x0E16 -- $0E16 Samus-on-elevator flag

local function num(v, default)
  v = tonumber(v)
  if v == nil then
    return default or 0
  end
  return v
end

local function format_state(st)
  return seeds.format_state(st)
end

local function has_morph(state)
  if ram.morph_ball then
    return ram.morph_ball(state)
  end
  return ram.band(state.collected_items, MORPH_BALL_MASK) ~= 0
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

local function elev_status()
  return num(ram.u8(ADDR_ELEV_STATUS))
end

function M.play_landing_to_parlor(session)
  seeds.play(session, "landing_site", ROOM_PARLOR, {landing_door_adapter = true})
end

function M.play_parlor_to_climb(session)
  if parlor.parlor_moonfall_enabled(session) then
    parlor.play_parlor_to_climb_moonfall(session)
    return
  end
  seeds.play(session, "parlor", ROOM_CLIMB)
end

function M.play_climb_to_pit(session)
  if climb.climb_moonfall_enabled(session) then
    climb.play_climb_to_pit_moonfall(session)
    return
  end
  seeds.play(session, "climb", ROOM_PIT)
end

function M.play_pit_to_elevator(session)
  seeds.play(session, "pit_room", ROOM_BLUE_BRINSTAR_ELEVATOR)
end

local function play_bb_elev_reactive(session)
  -- Settle door if still transitioning.
  local i
  for i = 1, 200 do
    local st = session.state
    if num(st.room_id) ~= ROOM_BLUE_BRINSTAR_ELEVATOR then
      if num(st.room_id) == ROOM_MORPH then
        return
      end
      break
    end
    if num(st.game_state) == GS_ORDINARY and num(st.door_transition) == 0 then
      break
    end
    session:step({"RIGHT"}, "bb_elev_door")
  end

  -- Walk onto pad center (~x128–145). Product boards from x≈142.
  for i = 1, 120 do
    local st = session.state
    if num(st.room_id) == ROOM_MORPH then
      return
    end
    if num(st.room_id) ~= ROOM_BLUE_BRINSTAR_ELEVATOR then
      break
    end
    local x = num(st.samus_x or st.x)
    if 128 <= x and x <= 148 and math.abs(num(st.velocity_x)) <= 1 then
      break
    end
    if x < 128 then
      session:step({"RIGHT", "B"}, "bb_elev_to_pad")
    elseif x > 148 then
      session:step({"LEFT"}, "bb_elev_to_pad")
    else
      session:step({"RIGHT"}, "bb_elev_to_pad")
    end
  end

  for i = 1, 8 do
    session:step({}, "bb_elev_plant")
  end

  -- Wait for elev standing flag ($0E16==1) then DOWN (snaps to elev pose 0).
  local boarded = false
  for i = 1, 40 do
    local st = session.state
    if num(st.room_id) == ROOM_MORPH then
      return
    end
    if num(st.pose) == 0 or num(st.samus_y or st.y) > 150 then
      boarded = true
      break
    end
    if elev_status() == 1 then
      session:step({"DOWN"}, "bb_elev_board")
    else
      session:step({}, "bb_elev_wait_flag")
    end
  end
  if not boarded and num(session.state.pose) ~= 0 then
    -- One more forced DOWN pair across both parity frames.
    session:step({"DOWN"}, "bb_elev_board")
    session:step({"DOWN"}, "bb_elev_board")
  end

  -- Ride / wait Morph room.
  for i = 1, 400 do
    local st = session.state
    if num(st.room_id) == ROOM_MORPH then
      if num(st.game_state) == GS_ORDINARY then
        return
      end
      session:step({}, "bb_elev_morph_settle")
    elseif num(st.room_id) ~= ROOM_BLUE_BRINSTAR_ELEVATOR then
      break
    elseif num(st.pose) == 0 or num(st.game_state) == 9 or num(st.game_state) == 11
        or num(st.samus_y or st.y) > 150 then
      session:step({}, "bb_elev_ride")
    elseif elev_status() == 1 then
      session:step({"DOWN"}, "bb_elev_board")
    else
      session:step({}, "bb_elev_ride")
    end
  end
  if num(session.state.room_id) ~= ROOM_MORPH then
    error("bb elev reactive missed Morph: " .. format_state(session.state))
  end
  -- Morph open-loop seed expects elev pose 0 entry (product). Do not force
  -- ordinary stand here — that desyncs the seed.
end

local function try_bb_elev_seed(session, seed, parity, reseat)
  if num(session.state.room_id) == ROOM_MORPH then
    return true
  end
  if num(session.state.room_id) ~= ROOM_BLUE_BRINSTAR_ELEVATOR then
    return false
  end
  local i
  if reseat then
    for i = 1, 80 do
      local st = session.state
      if num(st.room_id) ~= ROOM_BLUE_BRINSTAR_ELEVATOR then
        break
      end
      local x = num(st.samus_x or st.x)
      if 128 <= x and x <= 148 and math.abs(num(st.velocity_x)) <= 1 then
        break
      end
      if x < 128 then
        session:step({"RIGHT"}, "bb_elev_to_pad")
      elseif x > 148 then
        session:step({"LEFT"}, "bb_elev_to_pad")
      else
        session:step({}, "bb_elev_to_pad")
      end
    end
  end
  for i = 1, parity do
    session:step({}, "elevator_seed_parity")
  end
  span_idle(session, seeds.ELEVATOR_ALIGN_FRAMES, "elevator_seed_alignment")
  session:raw_actions(seed, "seed_bb_elev_hallway")
  local st = session.state
  local room = num(st.room_id)
  local gs = num(st.game_state)
  if room == ROOM_MORPH or (room == ROOM_BLUE_BRINSTAR_ELEVATOR and (gs == 9 or gs == 11)) then
    session:wait_until(function(s)
      return num(s.room_id) == ROOM_MORPH
    end, 180, "seed_bb_elev_hallway_transition_settle")
    return true
  end
  return false
end

-- BB elev → Morph. Prefer product seed; re-pin if elev status phase misses.
--
-- Elev standing flag $0E16 toggles each frame. Product open-loop DOWN
-- lands on a 1 frame; a faster Ceres path (TAS boot) can invert that
-- parity so the same seed crouches instead of boarding. First attempt is
-- exact product (parity 1, no pad walk). On miss, re-seat pad and try
-- parity 0 / 2, then WRAM-reactive board.
function M.play_elevator_to_morph_room(session)
  local seed = seeds.expand(seeds.bb_elev_hallway)
  -- Faster Ceres shifts $0E16 phase. Try both parities on the pad before
  -- the WRAM board; do not burn a reseat+replay when parity 0 is enough.
  if try_bb_elev_seed(session, seed, 0, false) then
    return
  end
  if try_bb_elev_seed(session, seed, 1, false) then
    return
  end
  play_bb_elev_reactive(session)
end

-- Morph Ball room seed. On miss, return to elev pad and re-seed once.
--
-- Product seed expects elev pose 0 @ x≈128. TAS boot + reactive BB elev can
-- land with matching kinematics yet still desync the open-loop tape by phase;
-- one pad return + idle-14 re-seed is the cheap residual (not thrash).
function M.play_morph_ball_collect(session)
  local seed = seeds.expand(seeds.morph_ball_room)
  seeds.play(session, "morph_ball_room", nil)
  if has_morph(session.state) then
    return
  end

  -- Walk back toward elev pad (product entry ~x128 y292 pose 0).
  local i
  for i = 1, 400 do
    local st = session.state
    if num(st.room_id) ~= ROOM_MORPH then
      break
    end
    if has_morph(session.state) then
      return
    end
    local x = num(st.samus_x or st.x)
    local pose = num(st.pose)
    if pose == 137 or pose == 138 then
      session:step({}, "morph_seed_kb")
    elseif 110 <= x and x <= 145 and math.abs(num(st.velocity_x)) <= 1 then
      if pose == 0 or math.abs(num(st.velocity_y)) <= 1 then
        break
      end
      session:step({}, "morph_return_elev")
    elseif x > 145 then
      session:step({"LEFT"}, "morph_return_elev")
    elseif x < 110 then
      session:step({"RIGHT"}, "morph_return_elev")
    else
      session:step({}, "morph_return_elev")
    end
  end

  for i = 1, 14 do
    session:step({}, "morph_seed_phase_align")
  end
  session:raw_actions(seed, "seed_morph_ball_room")
  if not has_morph(session.state) then
    error("Morph Ball was not acquired: " .. format_state(session.state))
  end
end

-- Landing Site → Morph Ball. Leave: room 0x9E9F, morph bit 0x0004.
function M.play_ship_to_morph(session)
  M.play_landing_to_parlor(session)
  M.play_parlor_to_climb(session)
  M.play_climb_to_pit(session)
  M.play_pit_to_elevator(session)
  M.play_elevator_to_morph_room(session)
  M.play_morph_ball_collect(session)
  local st = session.state
  if num(st.room_id) ~= ROOM_MORPH or not has_morph(st) then
    error("ship_to_morph leave failed: expected room 0x9E9F morph bit 0x0004, got "
      .. format_state(st))
  end
end

-- Ceres prefix + Landing→Morph. Prefix is ceres.spine / ceres.outbound.
function M.play_boot_to_morph(session)
  local ceres_spine = require("ceres.spine")
  ceres_spine.play(session)
  M.play_ship_to_morph(session)
end

M.ROOM_LANDING_SITE = ROOM_LANDING_SITE
M.ROOM_PARLOR = ROOM_PARLOR
M.ROOM_CLIMB = ROOM_CLIMB
M.ROOM_PIT = ROOM_PIT
M.ROOM_BLUE_BRINSTAR_ELEVATOR = ROOM_BLUE_BRINSTAR_ELEVATOR
M.ROOM_MORPH = ROOM_MORPH
M.MORPH_BALL_MASK = MORPH_BALL_MASK

return M
