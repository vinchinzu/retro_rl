-- Ceres outbound (elev→Ridley) and escape (Ridley→Landing) play callables.

local ram = require("ram")
local rooms = require("rooms")
local takeoff = require("takeoff")
local knockback = require("skills.knockback")
local moonfall = require("skills.moonfall")
local geom = require("ceres.geometry")
local arm = require("ceres.arm_pump")
local scientist = require("ceres.scientist")
local magnet = require("ceres.magnet")

local M = {}

local GS_ORDINARY = ram.GS_ORDINARY or 8

local ROOM_CERES_ELEVATOR = rooms.ROOM_CERES_ELEVATOR
local ROOM_CERES_FALLING = rooms.ROOM_CERES_FALLING
local ROOM_CERES_FLAT = rooms.ROOM_CERES_FLAT
local ROOM_CERES_MAGNET = rooms.ROOM_CERES_MAGNET
local ROOM_CERES_RIDLEY = rooms.ROOM_CERES_RIDLEY
local ROOM_CERES_SCIENTIST = rooms.ROOM_CERES_SCIENTIST
local ROOM_LANDING_SITE = rooms.ROOM_LANDING_SITE or 0x91F8

local CERES_FALLING_EXIT_HOP = geom.CERES_FALLING_EXIT_HOP
local CERES_FALLING_FLOOR_HOP = geom.CERES_FALLING_FLOOR_HOP
local CERES_MAGNET_MID_HOP = geom.CERES_MAGNET_MID_HOP
local CERES_MAGNET_TOP_HOP = geom.CERES_MAGNET_TOP_HOP
local _CERES_FALLING_EAST_DOOR_TRIGGER_X = geom._CERES_FALLING_EAST_DOOR_TRIGGER_X
local _CERES_FALLING_EAST_DOOR_WALK_FRAMES = geom._CERES_FALLING_EAST_DOOR_WALK_FRAMES
local _CERES_FALLING_OUT_HOP_X = geom._CERES_FALLING_OUT_HOP_X
local _CERES_FALLING_OUT_TAKEOFF_X = geom._CERES_FALLING_OUT_TAKEOFF_X
local _CERES_FALLING_OUT_UNSPIN_AIR_FRAME = geom._CERES_FALLING_OUT_UNSPIN_AIR_FRAME
local _CERES_FALLING_OUT_FLOOR_Y = geom._CERES_FALLING_OUT_FLOOR_Y
local _CERES_FALLING_OUT_PLAT_Y = geom._CERES_FALLING_OUT_PLAT_Y
local _CERES_FIRST_DOOR_FADE = geom._CERES_FIRST_DOOR_FADE
local _CERES_FIRST_DOOR_X = geom._CERES_FIRST_DOOR_X
local _CERES_FIRST_INVERT_L_X = geom._CERES_FIRST_INVERT_L_X
local _CERES_FIRST_INVERT_L_X_END = geom._CERES_FIRST_INVERT_L_X_END
local _CERES_FIRST_PAD_Y = geom._CERES_FIRST_PAD_Y
local _CERES_MAGNET_BOT_Y = geom._CERES_MAGNET_BOT_Y
local _CERES_MAGNET_DOOR_Y = geom._CERES_MAGNET_DOOR_Y
local _CERES_MAGNET_MID_Y = geom._CERES_MAGNET_MID_Y
local _CERES_MAGNET_OUT_HOP1_DOWN_Y = geom._CERES_MAGNET_OUT_HOP1_DOWN_Y
local _CERES_MAGNET_OUT_HOP1_TURN_Y = geom._CERES_MAGNET_OUT_HOP1_TURN_Y
local _CERES_MAGNET_OUT_HOP2_DOWN_Y = geom._CERES_MAGNET_OUT_HOP2_DOWN_Y
local _CERES_MAGNET_OUT_HOP2_TURN_Y = geom._CERES_MAGNET_OUT_HOP2_TURN_Y
local _CERES_MAGNET_OUT_MID_HOP_Y = geom._CERES_MAGNET_OUT_MID_HOP_Y
local _CERES_MAGNET_OUT_MID_UNSPIN_AIR_FRAME = geom._CERES_MAGNET_OUT_MID_UNSPIN_AIR_FRAME
local _CERES_MAGNET_OUT_STEAM_RECOVER = geom._CERES_MAGNET_OUT_STEAM_RECOVER
local _CERES_MAGNET_OUT_STEAM_X_HI = geom._CERES_MAGNET_OUT_STEAM_X_HI
local _CERES_MAGNET_OUT_STEAM_X_LO = geom._CERES_MAGNET_OUT_STEAM_X_LO
local _CERES_MAGNET_OUT_TOP_HOP_X = geom._CERES_MAGNET_OUT_TOP_HOP_X
local _CERES_MAGNET_TOP_Y = geom._CERES_MAGNET_TOP_Y

-- Wiki Ceres 1: moonfall. Sniq 100% lsnes (pad f8639→door f8788) short-hops
-- RIGHT off the pad, air-turns pose 25→26, lands y=75 (pose 229), then a
-- spinning moonfall whose idle weave is past 171/267. From this pin the raw
-- body runs one simulation frame ahead; a neutral lead aligns both ledges.
-- Keep the first B+RIGHT setup or snes9x lands on 171. Do not d-pad-steer a
-- RAM corridor — WJ-check lands 267/363.
-- https://wiki.supermetroid.run/KPDR_Room_Strategies
local FALLING_SETTLE = 180
-- PJBoy $0A1F: 2 jump, 3 spin, 6 fall, 23 used-item / gun-jump.
local _AIR_MT = {[2] = true, [3] = true, [6] = true, [23] = true}
local WIKI_URL = moonfall.WIKI_URL or "https://wiki.supermetroid.run/Moonwalk"

local function I(v)
  return tonumber(v) or 0
end

local function state_str(st)
  if st == nil then
    return "nil"
  end
  return string.format(
    "room=0x%04X gs=%s xy=(%s,%s) pose=%s mx=%s",
    I(st.room_id),
    tostring(st.game_state),
    tostring(st.samus_x),
    tostring(st.samus_y),
    tostring(st.pose),
    tostring(st.momentum_x)
  )
end

local function timeout(msg)
  error("TimeoutError: " .. msg, 2)
end

local function replace(track, upd)
  local n = {}
  for k, v in pairs(track) do
    n[k] = v
  end
  for k, v in pairs(upd) do
    n[k] = v
  end
  return n
end

local function step(session, names, reason)
  if names == nil or #names == 0 then
    if session.idle then
      session:idle(reason)
    else
      session:step({}, reason)
    end
  else
    session:step(names, reason)
  end
end

local function names_eq(a, b)
  if a == nil or b == nil then
    return a == b
  end
  if #a ~= #b then
    return false
  end
  for i = 1, #a do
    if a[i] ~= b[i] then
      return false
    end
  end
  return true
end

local function has_button(names, btn)
  if names == nil then
    return false
  end
  for i = 1, #names do
    if names[i] == btn then
      return true
    end
  end
  return false
end

local function hop_ready(hop, state)
  if hop == nil or type(hop.ready) ~= "function" then
    return false
  end
  return hop:ready(state)
end

local function hop_covers_y(hop, y, slack)
  if hop == nil then
    return false
  end
  if type(hop.covers_y) == "function" then
    if slack == nil then
      return hop:covers_y(y)
    end
    return hop:covers_y(y, slack)
  end
  slack = slack or 16
  return math.abs(I(y) - I(hop.y)) <= slack
end

local function hop_at_ledge_end(hop, x)
  if hop == nil then
    return false
  end
  if type(hop.at_ledge_end) == "function" then
    return hop:at_ledge_end(x)
  end
  local slack = 12
  local side = hop.side or (hop.takeoff and hop.takeoff.side)
  if side == "RIGHT" then
    return I(x) >= I(hop.x_hi or hop.xHi) - slack
  end
  return I(x) <= I(hop.x_lo or hop.xLo) + slack
end

local function is_airborne(state)
  if moonfall.is_airborne then
    return moonfall.is_airborne(state)
  end
  local mt = I(state.movement_type)
  return mt == 2 or mt == 3 or mt == 6
end

local function is_kb(state)
  if knockback.is_knockback then
    return knockback.is_knockback(state)
  end
  local p = I(state.pose)
  return p == 137 or p == 138
end

local function shoulder_pump_button(i)
  if takeoff.shoulder_pump_button then
    return takeoff.shoulder_pump_button(i)
  end
  if math.floor(I(i) / 2) % 2 == 0 then
    return "L"
  end
  return "R"
end

local function spin_jump(side)
  if takeoff.spin_jump then
    return takeoff.spin_jump(side)
  end
  return {side, "B", "A"}
end

local function clear_knockback(session, direction, reason)
  local fn = arm._ceres_clear_knockback or arm.clear_knockback
  fn(session, direction, reason)
end

local function wait_ordinary(session, room_id, reason, timeout_f)
  local fn = arm._ceres_wait_ordinary or arm.wait_ordinary
  if fn then
    fn(session, room_id, reason, timeout_f)
    return
  end
  session:wait_until(function(s)
    return I(s.room_id) == room_id and I(s.game_state) == GS_ORDINARY
  end, timeout_f, reason)
end

local function play_scientist_to_flat(session)
  local fn = scientist.play_scientist_to_flat or scientist.play_ceres_scientist_to_flat
  return fn(session)
end

local function play_scientist_to_magnet(session)
  local fn = scientist.play_scientist_to_magnet or scientist.play_ceres_scientist_to_magnet
  return fn(session)
end

local function scientist_cross(direction)
  local ctor = scientist.CeresScientistCross
  if type(ctor) == "table" and type(ctor.new) == "function" then
    return ctor.new(direction)
  end
  return ctor(direction)
end

local function expand_button_spans(spans)
  local out = {}
  for i = 1, #spans do
    local names = spans[i][1]
    local n = spans[i][2]
    for _ = 1, n do
      out[#out + 1] = names
    end
  end
  return out
end

-- Sniq 100% lsnes pad→door (movie f8639–8788). Flattened in action().
M.CERES_FIRST_TAS_PAD_SPANS = {
  {{"B", "RIGHT"}, 1},
  {{"B", "Y", "RIGHT", "A"}, 1},
  {{"B", "RIGHT"}, 10},
  {{"B", "LEFT"}, 1},
  {{"B", "L"}, 1},
  {{"B"}, 1},
  {{"B", "RIGHT", "X"}, 1},
  {{"B", "RIGHT", "A"}, 1},
  {{"B"}, 1},
  {{"B", "RIGHT"}, 7},
  {{"RIGHT"}, 7},
  {{}, 4},
  {{"LEFT"}, 1},
  {{}, 7},
  {{"LEFT"}, 15},
  {{}, 1},
  {{"RIGHT"}, 1},
  {{}, 13},
  {{"RIGHT"}, 1},
  {{}, 7},
  {{"LEFT"}, 1},
  {{}, 5},
  {{"LEFT"}, 1},
  {{}, 7},
  {{"RIGHT"}, 1},
  {{}, 4},
  {{"LEFT", "RIGHT"}, 1},
  {{}, 14},
  {{"RIGHT"}, 3},
  {{"L"}, 1},
  {{"RIGHT"}, 4},
  {{"B", "RIGHT"}, 3},
  {{"B", "LEFT", "X"}, 1},
  {{"RIGHT"}, 1},
  {{"B"}, 1},
  {{"B", "RIGHT"}, 1},
  {{"B", "RIGHT", "L"}, 12},
  {{"B", "RIGHT"}, 1},
  {{"B", "RIGHT", "L"}, 1},
  {{"B", "RIGHT"}, 1},
  {{"B", "RIGHT", "L"}, 1},
  {{"B", "RIGHT"}, 1},
  {{"B", "RIGHT", "L"}, 1},
  {{"B", "RIGHT"}, 1},
}
M.CERES_FIRST_TAS_PAD = expand_button_spans(M.CERES_FIRST_TAS_PAD_SPANS)

local function _ceres_first_tas_phase(held)
  if held < 12 then
    return "hop"
  end
  if held < 13 then
    return "air_turn"
  end
  if held < 25 then
    return "moon_arm"
  end
  return "fall"
end

local function _ceres_first_door_fade(fade_i)
  -- TAS f8789–8949: B+RIGHT on gs=9, idle, B+RIGHT+R on last gs=11.
  if fade_i <= 0 then
    return {"RIGHT", "B"}
  end
  if fade_i == _CERES_FIRST_DOOR_FADE - 1 then
    return {"RIGHT", "B", "R"}
  end
  return {}
end

local function _ceres_first_door_run(state)
  -- Last-floor L on pose 17. R aims leftover 15; B+RIGHT from p17 dumps 9.
  if I(state.samus_y) < 650 or I(state.samus_x) < _CERES_FIRST_DOOR_X then
    return nil
  end
  if I(state.pose) == 17 then
    return {"B", "RIGHT", "L"}
  end
  return nil
end

local function _ceres_first_floor_pad(state, names, track, held)
  -- Retimed snes9x floor suffix after the intact TAS moonfall prefix.
  local y = I(state.samus_y)
  local nxt = held + 1
  -- The raw bottom-contact turn backs snes9x up by two pixels. Replacing it
  -- with the run chord preserves the following RIGHT edge and recovers the
  -- neutral lead frame without moving the next-room seat far from product.
  if y >= 650 and I(state.pose) == 164 and names_eq(names, {"B", "LEFT", "X"}) then
    names = {"B", "RIGHT"}
  end
  -- One early B release in the run-up restores the established product door
  -- subpixel after the faster bottom turn; the later reactive rooms then see
  -- their previously proven entry timing without any controller changes.
  if y >= 650 and held == 133 and names_eq(names, {"B", "RIGHT", "L"}) then
    names = {"RIGHT", "L"}
  end
  if (
    not track.invert_l
    and y >= 650
    and I(state.pose) == 17
    and _CERES_FIRST_INVERT_L_X <= I(state.samus_x)
    and I(state.samus_x) < _CERES_FIRST_INVERT_L_X_END
    and has_button(names, "L")
    and nxt < #M.CERES_FIRST_TAS_PAD
  ) then
    names = M.CERES_FIRST_TAS_PAD[nxt + 1]
    track = replace(track, {invert_l = true})
    nxt = nxt + 1
  end
  local lip = _ceres_first_door_run(state)
  if lip ~= nil then
    return lip, track, nxt
  end
  return names, track, nxt
end

local function with_action(track, fn)
  function track:action(state)
    return fn(state, self)
  end
  return track
end

function M.CeresFirstMoonfallTrack(fields)
  fields = fields or {}
  return with_action({
    phase = fields.phase or "ride",
    held = fields.held or 0,
    pump_i = fields.pump_i or 0,
    invert_l = fields.invert_l or false,
  }, function(state, track)
    return M.ceres_first_moonfall_action(state, track)
  end)
end

local function _ceres_first_airborne(state)
  if is_airborne(state) then
    return true
  end
  if _AIR_MT[I(state.movement_type)] then
    return true
  end
  local vd = I(state.vertical_direction)
  if vd == 1 or vd == 2 then
    return true
  end
  return false
end

function M.ceres_first_moonfall_action(state, track)
  -- One-frame Ceres elevator → Falling moonfall policy (ROM-free).
  local x = I(state.samus_x)
  local y = I(state.samus_y)
  local room = I(state.room_id)
  local phase = track.phase
  local held = track.held
  local grounded = not _ceres_first_airborne(state)

  if room == ROOM_CERES_FALLING and I(state.game_state) == 8 then
    return {}, replace(track, {phase = "done", held = 0})
  end
  -- TAS: B+RIGHT on gs=9, idle the fade, B+RIGHT+R on last gs=11 (f8949).
  local gs = I(state.game_state)
  if gs == 9 or gs == 10 or gs == 11 then
    local fade_i = (phase == "exit") and held or 0
    return _ceres_first_door_fade(fade_i), replace(track, {phase = "exit", held = fade_i + 1})
  end
  if room ~= ROOM_CERES_ELEVATOR and phase ~= "exit" then
    return {"RIGHT", "B"}, replace(track, {phase = "exit", held = 0})
  end

  if is_kb(state) and phase ~= "exit" and phase ~= "done" and phase ~= "land" then
    local away = (x >= 190) and "LEFT" or "RIGHT"
    local held_kb = held
    if held_kb < 25 then
      held_kb = 25
    end
    return {away}, replace(track, {phase = "fall", held = held_kb})
  end

  if phase == "ride" then
    if y >= _CERES_FIRST_PAD_Y or is_airborne(state) then
      -- The pin is the visible post-f8639 seat, but snes9x still needs
      -- that B+RIGHT setup chord to make the moonfall pass the ledges.
      -- Lead it with one neutral frame so the pane does not run a
      -- simulation frame ahead of the authoring core.
      return {}, replace(track, {phase = "hop", held = 0})
    end
    return {}, track
  end

  if phase == "hop" or phase == "air_turn" or phase == "moon_arm" or phase == "fall" then
    if room ~= ROOM_CERES_ELEVATOR then
      return {"RIGHT", "B"}, replace(track, {phase = "exit", held = 0})
    end
    if held < #M.CERES_FIRST_TAS_PAD then
      local names = M.CERES_FIRST_TAS_PAD[held + 1]
      local nxt
      names, track, nxt = _ceres_first_floor_pad(state, names, track, held)
      return names, replace(track, {phase = _ceres_first_tas_phase(held), held = nxt})
    end
    if grounded and y >= 650 and x >= 175 then
      return {"RIGHT", "B"}, replace(track, {phase = "land", held = 0, pump_i = 0})
    end
    if y >= 640 then
      return {"RIGHT", "B"}, replace(track, {phase = "land", held = 0, pump_i = 0})
    end
    return {"RIGHT", "B"}, replace(track, {phase = "fall", held = held + 1})
  end

  if phase == "land" then
    if room ~= ROOM_CERES_ELEVATOR then
      return {"RIGHT", "B"}, replace(track, {phase = "exit", held = 0})
    end
    if is_kb(state) and held < 16 then
      return {}, replace(track, {held = held + 1})
    end
    if y > 655 then
      return {"RIGHT", "A"}, replace(track, {held = held + 1})
    end
    local pump = shoulder_pump_button(track.pump_i)
    return {"RIGHT", "B", pump}, replace(track, {held = held + 1, pump_i = track.pump_i + 1})
  end

  if phase == "exit" then
    if room == ROOM_CERES_FALLING and I(state.game_state) == 8 then
      return {}, replace(track, {phase = "done", held = 0})
    end
    return {"RIGHT", "B"}, replace(track, {held = held + 1})
  end

  return {}, track
end

function M.play_first_room_moonfall(session, max_frames, restore_moonwalk)
  -- TAS-shaped Ceres 1 moonfall. Pokes $09E4 on, off after Falling settle.
  -- File option, not a resource write.
  max_frames = max_frames or 900
  if restore_moonwalk == nil then
    restore_moonwalk = true
  end
  ram.set_moonwalk(true)
  if session.refresh then
    session:refresh()
  end
  if session.state then
    session.state.moonwalk = 1
  end
  moonfall.require_moonwalk_on(session.state, "ceres_first_moonfall")
  local gs0 = I(session.state.game_state)
  if I(session.state.room_id) ~= ROOM_CERES_ELEVATOR and gs0 ~= 8 and gs0 ~= 11 then
    error(string.format(
      "ceres first moonfall: expected elev 0x%04X, got %s",
      ROOM_CERES_ELEVATOR,
      state_str(session.state)
    ), 2)
  end

  local track = M.CeresFirstMoonfallTrack()
  local done = false
  for _ = 1, max_frames do
    local names
    names, track = M.ceres_first_moonfall_action(session.state, track)
    step(session, names, "ceres_first_" .. track.phase)
    if track.phase == "done" or (
      I(session.state.room_id) == ROOM_CERES_FALLING
      and I(session.state.game_state) == 8
    ) then
      done = true
      break
    end
  end
  if not done then
    timeout(string.format(
      "ceres first moonfall missed Falling after %df: %s phase=%s (%s)",
      max_frames,
      state_str(session.state),
      tostring(track.phase),
      WIKI_URL
    ))
  end

  if I(session.state.room_id) ~= ROOM_CERES_FALLING or I(session.state.game_state) == 11 then
    wait_ordinary(session, ROOM_CERES_FALLING, "ceres_first_falling_settle", FALLING_SETTLE)
  end
  if restore_moonwalk then
    ram.set_moonwalk(false)
    if session.refresh then
      session:refresh()
    end
    if session.state then
      session.state.moonwalk = 0
    end
  end
end

function M.CeresFallingTrack(fields)
  -- TAS Ceres 2: arm-pump the slope, magnet-feet onto y=171, fly the exit.
  --
  -- Sniq 100% lsnes (gs=8 f8950→door f9071): B+RIGHT with L every other
  -- frame down the y=139 slope, then a 3f short hop at x≈162 y=187
  -- (B+RIGHT+A, B+A, B+LEFT+RIGHT+A) so magnet feet plant y=171. A 3f
  -- spin jump hangs at y=178 and misses the shelf; Sniq 8977's `B+RIGHT+X`
  -- on the 4th air frame unspins it onto the lip instead. Run the shelf, 4f
  -- B+RIGHT+A + 2f A, idle-spin so y stays ~148 into the east door.
  --
  -- `door_gate` counts frames since the east door's proximity trigger and
  -- is what keeps the approach off the closed shutter — see the
  -- `_CERES_FALLING_EAST_DOOR_*` block in geometry. -1 means not yet armed.
  fields = fields or {}
  local door_gate = -1
  if fields.door_gate ~= nil then
    door_gate = fields.door_gate
  end
  return with_action({
    phase = fields.phase or "ledge",
    pump_i = fields.pump_i or 0,
    floor_hopped = fields.floor_hopped or false,
    exit_hopped = fields.exit_hopped or false,
    hop_held = fields.hop_held or 0,
    steam_shot = fields.steam_shot or false,
    door_gate = door_gate,
  }, function(state, track)
    return M.ceres_falling_magnet_feet_action(state, track)
  end)
end

local function _ceres_falling_air(state)
  if is_airborne(state) then
    return true
  end
  if _AIR_MT[I(state.movement_type)] then
    return true
  end
  local vd = I(state.vertical_direction)
  return vd == 1 or vd == 2
end

local function _ceres_pump(direction, i)
  return {direction, "B", shoulder_pump_button(i)}
end

local function _ceres_shelf_pump(direction, i)
  -- TAS Ceres 2 y=171 shelf: B+dir, L on odd frames. Not the entry slope.
  if i % 2 ~= 0 then
    return {direction, "B", "L"}
  end
  return {direction, "B"}
end

local function _ceres_magnet_feet_jump(held)
  -- Wiki Ceres 2 short hop + L/R magnet feet onto y=171.
  --
  -- Sniq 100% lsnes f8973–8975: 1f B+RIGHT+A, 1f B+A, 1f B+LEFT+RIGHT+A.
  -- Spin-jump (dir+B+A held) never plants the shelf. Do not X during the
  -- hop — that unspins into the y=171 face.
  if held <= 1 then
    return {"RIGHT", "B", "A"}
  end
  if held == 2 then
    return {"B", "A"}
  end
  return {"LEFT", "RIGHT", "B", "A"}
end

local function _ceres_falling_arm_door_gate(track, x)
  -- Tick the east-door shutter clock; arm it on the proximity trigger.
  if track.door_gate >= 0 then
    return replace(track, {door_gate = track.door_gate + 1})
  end
  if x >= _CERES_FALLING_EAST_DOOR_TRIGGER_X then
    return replace(track, {door_gate = 0})
  end
  return track
end

function M.ceres_falling_magnet_feet_action(state, track)
  -- One-frame Falling Tile → Magnet policy.
  local room = I(state.room_id)
  local gs = I(state.game_state)
  local x = I(state.samus_x)
  local y = I(state.samus_y)
  local air = _ceres_falling_air(state)

  if room == ROOM_CERES_MAGNET and gs == 8 then
    return {}, replace(track, {phase = "done"})
  end
  if gs ~= GS_ORDINARY then
    return {}, replace(track, {phase = "door"})
  end
  if room ~= ROOM_CERES_FALLING then
    return {"RIGHT", "B"}, replace(track, {phase = "exit"})
  end

  local floor = CERES_FALLING_FLOOR_HOP
  local exit_hop = CERES_FALLING_EXIT_HOP

  -- The east door is a proximity shutter: crossing its trigger x starts a
  -- fixed open cycle that no input shortens. Arm the counter the frame the
  -- trigger is crossed — in the air on the exit hop, normally — so the
  -- approach below can spend the cycle instead of eating it on the face.
  track = _ceres_falling_arm_door_gate(track, x)

  -- Entry ledge is also y=139 ≤ 176. Only the y=187 floor (x≳148) is the
  -- magnet-feet hop. An air+floor_hopped jump on the slope burns run speed.
  local on_floor = y >= _CERES_FALLING_OUT_FLOOR_Y - 2
  local on_shelf = (not air) and y <= _CERES_FALLING_OUT_PLAT_Y + 2 and x >= 148

  -- Walk, do not dash, until the shutter has had its cycle. Dashing here
  -- arrives on a closed face at full speed and pays the pose-137 stall plus
  -- a whole dash rebuild; walking first arrives on an open one still dashing.
  if (
    (not air)
    and 0 <= track.door_gate
    and track.door_gate < _CERES_FALLING_EAST_DOOR_WALK_FRAMES
    and x >= _CERES_FALLING_EAST_DOOR_TRIGGER_X - 8
  ) then
    return {"RIGHT"}, replace(track, {phase = "gate"})
  end

  if air and track.hop_held >= 1 and not track.exit_hopped and not on_shelf then
    if track.hop_held < 3 then
      local nxt = track.hop_held + 1
      return _ceres_magnet_feet_jump(nxt), replace(track, {phase = "floor_hop", hop_held = nxt})
    end
    if y <= _CERES_FALLING_OUT_PLAT_Y + 2 then
      return _ceres_shelf_pump("RIGHT", track.pump_i), replace(track, {
        phase = "plat",
        hop_held = 0,
        pump_i = track.pump_i + 1,
      })
    end
    if on_floor and track.hop_held >= 6 then
      return _ceres_pump("RIGHT", track.pump_i), replace(track, {
        phase = "ledge",
        floor_hopped = false,
        hop_held = 0,
        pump_i = track.pump_i + 1,
      })
    end
    local names = {"RIGHT", "B"}
    if track.hop_held == _CERES_FALLING_OUT_UNSPIN_AIR_FRAME then
      names = {"RIGHT", "B", "X"}
    end
    return names, replace(track, {phase = "magnet_feet", hop_held = track.hop_held + 1})
  end

  if (not air) and on_floor then
    if (
      x <= 168
      and I(state.momentum_x) >= 1
      and (
        hop_ready(floor, state)
        or hop_at_ledge_end(floor, x)
        or x >= _CERES_FALLING_OUT_HOP_X
      )
    ) then
      return _ceres_magnet_feet_jump(1), replace(track, {
        phase = "floor_hop",
        floor_hopped = true,
        hop_held = 1,
      })
    end
    return _ceres_pump("RIGHT", track.pump_i), replace(track, {
      phase = "ledge",
      hop_held = 0,
      pump_i = track.pump_i + 1,
    })
  end

  if air and track.exit_hopped then
    -- TAS: 4f B+RIGHT+A, 2f A, ~19f idle-spin, then B+LEFT+X and run.
    -- Idling until y>175 lands the door lip (pose 156 stall at x=462).
    if track.hop_held < 4 then
      return spin_jump("RIGHT"), replace(track, {phase = "exit_hop", hop_held = track.hop_held + 1})
    end
    if track.hop_held < 6 then
      return {"A"}, replace(track, {phase = "exit_hop", hop_held = track.hop_held + 1})
    end
    if track.hop_held < 25 and y <= 170 then
      return {}, replace(track, {phase = "exit_hop", hop_held = track.hop_held + 1})
    end
    return {"RIGHT", "B"}, replace(track, {phase = "exit"})
  end

  if (
    (not air)
    and hop_covers_y(exit_hop, y, 12)
    and (
      hop_ready(exit_hop, state)
      or hop_at_ledge_end(exit_hop, x)
      or x >= _CERES_FALLING_OUT_TAKEOFF_X
    )
    and not track.exit_hopped
  ) then
    return spin_jump("RIGHT"), replace(track, {phase = "exit_hop", exit_hopped = true, hop_held = 1})
  end

  if x >= 450 and y <= 150 and not track.steam_shot then
    return {"RIGHT", "B", "X"}, replace(track, {phase = "exit", steam_shot = true})
  end

  if x >= 430 then
    return {"RIGHT", "B"}, replace(track, {phase = "exit"})
  end

  if air then
    return {"RIGHT", "B"}, replace(track, {
      phase = track.exit_hopped and "exit_hop" or "ledge",
    })
  end

  local pump_names
  if on_shelf then
    pump_names = _ceres_shelf_pump("RIGHT", track.pump_i)
  else
    pump_names = _ceres_pump("RIGHT", track.pump_i)
  end
  return pump_names, replace(track, {
    phase = on_shelf and "plat" or "ledge",
    pump_i = track.pump_i + 1,
    floor_hopped = track.floor_hopped or on_shelf,
    hop_held = on_shelf and 0 or track.hop_held,
  })
end

local function _play_track(session, src, dest, track, action, door_kb, tag, missed, max_frames)
  if I(session.state.room_id) == dest and I(session.state.game_state) == 8 then
    return
  end
  if I(session.state.room_id) ~= src or I(session.state.game_state) ~= 8 then
    wait_ordinary(session, src, tag .. "_ordinary", FALLING_SETTLE)
  end
  local broke = false
  for _ = 1, max_frames do
    local st = session.state
    if I(st.room_id) == dest and I(st.game_state) == 8 then
      return
    end
    if is_kb(st) then
      if door_kb(st) then
        session:step({"RIGHT"}, tag .. "_door_kb")
      else
        clear_knockback(session, "RIGHT", tag .. "_out")
      end
    else
      local names
      names, track = action(st, track)
      step(session, names, tag .. "_" .. track.phase)
      if track.phase == "done" then
        broke = true
        break
      end
    end
  end
  if not broke then
    timeout(string.format(
      "%s after %df: %s phase=%s",
      missed,
      max_frames,
      state_str(session.state),
      tostring(track.phase)
    ))
  end
  if I(session.state.room_id) ~= dest or I(session.state.game_state) == 11 then
    wait_ordinary(session, dest, tag .. "_settle", FALLING_SETTLE)
  end
end

function M.play_falling_to_magnet(session, max_frames)
  -- Wiki Ceres 2 magnet-feet hop. Waits dest gs=8.
  max_frames = max_frames or 700
  _play_track(
    session,
    ROOM_CERES_FALLING,
    ROOM_CERES_MAGNET,
    M.CeresFallingTrack(),
    M.ceres_falling_magnet_feet_action,
    function(s)
      return I(s.samus_x) >= 400
    end,
    "ceres_falling",
    "ceres falling magnet-feet missed Magnet",
    max_frames
  )
end

function M.CeresMagnetTrack(fields)
  -- Wiki Ceres 3: jump before each magnet-stair ledge.
  --
  -- Sniq 100% lsnes (gs=8 f9232→door f9415): B+RIGHT with L every other
  -- frame off the west door, jump the y=139 ledge at x≈134, idle-spin past
  -- the lip, DOWN then 1f RIGHT then LEFT onto y=219 (~x192). L-pump LEFT
  -- down the slope, jump at y≈262 x≈124, X-unspin air-frame 2, A-hold,
  -- DOWN+A, 1f LEFT then RIGHT onto y=347. L-pump RIGHT to the east door.
  -- Steam at x~160 is 1f RIGHT (no B) then LEFT+B+X. Sniq's L-then-pose-74
  -- gun is one frame on bsnes; on this leftover LEFT+B+X is pose 37
  -- either way. Do not L-pump the turnaround (moonwalk). Do not L/R on
  -- the stairs (pose 41).
  fields = fields or {}
  return with_action({
    phase = fields.phase or "top",
    hop_held = fields.hop_held or 0,
    pump_i = fields.pump_i or 0,
    steam_shot = fields.steam_shot or false,
    steam_held = fields.steam_held or 0,
    air_turn = fields.air_turn or false,
  }, function(state, track)
    return M.ceres_magnet_to_scientist_action(state, track)
  end)
end

function M.ceres_magnet_to_scientist_action(state, track)
  -- One-frame Magnet Stairs → Scientist policy (ROM-free).
  local room = I(state.room_id)
  local gs = I(state.game_state)
  local x = I(state.samus_x)
  local y = I(state.samus_y)
  local air = _ceres_falling_air(state)

  if room == ROOM_CERES_SCIENTIST and gs == 8 then
    return {}, replace(track, {phase = "done"})
  end
  if gs ~= GS_ORDINARY then
    return {}, replace(track, {phase = "exit", hop_held = track.hop_held + 1})
  end
  if room ~= ROOM_CERES_MAGNET then
    return {"RIGHT", "B"}, replace(track, {phase = "exit"})
  end

  local top = CERES_MAGNET_TOP_HOP
  local mid = CERES_MAGNET_MID_HOP

  if (not air) and y >= _CERES_MAGNET_DOOR_Y - 8 then
    local in_steam = _CERES_MAGNET_OUT_STEAM_X_LO <= x and x <= _CERES_MAGNET_OUT_STEAM_X_HI
    if in_steam and not track.steam_shot and track.steam_held == 0 then
      -- Kill dash one frame so LEFT+B+X meets the jet at x≈170.
      -- L-only then LEFT+B+X is pose 141 moonwalk on this leftover.
      return {"RIGHT"}, replace(track, {phase = "exit", steam_held = 1})
    end
    if track.steam_held == 1 and not track.steam_shot then
      return {"LEFT", "B", "X"}, replace(track, {phase = "exit", steam_shot = true, steam_held = 2})
    end
    if track.steam_shot and track.steam_held < _CERES_MAGNET_OUT_STEAM_RECOVER then
      -- Pose 37 faces left. RIGHT+B+L here is moonwalk.
      return {"RIGHT", "B"}, replace(track, {phase = "exit", steam_held = track.steam_held + 1})
    end
    local names = _ceres_shelf_pump("RIGHT", track.pump_i)
    return names, replace(track, {phase = "exit", pump_i = track.pump_i + 1, hop_held = 0})
  end

  if (
    (not air)
    and y <= _CERES_MAGNET_TOP_Y + 16
    and (track.phase == "top" or track.phase == "jump1" or track.phase == "drop1")
  ) then
    if hop_ready(top, state) or hop_at_ledge_end(top, x) or x >= _CERES_MAGNET_OUT_TOP_HOP_X then
      return spin_jump("RIGHT"), replace(track, {phase = "jump1", hop_held = 1})
    end
    local names = _ceres_shelf_pump("RIGHT", track.pump_i)
    return names, replace(track, {
      phase = "top",
      hop_held = track.hop_held + 1,
      pump_i = track.pump_i + 1,
    })
  end

  if track.phase == "jump1" then
    if (not air) and y >= _CERES_MAGNET_MID_Y - 8 then
      local names = _ceres_shelf_pump("LEFT", 0)
      return names, replace(track, {phase = "mid", hop_held = 0, pump_i = 1, air_turn = false})
    end
    if (not air) and y <= _CERES_MAGNET_TOP_Y + 16 then
      return {"RIGHT", "B"}, replace(track, {phase = "top", hop_held = 0})
    end
    local held = track.hop_held + 1
    -- 2f spin then idle. Empty input keeps pose 25 so the 139 lip is
    -- fallen past; DOWN on the rise plants it (pose 41).
    if track.hop_held < 2 then
      return spin_jump("RIGHT"), replace(track, {hop_held = held})
    end
    if y < _CERES_MAGNET_OUT_HOP1_DOWN_Y then
      return {}, replace(track, {hop_held = held})
    end
    return {"DOWN"}, replace(track, {phase = "drop1", hop_held = held})
  end

  if track.phase == "drop1" then
    if (not air) and y >= _CERES_MAGNET_MID_Y - 8 then
      local names = _ceres_shelf_pump("LEFT", 0)
      return names, replace(track, {phase = "mid", hop_held = 0, pump_i = 1, air_turn = false})
    end
    if (not air) and y <= _CERES_MAGNET_TOP_Y + 16 then
      return {"RIGHT", "B"}, replace(track, {phase = "top", hop_held = 0})
    end
    local held = track.hop_held + 1
    if y < _CERES_MAGNET_OUT_HOP1_TURN_Y then
      return {"DOWN"}, replace(track, {hop_held = held})
    end
    if not track.air_turn then
      return {"RIGHT"}, replace(track, {hop_held = held, air_turn = true})
    end
    return {"LEFT"}, replace(track, {hop_held = held})
  end

  if (
    track.phase == "mid"
    or track.phase == "jump2"
    or track.phase == "drop2"
    or ((not air) and _CERES_MAGNET_MID_Y - 8 <= y and y < _CERES_MAGNET_BOT_Y - 8)
  ) then
    if (not air) and y >= _CERES_MAGNET_BOT_Y - 8 then
      local names = _ceres_shelf_pump("RIGHT", 0)
      return names, replace(track, {phase = "bot", hop_held = 0, pump_i = 1, air_turn = false})
    end
    if air and track.phase == "jump2" then
      local held = track.hop_held + 1
      if track.hop_held == _CERES_MAGNET_OUT_MID_UNSPIN_AIR_FRAME then
        return {"LEFT", "B", "A", "X"}, replace(track, {hop_held = held})
      end
      if track.hop_held < 3 then
        return spin_jump("LEFT"), replace(track, {hop_held = held})
      end
      -- TAS: 1f LEFT (release A) then A, not an immediate DOWN.
      return {"LEFT"}, replace(track, {phase = "drop2", hop_held = held})
    end
    if air and track.phase == "drop2" then
      local held = track.hop_held + 1
      if y >= _CERES_MAGNET_OUT_HOP2_TURN_Y and track.air_turn then
        return {"RIGHT"}, replace(track, {hop_held = held})
      end
      if y >= _CERES_MAGNET_OUT_HOP2_TURN_Y then
        return {"LEFT"}, replace(track, {hop_held = held, air_turn = true})
      end
      if y >= _CERES_MAGNET_OUT_HOP2_DOWN_Y then
        return {"DOWN", "A"}, replace(track, {hop_held = held})
      end
      return {"A"}, replace(track, {hop_held = held})
    end
    if (
      (not air)
      and (track.phase == "mid" or track.phase == "drop1")
      and y >= _CERES_MAGNET_OUT_MID_HOP_Y
      and y <= 280
      and x <= 128
      and (hop_ready(mid, state) or hop_at_ledge_end(mid, x) or x <= 124)
    ) then
      return spin_jump("LEFT"), replace(track, {phase = "jump2", hop_held = 1, air_turn = false})
    end
    if not air then
      local names = _ceres_shelf_pump("LEFT", track.pump_i)
      return names, replace(track, {phase = "mid", hop_held = 0, pump_i = track.pump_i + 1})
    end
    return {"LEFT"}, replace(track, {hop_held = track.hop_held + 1})
  end

  if y >= _CERES_MAGNET_BOT_Y - 16 or track.phase == "bot" or track.phase == "exit" then
    local names = _ceres_shelf_pump("RIGHT", track.pump_i)
    return names, replace(track, {phase = "bot", pump_i = track.pump_i + 1})
  end

  if air then
    return {"RIGHT", "B"}, track
  end
  local names = _ceres_shelf_pump("RIGHT", track.pump_i)
  return names, replace(track, {phase = "top", pump_i = track.pump_i + 1})
end

function M.play_magnet_to_scientist(session, max_frames)
  -- Wiki Ceres 3 jump-before-ledge. Waits dest gs=8.
  max_frames = max_frames or 700
  _play_track(
    session,
    ROOM_CERES_MAGNET,
    ROOM_CERES_SCIENTIST,
    M.CeresMagnetTrack(),
    M.ceres_magnet_to_scientist_action,
    function(s)
      return I(s.samus_y) >= _CERES_MAGNET_DOOR_Y - 20
    end,
    "ceres_magnet",
    "ceres magnet stairs missed Scientist",
    max_frames
  )
end

function M.CeresFlatEscape()
  -- One-frame reverse Ceres 5 (Flat → Scientist). Never jump.
  --
  -- Sniq 100% lsnes (gs=8 f11939→door f12036) never presses A: LEFT+B+L/R
  -- across y=139 from (472,139) p18. stuck-jump on this corridor is leftover.
  local self = {pump_i = 0}
  function self:action(state)
    if I(state.game_state) ~= GS_ORDINARY then
      return {"LEFT"}
    end
    local names = {"LEFT", "B", shoulder_pump_button(self.pump_i)}
    self.pump_i = self.pump_i + 1
    return names
  end
  return self
end

local function _flat_escape_past(state)
  -- True in Scientist/Magnet ordinary — not the Flat→Scientist door.
  if I(state.game_state) ~= GS_ORDINARY then
    return false
  end
  local rid = I(state.room_id)
  return rid == ROOM_CERES_SCIENTIST or rid == ROOM_CERES_MAGNET
end

local function _flat_outbound_past(state)
  if I(state.game_state) ~= GS_ORDINARY then
    return false
  end
  return I(state.room_id) == ROOM_CERES_RIDLEY
end

function M.play_flat_to_ridley(session)
  -- Flat ordinary → Ridley. TAS dwell never presses A.
  if _flat_outbound_past(session.state) then
    return
  end
  local reached = false
  for _ = 1, 180 do
    local st = session.state
    if _flat_outbound_past(st) then
      return
    end
    if I(st.room_id) == ROOM_CERES_FLAT and I(st.game_state) == GS_ORDINARY then
      reached = true
      break
    end
    session:step({"RIGHT"}, "ceres_flat_out_door")
  end
  if not reached then
    local st = session.state
    if I(st.room_id) ~= ROOM_CERES_FLAT then
      timeout("ceres flat outbound ordinary missed: " .. state_str(st))
    end
  end

  local cross = scientist_cross("RIGHT")
  for _ = 1, 400 do
    local st = session.state
    if _flat_outbound_past(st) then
      return
    end
    if is_kb(st) then
      clear_knockback(session, "RIGHT", "ceres_flat_out")
    else
      local names = cross:action(st)
      local reason
      if I(st.game_state) ~= GS_ORDINARY then
        reason = "ceres_flat_out_fade"
      else
        reason = "ceres_flat_out"
      end
      step(session, names, reason)
    end
  end
  timeout("ceres flat missed Ridley: " .. state_str(session.state))
end

function M.play_flat_to_scientist(session)
  -- Flat ordinary → Scientist (or Magnet if the door overshoots).
  --
  -- No-op when already past the room. Waits out the Ridley→Flat door.
  -- TAS dwell never jumps; product stuck-jump is the leftover versus 259f.
  if _flat_escape_past(session.state) then
    return
  end
  local reached = false
  for _ = 1, 220 do
    local st = session.state
    if _flat_escape_past(st) then
      return
    end
    if I(st.room_id) == ROOM_CERES_FLAT and I(st.game_state) == GS_ORDINARY then
      reached = true
      break
    end
    session:step({"LEFT"}, "ceres_flat_door")
  end
  if not reached then
    local st = session.state
    if I(st.room_id) ~= ROOM_CERES_FLAT then
      timeout("ceres flat ordinary missed: " .. state_str(st))
    end
  end

  local cross = M.CeresFlatEscape()
  for _ = 1, 400 do
    local st = session.state
    if _flat_escape_past(st) then
      return
    end
    if is_kb(st) then
      clear_knockback(session, "LEFT", "ceres_flat")
    else
      local names = cross:action(st)
      local reason
      if I(st.game_state) ~= GS_ORDINARY then
        reason = "ceres_flat_fade"
      else
        reason = "ceres_flat"
      end
      step(session, names, reason)
    end
  end
  timeout("ceres flat missed Scientist: " .. state_str(session.state))
end

function M.play_to_ridley_door(session)
  -- Ceres elevator → Ridley room ordinary settle (no fight).
  M.play_first_room_moonfall(session)
  M.play_falling_to_magnet(session)
  M.play_magnet_to_scientist(session)
  -- Dead Scientist is its own hop: TAS-style arm-pump run, never jump.
  play_scientist_to_flat(session)
  M.play_flat_to_ridley(session)
  wait_ordinary(session, ROOM_CERES_RIDLEY, "ceres_ridley_door", 200)
end

function M.play_outbound_to_ridley(session)
  -- Ceres elevator → Ridley + countdown (classic L↔R arm-pump).
  --
  -- Elev→Falling uses spinning moonfall (wiki Ceres 1 / Sniq pad body)
  -- then wiki Ceres 2 magnet-feet and Ceres 3 jump-before-ledge. Dead Scientist
  -- arm-pumps the pit and stairs (no jump);
  -- Flat→Ridley is room-gated arm-pump. Fight body is tail-tank
  -- play_fight. Escape re-solves magnet / falling / elev
  -- reactively.
  M.play_to_ridley_door(session)
  -- Late import: combat.__init__ → progression → early_spine → this module.
  local ridley = require("ceres.ridley")
  local fight = ridley.play_fight or ridley.play_ceres_ridley_fight
  local evidence = fight(session)
  local req = ridley.require_countdown or ridley.require_ceres_ridley_countdown
  req(evidence)
  -- Tail-tank often ends in KB (pose 137/138). Escape LEFT+A needs standing.
  for _ = 1, 40 do
    if not is_kb(session.state) then
      break
    end
    step(session, {}, "ceres_ridley_settle")
  end
end

function M.play_escape_to_landing(session)
  -- Ceres reverse + elev → Zebes Landing (arm-pump + WRAM-reactive).
  --
  -- Magnet escape is TAS 347→267→steam→219→139.
  -- Falling / elev still re-solve from room, y, pose, knockback.
  -- Ridley exit is still the product LEFT+A (not tuned). Reverse Ceres 5
  -- (Flat) and reverse Ceres 4 (Scientist) never jump.
  -- Leave Ridley left (jump clear of platform). Not tuned this sitting.
  session:span({"LEFT", "A"}, 24, "ceres_ridley_exit")
  M.play_flat_to_scientist(session)
  play_scientist_to_magnet(session)
  magnet.play_magnet_to_falling(session)
  magnet.play_falling_to_elev(session)
  magnet.reactive_elev_climb(session)

  session:wait_until(function(state)
    return I(state.room_id) == ROOM_LANDING_SITE and I(state.game_state) == 8
  end, 3000, "zebes_landing_transition")
  local stable = 0
  local settled = false
  for _ = 1, 1200 do
    if I(session.state.samus_y) == 1088 then
      stable = stable + 1
      if stable >= 30 then
        settled = true
        break
      end
    else
      stable = 0
    end
    step(session, {}, "zebes_ship_final_settle")
  end
  if not settled then
    timeout("Zebes ship never reached final settle: " .. state_str(session.state))
  end
  local info = session.info or {}
  local start = info.ceres_elev_start
  if start ~= nil then
    local used = I(session.frame) - I(start)
    if used > magnet.CERES_ELEV_MAX_FRAMES then
      timeout(string.format(
        "ceres elev_to_landing %df exceeded %df",
        used,
        magnet.CERES_ELEV_MAX_FRAMES
      ))
    end
  end
end

M.play_ceres_first_room_moonfall = M.play_first_room_moonfall
M.play_ceres_falling_to_magnet = M.play_falling_to_magnet
M.play_ceres_magnet_to_scientist = M.play_magnet_to_scientist
M.play_ceres_flat_to_ridley = M.play_flat_to_ridley
M.play_ceres_flat_to_scientist = M.play_flat_to_scientist
M.play_ceres_to_ridley_door = M.play_to_ridley_door
M.play_ceres_outbound_to_ridley = M.play_outbound_to_ridley
M.play_ceres_escape_to_landing = M.play_escape_to_landing

return M
