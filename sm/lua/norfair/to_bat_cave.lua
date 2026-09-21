-- Bubble Mountain climb: mid-left entry → top-right Super door → Bat Cave.
-- Lua 5.1. Session: step / hold / wait_until / span.

local iceg = require("ice.geometry")
local geom = require("skills.geometry")
local door = require("skills.door")
local mid = require("norfair._to_bat_cave_mid")

local ROOM_BUBBLE, ROOM_BAT = 0xACB3, 0xB07A

local P = {
  ROOM_ID = ROOM_BUBBLE,
  EXIT_ROOM_ID = ROOM_BAT,
  LOWER_FRAMES = 3500,
  MID_REPIN_FRAMES = 900,
  MID_FRAMES = 5500,
  DOOR_FRAMES = 1200,
  TO_BAT_SETTLE_FRAMES = 320,
  DOOR_SUPER_X = 420,
  DOOR_SUPER_Y = 160,
  DOOR_WJ_PERIOD = 10,
  DOOR_WJ_INTO = 3,
  DOOR_WJ_BOUNCE = 2,
  DOOR_X_CAP = 480,
  DOOR_OUTER_X = 400,
  DOOR_CROUCH_FRAMES = 0,
  MID_Y = 400,
  TOP_Y = 200,
  TOP_X = 300,
  TRUE_GROUND = {[1] = true, [2] = true, [9] = true, [10] = true},
  STAND_PIN = {[1] = true, [2] = true, [9] = true, [10] = true, [25] = true, [26] = true, [27] = true, [28] = true},
  STANDING_POSES = geom.STANDING_POSES or iceg.STANDING_POSES,
  CAVITY_X_MAX = 395,
  MID_STAND_X = {77, 160},
  FLOOR_SHELF_X = 108,
  LOWER_SHELVES = {{120, 560}, {110, 515}, {100, 475}, {90, 450}, {70, 420}, {50, 395}},
  LIP_X = {65, 100},
  LIP_Y = {410, 450},
  HEIGHT_CLASS_Y = 280,
  MID_RESEAT_Y = 320,
  RIGHT_SHELF_X = 300,
  RIGHT_SHELF_Y = 390,
  RIGHT_WJ_PERIOD = 8,
  RIGHT_WJ_INTO = 2,
  RIGHT_WJ_BOUNCE = 2,
  MIDHIGH_Y = 450,
  LIP_CHARGE = 12,
  LIP_SPIN = 44,
  LIP_EXTEND = 70,
  SAVE_RUNWAY_X = {25, 90},
  SAVE_RUNWAY_Y = {380, 430},
  SAVE_RUNWAY_FIRE_X = {25, 60},
  SAVE_HUMAN_SEAT_X = {25, 30},
  SAVE_ARM_PUMP = false,
  FLOOR_RECLIMB_Y = 480,
  FLOOR_RUNWAY_X = {270, 310},
  FLOOR_RUNWAY_Y = 500,
  FLOOR_RECLIMB_CHARGE = 12,
  FLOOR_RECLIMB_SPIN = 44,
  PHASE_C_X_MIN = 300,
  PHASE_C_Y_MAX = 430,
  PHASE_C_Y_MIN = 200,
  PHASE_D_X = 300,
  PHASE_D_Y = 200,
}

local M = {}
M.policy = P
M.ROOM_BAT_CAVE, M.ROOM_BUBBLE = ROOM_BAT, ROOM_BUBBLE
M.BUBBLE_PHASE_C_X_MIN = P.PHASE_C_X_MIN
M.BUBBLE_PHASE_C_Y_MAX = P.PHASE_C_Y_MAX
M.BUBBLE_PHASE_C_Y_MIN = P.PHASE_C_Y_MIN
M.BUBBLE_PHASE_D_X = P.PHASE_D_X
M.BUBBLE_PHASE_D_Y = P.PHASE_D_Y

local function new_track(session)
  return geom.new_climb_track(session, {label = "bubble_to_bat_cave"})
end

local function fire_or_mid_pin(state)
  return geom.on_save_runway(state, P) and not geom.on_launch_lip(state, P)
end

local function land_and_prepare(session, track, land_frames)
  land_frames = land_frames or 40
  local label = track.label
  iceg.require_room(session, ROOM_BUBBLE, label)
  for _ = 1, land_frames do
    local state = session:hold(1, {}, label .. "_land")
    if geom.on_mid_iso_pin(state, P) then break end
    if state.velocity_y == 0 and P.STANDING_POSES[state.pose] then break end
  end
  iceg.unmorph(session)
  iceg.select_weapon(session, 0)
  if session.state.samus_x > track.max_x then track.max_x = session.state.samus_x end
  if session.state.samus_y < track.min_y then track.min_y = session.state.samus_y end
end

local function lower_to_mid_pin(session, track)
  local label = track.label
  if fire_or_mid_pin(session.state) then
    track.mid_reached = true
    geom.track_state(session, track, session.state, P)
    return
  end
  for frame = 0, 139 do
    local state = session.state
    if state.room_id ~= ROOM_BUBBLE then break end
    geom.track_state(session, track, state, P)
    if fire_or_mid_pin(state) then
      track.mid_reached = true
      break
    end
    if geom.avoid_wrong_door(session, track, state, P) then
      -- continue
    elseif state.pose == 137 or state.pose == 138 then
      session:hold(1, {"RIGHT", "B", "A"}, label .. "_floor_kb")
    elseif state.samus_x >= P.FLOOR_SHELF_X and geom.is_true_ground(state, {poses = P.TRUE_GROUND}) then
      break
    elseif frame % 12 < 3 then
      session:hold(1, {"RIGHT", "B", "X"}, label .. "_floor_shot")
    else
      session:hold(1, {"RIGHT", "B"}, label .. "_floor_walk")
    end
  end
  local shelf_i = 1
  for _ = 1, P.LOWER_FRAMES do
    local state = session.state
    if state.room_id ~= ROOM_BUBBLE then break end
    geom.track_state(session, track, state, P)
    if fire_or_mid_pin(state) then
      track.mid_reached = true
      break
    end
    if geom.avoid_wrong_door(session, track, state, P) then
      -- continue
    elseif state.pose == 137 or state.pose == 138 then
      for k = 1, 8 do session:hold(1, {"RIGHT", "B", "A"}, label .. "_lower_kb") end
    elseif state.pose == 27 or state.pose == 28 then
      session:hold(1, {"UP"}, label .. "_lower_unmorph")
    else
      local x, y = state.samus_x, state.samus_y
      if x > 250 and y > P.MID_Y then
        session:hold(1, {"LEFT", "B"}, label .. "_lower_cavity")
      else
        local shelves = P.LOWER_SHELVES
        while shelf_i < #shelves
            and y <= shelves[shelf_i][2] + 12
            and math.abs(x - shelves[shelf_i][1]) < 40 do
          shelf_i = shelf_i + 1
        end
        local tx, ty = shelves[shelf_i][1], shelves[shelf_i][2]
        local grounded = geom.is_true_ground(state, {poses = P.TRUE_GROUND})
        if grounded and y > ty + 20 then
          for k = 1, 8 do session:hold(1, {"A"}, label .. "_lower_charge") end
          local dir_h
          if x < tx - 10 then dir_h = "RIGHT"
          elseif x > tx + 10 then dir_h = "LEFT"
          else dir_h = (x < 115) and "RIGHT" or "LEFT" end
          local hop = ((y - ty) > 80) and 28 or 36
          for k = 1, hop do
            state = session:hold(1, {dir_h, "B", "A"}, label .. "_lower_hop")
            geom.track_state(session, track, state, P)
            if state.room_id ~= ROOM_BUBBLE then break end
            if fire_or_mid_pin(state) then
              track.mid_reached = true
              break
            end
          end
          if track.mid_reached or state.room_id ~= ROOM_BUBBLE then break end
        elseif grounded and y <= ty + 20 then
          if math.abs(x - tx) > 8 then
            local dir_h = (x < tx) and "RIGHT" or "LEFT"
            session:hold(1, {dir_h, "B"}, label .. "_lower_align")
          else
            session:hold(1, {}, label .. "_lower_idle")
          end
          if fire_or_mid_pin(state) then
            track.mid_reached = true
            break
          end
        else
          local dir_h
          if x < tx - 5 then dir_h = "RIGHT"
          elseif x > tx + 5 then dir_h = "LEFT"
          else dir_h = (x > 120) and "LEFT" or "RIGHT" end
          session:hold(1, {dir_h, "B", "A"}, label .. "_lower_air")
        end
      end
    end
  end
end

local function mid_repin(session, track)
  local label = track.label
  local stand_lo, stand_hi = P.MID_STAND_X[1], P.MID_STAND_X[2]
  for _ = 1, P.MID_REPIN_FRAMES do
    local state = session.state
    if state.room_id ~= ROOM_BUBBLE then break end
    geom.track_state(session, track, state, P)
    if geom.phase_d_top_band(state, P) then
      track.top_reached = true
      track.standing_mid_pinned = true
      break
    end
    if geom.on_save_runway(state, P) and not geom.on_launch_lip(state, P) then
      track.standing_mid_pinned = true
      track.mid_reached = true
      break
    end
    if geom.avoid_wrong_door(session, track, state, P) then
      -- continue
    elseif state.pose == 137 or state.pose == 138 then
      for k = 1, 10 do session:hold(1, {"RIGHT", "B", "A"}, label .. "_repin_kb") end
    elseif geom.on_mid_iso_pin(state, P) then
      for k = 1, 4 do
        state = session:hold(1, {}, label .. "_repin_settle")
        geom.track_state(session, track, state, P)
      end
      if state.room_id == ROOM_BUBBLE and geom.on_mid_iso_pin(state, P) then
        track.standing_mid_pinned = true
        track.mid_reached = true
        break
      end
    else
      local x, y = state.samus_x, state.samus_y
      if x > P.CAVITY_X_MAX and y > P.TOP_Y then
        session:hold(1, {"LEFT", "B"}, label .. "_repin_cap")
      elseif y > P.MID_Y + 10 then
        session:hold(1, {(x < 160) and "RIGHT" or "LEFT", "B", "A"}, label .. "_repin_low_spin")
      elseif state.velocity_y == 0 and P.STAND_PIN[state.pose] then
        if x < stand_lo then
          session:hold(1, {"RIGHT", "B"}, label .. "_repin_walk_r")
        elseif x > stand_hi then
          session:hold(1, {"LEFT", "B"}, label .. "_repin_walk_l")
        else
          session:hold(1, {}, label .. "_repin_idle")
        end
      else
        local dir_h = (x < stand_lo) and "RIGHT" or ((x > stand_hi) and "LEFT" or "RIGHT")
        session:hold(1, {dir_h, "B"}, label .. "_repin_air")
      end
    end
  end
end

local function play_full(session, track)
  land_and_prepare(session, track, 40)
  lower_to_mid_pin(session, track)
  mid_repin(session, track)
  mid.run_mid_loop(session, track, "launch", P)
  return door.top_super_door(session, track, {policy = P})
end

function M.play_bubble_to_bat_cave(session)
  return play_full(session, new_track(session))
end

return M
