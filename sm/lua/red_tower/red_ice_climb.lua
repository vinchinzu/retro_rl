-- Enemy-aware Red Tower Ice climb: bottom floor → ordinary Hellway left-door.
-- Freeze Rippers, hop platforms. No wall-jump on ice tops.

local rooms = require("rooms")
local ram = require("ram")
local ctrl = require("red_tower.ctrl")
local geom = require("red_tower.geometry")
local hell = require("red_tower.red_to_hellway")

local M = {}
local ROOM_RED_TOWER = rooms.ROOM_RED_TOWER or 0xA253
local ROOM_HELLWAY = rooms.ROOM_HELLWAY or 0xA2F7
local ENEMY_BASE = 0x0F78
local ENEMY_STRIDE = 0x40
local RIPPER_ID = 0xD47F
local ICE_BEAM_MASK = ctrl.ICE_BEAM_MASK
local HI_JUMP_MASK = ctrl.HI_JUMP_MASK
local STAND = {[1]=true,[2]=true}
local CROUCH = {[39]=true,[40]=true}
local TRUE_MORPH = {[29]=true,[30]=true,[31]=true,[32]=true}

M.RIPPER_ID = RIPPER_ID
M.BOTTOM_RIPPER_Y = 2376
M.MID_RIPPER_Y = 2280
M.LOW_RIPPER_3_Y = 2184
M.LOW_RIPPER_4_Y = 2048
M.TUNNEL_FLOOR_Y = 1883
M.MID_FLOOR_Y = 1625
M.THIN_SEAT_Y = 587
M.UPPER_RIPPER_1_Y = 520
M.UPPER_RIPPER_2_Y = 416
M.UPPER_RIPPER_3_Y = 320
M.UPPER_RIPPER_4_Y = 232
M.UPPER_RIPPER_1_LAND_Y = 495
M.UPPER_RIPPER_2_LAND_Y = 391
M.UPPER_RIPPER_3_LAND_Y = 295
M.UPPER_RIPPER_4_LAND_Y = 207
M.MID_RIPPER_LAND_Y = 2255
M.LOW_RIPPER_3_LAND_Y = 2159
M.LOW_RIPPER_4_LAND_Y = 2023
M.BOTTOM_RIPPER_LAND_Y = 2351

local function ck(id, x0, x1, y0, y1, extra)
  extra = extra or {}
  extra.checkpoint_id = id
  extra.x0, extra.x1, extra.y0, extra.y1 = x0, x1, y0, y1
  extra.grounded = extra.grounded ~= false
  extra.room_id = extra.room_id or ROOM_RED_TOWER
  extra.min_freeze_timer = extra.min_freeze_timer or 0
  return extra
end

local function matches(cp, state)
  if ctrl.num(state.room_id) ~= cp.room_id then
    return false
  end
  local x, y = ctrl.x(state), ctrl.y(state)
  if not (cp.x0 <= x and x <= cp.x1 and cp.y0 <= y and y <= cp.y1) then
    return false
  end
  if cp.grounded and not (ctrl.num(state.velocity_y) == 0 and ctrl.num(state.vertical_direction) == 0) then
    return false
  end
  return true
end

M.BOTTOM_FLOOR = ck("bottom_floor", 48, 220, 2435, 2450)
M.LOWER_RIPPER_1 = ck("lower_ripper_1", 55, 175, 2335, 2360, {support_enemy_y = 2376, min_freeze_timer = 30})
M.LOWER_RIPPER_2 = ck("lower_ripper_2", 55, 205, 2238, 2270, {support_enemy_y = 2280, min_freeze_timer = 30})
M.LOWER_RIPPER_3 = ck("lower_ripper_3", 55, 205, 2142, 2174, {support_enemy_y = 2184, min_freeze_timer = 30})
M.LOWER_RIPPER_4 = ck("lower_ripper_4", 55, 205, 2006, 2038, {support_enemy_y = 2048, min_freeze_timer = 30})
M.TUNNEL_FLOOR = ck("tunnel_floor", 80, 125, 1870, 1895)
M.MID_FLOOR = ck("mid_floor", 130, 195, 1618, 1632)
M.THIN_SEAT = ck("thin_seat", 70, 110, 575, 600)
M.UPPER_RIPPER_1 = ck("upper_ripper_1", 70, 180, 478, 512, {support_enemy_y = 520, min_freeze_timer = 30})
M.UPPER_RIPPER_2 = ck("upper_ripper_2", 65, 180, 374, 408, {support_enemy_y = 416, min_freeze_timer = 30})
M.UPPER_RIPPER_3 = ck("upper_ripper_3", 65, 180, 278, 312, {support_enemy_y = 320, min_freeze_timer = 30})
M.UPPER_RIPPER_4 = ck("upper_ripper_4", 65, 180, 190, 224, {support_enemy_y = 232, min_freeze_timer = 30})
M.HELLWAY_SILL = ck("hellway_sill", 16, 80, 120, 175, {grounded = false, room_id = ROOM_HELLWAY})

local function wram_u16(session, addr)
  if ram.u16 then
    return ram.u16(session, addr)
  end
  local buf = ctrl.wram(session)
  return ctrl.u16(buf, addr)
end

function M.read_rippers(session)
  local out = {}
  local slot
  for slot = 0, 11 do
    local base = ENEMY_BASE + slot * ENEMY_STRIDE
    if wram_u16(session, base) == RIPPER_ID then
      local x = wram_u16(session, base + 0x02)
      local y = wram_u16(session, base + 0x06)
      if not (x >= 0xFE00 or y >= 0xFE00 or (x == 0 and y == 0)) then
        out[#out + 1] = {
          slot = slot, x = x, y = y,
          freeze_timer = wram_u16(session, base + 0x26),
        }
      end
    end
  end
  return out
end

function M.ripper_at_height(session, target_y, tolerance)
  tolerance = tolerance or 12
  local best, best_d
  local i, enemies = 1, M.read_rippers(session)
  for i = 1, #enemies do
    local d = math.abs(enemies[i].y - target_y)
    if d <= tolerance and (not best_d or d < best_d or (d == best_d and enemies[i].slot < best.slot)) then
      best, best_d = enemies[i], d
    end
  end
  return best
end

local function supported(session, state, cp)
  if not matches(cp, state) then
    return false
  end
  if not cp.support_enemy_y then
    return true
  end
  local enemy = M.ripper_at_height(session, cp.support_enemy_y)
  if not enemy or enemy.freeze_timer < cp.min_freeze_timer then
    return false
  end
  return math.abs(ctrl.x(state) - enemy.x) <= 24
end

local function has_ice_hj(state)
  return ctrl.band(ctrl.num(state.equipped_beams), ICE_BEAM_MASK) == ICE_BEAM_MASK
    and ctrl.band(ctrl.num(state.equipped_items), HI_JUMP_MASK) == HI_JUMP_MASK
end

function M.can_attach_bottom_edge(state)
  return matches(M.BOTTOM_FLOOR, state) and has_ice_hj(state)
end
function M.can_attach_ripper1_edge(state)
  return matches(M.LOWER_RIPPER_1, state) and has_ice_hj(state)
end
function M.can_attach_ripper2_edge(state)
  return matches(M.LOWER_RIPPER_2, state) and has_ice_hj(state)
end
function M.can_attach_ripper3_edge(state)
  return matches(M.LOWER_RIPPER_3, state) and has_ice_hj(state)
end
function M.can_attach_ripper4_edge(state)
  return matches(M.LOWER_RIPPER_4, state) and has_ice_hj(state)
end
function M.can_attach_tunnel_edge(state)
  return matches(M.TUNNEL_FLOOR, state) and has_ice_hj(state)
end
function M.can_attach_mid_floor_edge(state)
  return matches(M.MID_FLOOR, state) and has_ice_hj(state)
end
function M.can_attach_thin_seat_edge(state)
  return matches(M.THIN_SEAT, state) and has_ice_hj(state)
end
function M.can_attach_upper_ripper1_edge(state)
  return matches(M.UPPER_RIPPER_1, state) and has_ice_hj(state)
end
function M.can_attach_upper_ripper2_edge(state)
  return matches(M.UPPER_RIPPER_2, state) and has_ice_hj(state)
end
function M.can_attach_upper_ripper3_edge(state)
  return matches(M.UPPER_RIPPER_3, state) and has_ice_hj(state)
end

local function toward(x, target)
  if x < target then return "RIGHT" end
  return "LEFT"
end

local function grounded(state)
  return ctrl.num(state.velocity_y) == 0 and ctrl.num(state.vertical_direction) == 0
end

-- Generic hop: freeze next Ripper, standing/crouch jump, drift onto ice.
local function play_hop(session, spec)
  local from_cp, to_cp = spec.from_cp, spec.to_cp
  if not spec.attach(session.state) then
    ctrl.timeout(spec.policy_id .. ": not on " .. from_cp.checkpoint_id .. " " .. ctrl.brief(session.state))
  end
  local phase = spec.start_phase or "stand"
  local frames, attempts, phase_frames, settle_frames = 0, 0, 0, 0
  local target_x, failed, complete, failure = 0, false, false, ""
  local max_frames = spec.max_frames or 360
  local function emit(names, reason)
    frames = frames + 1
    phase_frames = phase_frames + 1
    if frames > max_frames then
      failed, failure, phase = true, "budget>" .. max_frames .. "f", "failed"
      return
    end
    ctrl.step(session, names, reason)
  end
  local function set_phase(p)
    phase, phase_frames = p, 0
  end
  while not complete and not failed do
    local state = session.state
    if spec.on_hellway and spec.on_hellway(state) then
      complete = true
      break
    end
    if ctrl.num(state.room_id) == ROOM_HELLWAY and spec.allow_hellway_door then
      emit({"RIGHT"}, spec.tag .. "_door")
    elseif ctrl.num(state.room_id) ~= ROOM_RED_TOWER then
      failed, failure = true, string.format("left room 0x%04X", ctrl.num(state.room_id))
    elseif phase == "stand" then
      local pose = ctrl.num(state.pose)
      if STAND[pose] or pose == 3 or pose == 4 or (spec.allow_crouch_stand and CROUCH[pose]) then
        if spec.face_right then set_phase("face") else set_phase("acquire") end
      else
        emit({"UP"}, spec.tag .. "_stand")
      end
    elseif phase == "face" then
      if ctrl.num(state.facing) == ctrl.FACING_RIGHT or phase_frames >= 8 then
        set_phase("acquire")
      elseif ctrl.x(state) >= 100 then
        emit({"UP"}, spec.tag .. "_hold_seat")
      else
        emit({"RIGHT"}, spec.tag .. "_face")
      end
    elseif phase == "select_beam" then
      if ctrl.band(ctrl.num(state.equipped_beams), ICE_BEAM_MASK) ~= ICE_BEAM_MASK then
        failed, failure = true, "Ice Beam is not equipped"
      elseif ctrl.band(ctrl.num(state.equipped_items), HI_JUMP_MASK) ~= HI_JUMP_MASK then
        failed, failure = true, "Hi-Jump is not equipped"
      elseif ctrl.num(state.selected_item) == 0 then
        set_phase("acquire")
      else
        emit({"SELECT"}, "red_ice_select_beam")
      end
    elseif phase == "acquire" then
      if spec.kind == "r4tun" then
        set_phase("crouch")
      elseif spec.kind == "ur3hw" then
        local support = M.ripper_at_height(session, M.UPPER_RIPPER_3_Y)
        local enemy = M.ripper_at_height(session, M.UPPER_RIPPER_4_Y)
        if not support or support.freeze_timer < 22 then
          failed, failure = true, "support thawed before next freeze"
        elseif not enemy then
          emit({}, spec.tag .. "_wait_r")
        else
          local signed = enemy.x - ctrl.x(state)
          local lo, hi = 10, 28
          if enemy.freeze_timer > 40 and signed >= lo then
            target_x = enemy.x
            set_phase("drop_aim")
          elseif lo <= signed and signed <= hi then
            if ctrl.num(state.pose) ~= 3 and ctrl.num(state.pose) ~= 4 then
              emit({"UP"}, spec.tag .. "_aim")
            else
              emit({"UP", "X"}, spec.tag .. "_freeze_shot")
            end
          else
            emit({"UP"}, spec.tag .. "_wait_dx")
          end
        end
      elseif spec.kind == "bottom" then
        local enemy = M.ripper_at_height(session, M.BOTTOM_RIPPER_Y)
        if not enemy then
          failed, failure = true, "lower Ripper missing"
        elseif enemy.freeze_timer > 40 then
          target_x = enemy.x
          set_phase("drop_aim")
        else
          local sx = ctrl.x(state)
          if 92 <= enemy.x and enemy.x <= 145 and math.abs(enemy.x - sx) <= 6 then
            emit({"UP", "X"}, "red_ice_freeze_shot")
          else
            local tx = math.max(90, math.min(148, enemy.x))
            if math.abs(tx - sx) <= 8 then
              emit({"UP"}, "red_ice_wait_phase")
            else
              emit({toward(sx, tx)}, "red_ice_track_phase")
            end
          end
        end
      else
        if spec.support_y then
          local support = M.ripper_at_height(session, spec.support_y)
          if not support or support.freeze_timer < 22 then
            failed, failure = true, "support thawed before next freeze"
          end
        end
        if not failed then
          local enemy = M.ripper_at_height(session, spec.target_y)
          if not enemy then
            emit({}, spec.tag .. "_wait_r")
          else
            local signed = enemy.x - ctrl.x(state)
            local signed_ok = spec.freeze_signed
            if signed_ok == nil then
              signed_ok = true
            end
            local offset = signed_ok and signed or math.abs(signed)
            local lo, hi = spec.freeze_lo or 8, spec.freeze_hi or 36
            if enemy.freeze_timer > 40 and offset >= lo then
              target_x = enemy.x
              set_phase("drop_aim")
            elseif lo <= offset and offset <= hi then
              if spec.aim_before_shot and ctrl.num(state.pose) ~= 3 and ctrl.num(state.pose) ~= 4 then
                emit({"UP"}, spec.tag .. "_aim")
              else
                emit({"UP", "X"}, spec.tag .. "_freeze_shot")
              end
            else
              emit({"UP"}, spec.tag .. "_wait_dx")
            end
          end
        end
      end
    elseif phase == "drop_aim" then
      if STAND[ctrl.num(state.pose)] or phase_frames >= 10 then
        if spec.kind == "bottom" then
          set_phase("step_off")
        elseif spec.kind == "ur3hw" then
          set_phase("break")
        elseif spec.crouch_jump or spec.kind == "r4tun" then
          set_phase("crouch")
        else
          set_phase("jump")
        end
      else
        emit({}, spec.tag .. "_drop_aim")
      end
    elseif phase == "step_off" then
      local enemy = M.ripper_at_height(session, M.BOTTOM_RIPPER_Y)
      local ex = enemy and enemy.x or target_x
      local x = ctrl.x(state)
      if math.abs(x - ex) >= 28 or x <= 68 or x >= 180 or phase_frames >= 28 then
        set_phase("brake")
      else
        local dir
        if x >= ex then
          dir = (x < 190) and "RIGHT" or "LEFT"
        else
          dir = (x > 60) and "LEFT" or "RIGHT"
        end
        emit({dir}, "red_ice_step_off")
      end
    elseif phase == "brake" then
      if phase_frames >= 4 then
        set_phase("jump")
      else
        local enemy = M.ripper_at_height(session, M.BOTTOM_RIPPER_Y)
        local ex = enemy and enemy.x or target_x
        emit({toward(ctrl.x(state), ex)}, "red_ice_brake")
      end
    elseif phase == "crouch" then
      if CROUCH[ctrl.num(state.pose)] or phase_frames >= 8 then
        set_phase("jump")
      else
        emit({"DOWN"}, spec.tag .. "_crouch")
      end
    elseif phase == "jump" then
      if spec.kind == "r4tun" then
        local y, x = ctrl.y(state), ctrl.x(state)
        local airborne = (not grounded(state)) or y <= 2015
        if airborne and (y <= 1860 or x <= 125) then
          set_phase("land")
        elseif airborne then
          emit({"LEFT", "A"}, "red_ice_r4tun_rise_left")
        else
          emit({"A"}, "red_ice_r4tun_jump")
        end
      else
        local until_y = spec.jump_until_y or ((spec.land_y or 0) - 27)
        if ctrl.y(state) <= until_y or phase_frames >= (spec.jump_max_frames or 32) then
          set_phase("land")
        else
          emit({"A"}, spec.tag .. "_jump")
        end
      end
    elseif phase == "break" then
      if phase_frames >= 12 then
        set_phase("rise")
      else
        emit({"UP", "X", "A"}, spec.tag .. "_break_shot")
      end
    elseif phase == "rise" then
      local y = ctrl.y(state)
      if y <= 140 then
        set_phase("sill")
      elseif y >= 360 then
        failed, failure = true, "fell xy=(" .. ctrl.x(state) .. "," .. y .. ")"
      elseif matches(M.UPPER_RIPPER_3, state) then
        failed, failure = true, "landed back on upper_ripper_3"
      else
        emit({"A"}, spec.tag .. "_rise")
      end
    elseif phase == "sill" then
      if spec.on_hellway and spec.on_hellway(state) then
        complete = true
      elseif ctrl.y(state) >= 360 then
        failed, failure = true, "fell xy=(" .. ctrl.x(state) .. "," .. ctrl.y(state) .. ")"
      else
        local y, x = ctrl.y(state), ctrl.x(state)
        if y <= 155 and (grounded(state) or y <= 142) then
          if x >= 200 then
            emit({"RIGHT", "X"}, spec.tag .. "_sill_right")
          else
            emit({"RIGHT"}, spec.tag .. "_sill_right")
          end
        else
          emit({"A"}, spec.tag .. "_sill_keep_up")
        end
      end
    elseif phase == "land" then
      if spec.kind == "r4tun" then
        if matches(M.TUNNEL_FLOOR, state) then
          set_phase("settle")
        elseif grounded(state) and matches(M.LOWER_RIPPER_4, state) then
          attempts = attempts + 1
          if attempts >= 2 or not matches(from_cp, state) then
            failed, failure = true, "landed back on r4"
          else
            set_phase("crouch")
          end
        elseif grounded(state) and matches(M.LOWER_RIPPER_3, state) then
          failed, failure = true, "fell past r4 " .. ctrl.brief(state)
        elseif ctrl.y(state) >= 2300 then
          failed, failure = true, "fell to shaft " .. ctrl.brief(state)
        elseif TRUE_MORPH[ctrl.num(state.pose)] then
          emit({"UP"}, spec.tag .. "_unmorph")
        else
          local x, y = ctrl.x(state), ctrl.y(state)
          if x > 107 then
            emit({"LEFT"}, spec.tag .. "_drift_left")
          elseif x < 92 and y <= M.TUNNEL_FLOOR_Y + 20 then
            emit({"RIGHT"}, spec.tag .. "_nudge")
          else
            emit({}, spec.tag .. "_fall")
          end
        end
      elseif spec.kind == "bottom" then
        if supported(session, state, M.LOWER_RIPPER_1) then
          set_phase("settle")
        elseif grounded(state) and (matches(M.BOTTOM_FLOOR, state) or not matches(M.LOWER_RIPPER_1, state)) then
          attempts = attempts + 1
          if attempts >= 2 or not matches(M.BOTTOM_FLOOR, state) then
            failed, failure = true, "missed Ripper " .. ctrl.brief(state)
          else
            set_phase("acquire")
          end
        elseif TRUE_MORPH[ctrl.num(state.pose)] then
          emit({"UP"}, "red_ice_unmorph")
        else
          local enemy = M.ripper_at_height(session, M.BOTTOM_RIPPER_Y)
          if not enemy or enemy.freeze_timer <= 30 then
            failed, failure = true, "Ripper thawed before landing"
          elseif math.abs(ctrl.x(state) - enemy.x) > 3 then
            emit({toward(ctrl.x(state), enemy.x)}, "red_ice_land_track")
          else
            emit({}, "red_ice_fall")
          end
        end
      else
        if supported(session, state, to_cp) then
          set_phase("settle")
        elseif grounded(state) and spec.fail_y and ctrl.y(state) >= spec.fail_y then
          failed, failure = true, "fell off seat " .. ctrl.brief(state)
        elseif grounded(state) and spec.past_cp and matches(spec.past_cp, state) then
          if spec.past_is_retry then
            attempts = attempts + 1
            if attempts >= 2 or not matches(from_cp, state) then
              failed, failure = true, "fell past " .. from_cp.checkpoint_id
            else
              set_phase(spec.face_right and "face" or "acquire")
            end
          else
            failed, failure = true, "fell past " .. from_cp.checkpoint_id
          end
        elseif grounded(state) and matches(from_cp, state) then
          attempts = attempts + 1
          if attempts >= 2 or not matches(from_cp, state) then
            failed, failure = true, "landed back on " .. from_cp.checkpoint_id
          else
            set_phase(spec.face_right and "face" or "acquire")
          end
        elseif TRUE_MORPH[ctrl.num(state.pose)] then
          emit({"UP"}, spec.tag .. "_unmorph")
        else
          local enemy = M.ripper_at_height(session, spec.target_y)
          local ex = enemy and enemy.x or target_x
          local y, x = ctrl.y(state), ctrl.x(state)
          local drift_high = (spec.land_y or 0) - (spec.drift_high_delta or 10)
          if y <= drift_high and math.abs(x - ex) > 3 then
            emit({toward(x, ex)}, spec.tag .. "_drift_high")
          elseif to_cp.y0 <= y and y <= to_cp.y1 + 5 then
            if spec.hover_track ~= false and math.abs(x - ex) > 3 then
              emit({toward(x, ex)}, spec.tag .. "_hover_track")
            else
              emit({}, spec.tag .. "_hover")
            end
          elseif math.abs(x - ex) > 3 then
            emit({toward(x, ex)}, spec.tag .. "_track")
          else
            emit({}, spec.tag .. "_fall")
          end
        end
      end
    elseif phase == "settle" then
      local ok
      if spec.kind == "r4tun" then
        ok = matches(M.TUNNEL_FLOOR, state)
      elseif spec.kind == "bottom" then
        ok = supported(session, state, M.LOWER_RIPPER_1)
      else
        ok = supported(session, state, to_cp)
      end
      if not ok then
        attempts = attempts + 1
        if attempts >= 2 or not matches(from_cp, state) then
          failed, failure = true, "unstable frozen support"
        else
          set_phase(spec.face_right and "face" or (spec.kind == "bottom" and "acquire" or "acquire"))
        end
      elseif settle_frames >= 8 then
        complete = true
      else
        settle_frames = settle_frames + 1
        emit({}, spec.tag .. "_checkpoint_settle")
      end
    else
      failed, failure = true, "unknown phase " .. tostring(phase)
    end
  end
  if failed or not complete then
    ctrl.timeout(string.format(
      "%s: %s; phase=%s frames=%d %s",
      spec.policy_id, failure ~= "" and failure or "did not complete",
      phase, frames, ctrl.brief(session.state)
    ))
  end
  return session.state
end

local function on_hellway(state)
  if ctrl.num(state.room_id) ~= ROOM_HELLWAY then
    return false
  end
  local gs = ctrl.num(state.game_state, 8)
  local door = ctrl.num(state.door_transition)
  local x, y = ctrl.x(state), ctrl.y(state)
  return gs == 8 and door == 0 and 16 <= x and x <= 80 and 100 <= y and y <= 180
end

function M.play_bottom_to_ripper1(session)
  return play_hop(session, {
    policy_id = "red_tower_ice_bottom_to_ripper1",
    tag = "red_ice",
    kind = "bottom",
    from_cp = M.BOTTOM_FLOOR,
    to_cp = M.LOWER_RIPPER_1,
    attach = M.can_attach_bottom_edge,
    start_phase = "select_beam",
    max_frames = 720,
  })
end

function M.play_ripper1_to_ripper2(session)
  return play_hop(session, {
    policy_id = "red_tower_ice_ripper1_to_ripper2", tag = "red_ice_r12",
    from_cp = M.LOWER_RIPPER_1, to_cp = M.LOWER_RIPPER_2,
    attach = M.can_attach_ripper1_edge, support_y = M.BOTTOM_RIPPER_Y,
    target_y = M.MID_RIPPER_Y, land_y = M.MID_RIPPER_LAND_Y,
    past_cp = M.BOTTOM_FLOOR, freeze_signed = false, hover_track = false,
    past_is_retry = true, max_frames = 280,
  })
end

function M.play_ripper2_to_ripper3(session)
  return play_hop(session, {
    policy_id = "red_tower_ice_ripper2_to_ripper3", tag = "red_ice_r23",
    from_cp = M.LOWER_RIPPER_2, to_cp = M.LOWER_RIPPER_3,
    attach = M.can_attach_ripper2_edge, support_y = M.MID_RIPPER_Y,
    target_y = M.LOW_RIPPER_3_Y, land_y = M.LOW_RIPPER_3_LAND_Y,
    past_cp = M.LOWER_RIPPER_1, hover_track = false, max_frames = 280,
  })
end

function M.play_ripper3_to_ripper4(session)
  return play_hop(session, {
    policy_id = "red_tower_ice_ripper3_to_ripper4", tag = "red_ice_r34",
    from_cp = M.LOWER_RIPPER_3, to_cp = M.LOWER_RIPPER_4,
    attach = M.can_attach_ripper3_edge, support_y = M.LOW_RIPPER_3_Y,
    target_y = M.LOW_RIPPER_4_Y, land_y = M.LOW_RIPPER_4_LAND_Y,
    past_cp = M.LOWER_RIPPER_2, crouch_jump = true, jump_until_y = 2008,
    drift_high_delta = 8, jump_max_frames = 36, max_frames = 360,
  })
end

function M.play_ripper4_to_tunnel(session)
  return play_hop(session, {
    policy_id = "red_tower_ice_ripper4_to_tunnel", tag = "red_ice_r4tun",
    kind = "r4tun", from_cp = M.LOWER_RIPPER_4, to_cp = M.TUNNEL_FLOOR,
    attach = M.can_attach_ripper4_edge, allow_crouch_stand = true, max_frames = 240,
  })
end

function M.play_tunnel_to_mid_floor(session)
  if not M.can_attach_tunnel_edge(session.state) then
    ctrl.timeout("red_tower_ice_tunnel_to_mid_floor: not on tunnel_floor " .. ctrl.brief(session.state))
  end
  hell.tunnel_to_midplat(session, "red_tower_ice_tunnel_to_mid_floor_ledge")
  local i
  local landed = false
  for i = 1, 50 do
    local state = session.state
    if grounded(state) and 1740 <= ctrl.y(state) and ctrl.y(state) <= 1770
        and 115 <= ctrl.x(state) and ctrl.x(state) <= 180 then
      landed = true
      break
    end
    ctrl.hold(session, 1, {}, "red_tower_ice_tunnel_to_mid_floor_ledge_land")
  end
  if not landed then
    ctrl.timeout("red_tower_ice_tunnel_to_mid_floor: missed bomb ledge " .. ctrl.brief(session.state))
  end
  ctrl.hold(session, 1, {"UP"}, "red_tower_ice_tunnel_to_mid_floor_stand")
  for i = 1, 12 do
    ctrl.hold(session, 1, {}, "red_tower_ice_tunnel_to_mid_floor_drop_aim")
  end
  for i = 1, 50 do
    if ctrl.x(session.state) >= 168 then
      break
    end
    ctrl.hold(session, 1, {"RIGHT"}, "red_tower_ice_tunnel_to_mid_floor_center")
  end
  ctrl.hold(session, 4, {}, "red_tower_ice_tunnel_to_mid_floor_center_settle")
  if not ctrl.is_morph(session.state.pose) then
    ctrl.ensure_morph(session)
  end
  local cycle
  for cycle = 0, 49 do
    hell.ibj_double(session, "red_tower_ice_tunnel_to_mid_floor_ibj_" .. cycle, 171, 1595)
    if ctrl.num(session.state.room_id) ~= ROOM_RED_TOWER then
      break
    end
    if ctrl.y(session.state) <= 1610 then
      for i = 1, 80 do
        local state = ctrl.hold(session, 1, {"LEFT"}, "red_tower_ice_tunnel_to_mid_floor_catch")
        if matches(M.MID_FLOOR, state) then
          ctrl.hold(session, 8, {}, "red_tower_ice_tunnel_to_mid_floor_settle")
          if matches(M.MID_FLOOR, session.state) then
            return session.state
          end
          break
        end
      end
    end
  end
  ctrl.timeout("red_tower_ice_tunnel_to_mid_floor: mid floor not reached " .. ctrl.brief(session.state))
end

function M.play_mid_floor_to_thin_seat(session)
  if not M.can_attach_mid_floor_edge(session.state) then
    ctrl.timeout("red_tower_ice_mid_floor_to_thin_seat: not on mid_floor " .. ctrl.brief(session.state))
  end
  hell.play_upper_rle(session, hell.HUMAN_FLOOR_RLE, "red_tower_ice_mid_floor_to_thin_seat_handoff")
  hell.seat_left_after_handoff(session, "red_tower_ice_mid_floor_to_thin_seat_left_seat")
  ctrl.hold(session, 3, {"LEFT", "B"}, "red_tower_ice_mid_floor_to_thin_seat_launch_run")
  ctrl.hold(session, 12, {"LEFT", "B", "A"}, "red_tower_ice_mid_floor_to_thin_seat_launch")
  hell.period_wj(session, "p0", "LEFT", 600, 1200)
  hell.period_wj(session, "p1", "RIGHT", 800, 1050)
  hell.period_wj(session, "p2", "LEFT", 800, 900, 16, 2, 2)
  hell.period_wj(session, "p3", "RIGHT", 800, 720, 16, 2, 2)
  hell.period_wj(session, "p4", "LEFT", 500, nil, 22, 2, 2)
  hell.period_wj(session, "p5", "RIGHT", 500, nil, 22, 8, 2)
  ctrl.hold(session, 8, {}, "red_tower_ice_mid_floor_to_thin_seat_settle")
  if matches(M.THIN_SEAT, session.state) then
    return session.state
  end
  ctrl.timeout("red_tower_ice_mid_floor_to_thin_seat: thin seat not reached " .. ctrl.brief(session.state))
end

function M.play_thin_seat_to_upper_ripper1(session)
  return play_hop(session, {
    policy_id = "red_tower_ice_thin_seat_to_upper_ripper1", tag = "red_ice_thinur1",
    from_cp = M.THIN_SEAT, to_cp = M.UPPER_RIPPER_1,
    attach = M.can_attach_thin_seat_edge, target_y = M.UPPER_RIPPER_1_Y,
    land_y = M.UPPER_RIPPER_1_LAND_Y, face_right = true, fail_y = 620, max_frames = 360,
  })
end

function M.play_upper_ripper1_to_2(session)
  return play_hop(session, {
    policy_id = "red_tower_ice_upper_ripper1_to_2", tag = "red_ice_ur12",
    from_cp = M.UPPER_RIPPER_1, to_cp = M.UPPER_RIPPER_2,
    attach = M.can_attach_upper_ripper1_edge, support_y = M.UPPER_RIPPER_1_Y,
    target_y = M.UPPER_RIPPER_2_Y, land_y = M.UPPER_RIPPER_2_LAND_Y,
    past_cp = M.THIN_SEAT, max_frames = 360,
  })
end

function M.play_upper_ripper2_to_3(session)
  return play_hop(session, {
    policy_id = "red_tower_ice_upper_ripper2_to_3", tag = "red_ice_ur23",
    from_cp = M.UPPER_RIPPER_2, to_cp = M.UPPER_RIPPER_3,
    attach = M.can_attach_upper_ripper2_edge, support_y = M.UPPER_RIPPER_2_Y,
    target_y = M.UPPER_RIPPER_3_Y, land_y = M.UPPER_RIPPER_3_LAND_Y,
    past_cp = M.UPPER_RIPPER_1, max_frames = 360,
  })
end

function M.play_upper_ripper3_to_hellway(session)
  return play_hop(session, {
    policy_id = "red_tower_ice_upper_ripper3_to_hellway", tag = "red_ice_ur3_hw",
    kind = "ur3hw", from_cp = M.UPPER_RIPPER_3, to_cp = M.HELLWAY_SILL,
    attach = M.can_attach_upper_ripper3_edge, on_hellway = on_hellway,
    allow_hellway_door = true, max_frames = 480,
  })
end

function M.play_ice_climb_to_hellway(session)
  if not M.can_attach_bottom_edge(session.state) then
    ctrl.timeout("red_tower_ice_bottom_to_hellway: not on Ice+HJ bottom floor " .. ctrl.brief(session.state))
  end
  M.play_bottom_to_ripper1(session)
  M.play_ripper1_to_ripper2(session)
  M.play_ripper2_to_ripper3(session)
  M.play_ripper3_to_ripper4(session)
  M.play_ripper4_to_tunnel(session)
  M.play_tunnel_to_mid_floor(session)
  M.play_mid_floor_to_thin_seat(session)
  M.play_thin_seat_to_upper_ripper1(session)
  M.play_upper_ripper1_to_2(session)
  M.play_upper_ripper2_to_3(session)
  M.play_upper_ripper3_to_hellway(session)
  local state = session.state
  if not matches(M.HELLWAY_SILL, state) then
    ctrl.timeout(string.format(
      "red_tower_ice_bottom_to_hellway: not ordinary Hellway left-door room=0x%04X %s",
      ctrl.num(state.room_id), ctrl.brief(state)
    ))
  end
  return state
end

return M
