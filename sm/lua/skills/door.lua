-- Super-door pressure helpers.

local geometry = require("skills.geometry")
local knockback = require("skills.knockback")
local controller = require("skills.controller")

local door = {}

local _pol
local function default_policy()
  if not _pol then
    _pol = require("skills.policies.bubble_to_bat")
  end
  return _pol
end

function door.super_door_pressure_frame(session, frame, opts)
  opts = opts or {}
  local label = opts.label or "door"
  local face = opts.face or "RIGHT"
  local weapon = opts.weapon or 2
  local period = opts.period
  local shoot_end = opts.shoot_end
  local idle_end = opts.idle_end
  local face_end = opts.face_end
  local run_end = opts.run_end
  local ensure_weapon = opts.ensure_weapon
  if ensure_weapon == nil then
    ensure_weapon = true
  end
  local reason = opts.reason or "door"
  if ensure_weapon and tonumber(session.state.selected_item) ~= weapon then
    controller.select_weapon(session, weapon)
  end
  local phase = frame % period
  local inputs
  if phase < shoot_end then
    inputs = { face, "X" }
  elseif idle_end ~= nil and phase < idle_end then
    inputs = {}
  elseif face_end ~= nil and phase < face_end then
    inputs = { face }
  elseif run_end ~= nil and phase < run_end then
    inputs = { face, "B" }
  else
    inputs = { face, "B", "A" }
  end
  return session:hold(1, inputs, label .. "_" .. reason)
end

function door.top_super_door(session, track, opts)
  opts = opts or {}
  local pol = opts.policy or default_policy()
  local label = track.label
  local room_id = pol.ROOM_ID
  local exit_id = pol.EXIT_ROOM_ID

  if session.state.selected_item ~= 2 then
    controller.select_weapon(session, 2)
  end

  for _ = 1, 24 do
    local st = session.state
    if st.room_id ~= room_id then
      break
    end
    geometry.track_state(session, track, st, pol)
    local pose = tonumber(st.pose)
    if pol.TRUE_GROUND[pose] or pose == 25 or pose == 26 or pose == 27 or pose == 28 then
      break
    end
    session:hold(1, { "RIGHT" }, label .. "_door_land")
  end
  local crouch_n = tonumber(pol.DOOR_CROUCH_FRAMES) or 8
  if crouch_n < 0 then
    crouch_n = 0
  end
  for _ = 1, crouch_n do
    if session.state.room_id ~= room_id then
      break
    end
    session:hold(1, { "DOWN" }, label .. "_door_crouch")
  end
  for _ = 1, 4 do
    if session.state.room_id ~= room_id then
      break
    end
    session:hold(1, {}, label .. "_door_uncrouch")
  end

  local exhausted = true
  for frame = 0, pol.DOOR_FRAMES - 1 do
    local state = session.state
    if state.room_id == exit_id then
      exhausted = false
      break
    end
    if state.room_id ~= room_id then
      exhausted = false
      break
    end
    geometry.track_state(session, track, state, pol)
    if geometry.avoid_wrong_door(session, track, state, pol) then
      -- already consumed a frame
    elseif knockback.is_knockback(state) then
      for _ = 1, 8 do
        session:hold(1, { "RIGHT", "B", "A" }, label .. "_door_kb")
      end
    else
      if state.selected_item ~= 2 then
        controller.select_weapon(session, 2)
      end
      local x = tonumber(state.samus_x) or 0
      local y = tonumber(state.samus_y) or 0
      if y <= pol.DOOR_SUPER_Y and x >= pol.DOOR_SUPER_X then
        track.door_reached = true
        if frame % 5 < 2 then
          session:hold(1, { "RIGHT", "X" }, label .. "_door_super")
        else
          session:hold(1, { "RIGHT", "B" }, label .. "_door_press")
        end
        if session.state.room_id == exit_id then
          exhausted = false
          break
        end
      elseif y > pol.DOOR_FALL_Y then
        if x >= pol.DOOR_OUTER_X and pol.TRUE_GROUND[tonumber(state.pose)] then
          session:hold(1, { "LEFT", "B", "A" }, label .. "_door_shelf_escape")
        elseif x >= pol.DOOR_OUTER_X then
          session:hold(1, { "LEFT", "B" }, label .. "_door_outer_pull")
        elseif x < 360 then
          session:hold(1, { "RIGHT", "B", "A" }, label .. "_door_reclimb")
        elseif pol.DOOR_WJ_POSES[tonumber(state.pose)] or controller.is_wall_latch(state) then
          controller.walljump_once(session, pol.DOOR_WJ, label .. "_door_reclimb_wj")
        else
          local ph = frame % pol.DOOR_WJ_PERIOD
          if ph < pol.DOOR_WJ_INTO then
            session:hold(1, { "LEFT", "A" }, label .. "_door_reseek")
          elseif ph < pol.DOOR_WJ_INTO + pol.DOOR_WJ_BOUNCE then
            session:hold(1, { "RIGHT", "A" }, label .. "_door_rebounce")
          else
            session:hold(1, { "RIGHT", "B", "A" }, label .. "_door_respin")
          end
        end
        if session.state.room_id == exit_id then
          exhausted = false
          break
        end
      elseif pol.DOOR_WJ_POSES[tonumber(state.pose)] or controller.is_wall_latch(state) then
        controller.walljump_once(session, pol.DOOR_WJ, label .. "_door_wj")
        if session.state.room_id == exit_id then
          exhausted = false
          break
        end
      elseif x > pol.DOOR_X_CAP then
        session:hold(1, { "LEFT", "B" }, label .. "_door_cap")
        if session.state.room_id == exit_id then
          exhausted = false
          break
        end
      else
        local ph = frame % pol.DOOR_WJ_PERIOD
        if ph < pol.DOOR_WJ_INTO then
          session:hold(1, { "LEFT", "A" }, label .. "_door_wj_into")
        elseif ph < pol.DOOR_WJ_INTO + pol.DOOR_WJ_BOUNCE then
          session:hold(1, { "RIGHT", "A" }, label .. "_door_wj_bounce")
        else
          session:hold(1, { "RIGHT", "B", "A" }, label .. "_door_wj_spin")
        end
        if session.state.room_id == exit_id then
          exhausted = false
          break
        end
      end
    end
  end

  if session.state.room_id ~= exit_id then
    local state = session.state
    if exhausted then
      error(string.format(
        "TimeoutError: %s: exit Super door missed before room 0x%04X; room=0x%04X pose=%d xy=(%d,%d) door_transition=%d max_x=%s min_y=%s mid_reached=%s top_reached=%s door_reached=%s supers=%s selected=%s",
        label,
        exit_id,
        state.room_id,
        state.pose,
        state.samus_x,
        state.samus_y,
        state.door_transition,
        tostring(track.max_x),
        tostring(track.min_y),
        tostring(track.mid_reached),
        tostring(track.top_reached),
        tostring(track.door_reached),
        tostring(state.super_missiles),
        tostring(state.selected_item)
      ), 2)
    end
    error(string.format(
      "TimeoutError: %s: left climb room without ordinary exit; room=0x%04X pose=%d xy=(%d,%d)",
      label,
      state.room_id,
      state.pose,
      state.samus_x,
      state.samus_y
    ), 2)
  end

  return controller.wait_ordinary_room(session, exit_id, {
    settle_frames = pol.TO_BAT_SETTLE_FRAMES,
    label = label,
  })
end

door.bubble_top_super_door = door.top_super_door

return door
