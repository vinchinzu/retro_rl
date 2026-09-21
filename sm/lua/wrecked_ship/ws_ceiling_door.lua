-- Blue ceiling-door tap cycle shared by Basement hatch and Attic door.

local ctrl = require("wrecked_ship.ctrl")

local M = {}
local HURT = {[137]=true,[138]=true}
local LAND = {[1]=true,[2]=true,[9]=true,[10]=true}
local AIR = {[21]=true,[22]=true,[25]=true,[26]=true,[47]=true,[48]=true,[81]=true,[82]=true,[105]=true,[106]=true}
local SHOOT_FRAMES, CYCLE, FIRE_PHASE, WAIT_PHASE, SHAFT_SLACK = 240, 80, 60, 68, 4

function M.tap_up_action(frame, hold_charge, fire_phase, wait_phase)
  fire_phase = fire_phase or FIRE_PHASE
  wait_phase = wait_phase or WAIT_PHASE
  local phase = frame % CYCLE
  if phase < fire_phase then
    return {"UP", "X"}
  end
  local waiting = phase < wait_phase
  if hold_charge then
    waiting = frame < SHOOT_FRAMES or waiting
  end
  if waiting then
    return {"UP"}
  end
  return {"UP", "A"}
end

function M.ceiling_door_action(samus_x, samus_y, pose, frame, opts)
  opts = opts or {}
  if HURT[pose] then
    return {}
  end
  local seat_x, lip_y, shaft_y = opts.seat_x, opts.lip_y, opts.shaft_y
  local slack = opts.slack or 8
  if samus_y < shaft_y then
    if samus_x > seat_x + SHAFT_SLACK then
      return {"LEFT", "A"}
    end
    if samus_x < seat_x - SHAFT_SLACK then
      return {"RIGHT", "A"}
    end
    return {"A"}
  end
  if samus_y < lip_y then
    if AIR[pose] then
      if samus_x > seat_x + SHAFT_SLACK then
        return {"LEFT", "UP", "A"}
      end
      if samus_x < seat_x - SHAFT_SLACK then
        return {"RIGHT", "UP", "A"}
      end
    end
    if samus_x > seat_x + slack then
      return {"LEFT", "UP", "A"}
    end
    if samus_x < seat_x - slack then
      return {"RIGHT", "UP", "A"}
    end
    return M.tap_up_action(frame, opts.hold_charge, opts.fire_phase, opts.wait_phase)
  end
  return nil
end

function M.play_ceiling_door(session, opts)
  local dest = opts.dest_room
  local label = opts.label
  if ctrl.num(session.state.room_id) == dest then
    return
  end
  local shoot_i, i = 0, 1
  for i = 1, opts.budget or 800 do
    local st = session.state
    opts.guard(session, label)
    if ctrl.num(st.room_id) == dest then
      return
    end
    local side = opts.side_rooms
    local in_side = false
    if side then
      local s
      for s = 1, #side do
        if ctrl.num(st.room_id) == side[s] then
          in_side = true
        end
      end
    end
    if in_side then
      if opts.on_side_room then
        opts.on_side_room(session, label)
      else
        ctrl.timeout(label .. ": side room 0x" .. string.format("%04X", ctrl.num(st.room_id)))
      end
    elseif ctrl.is_knockback(st) then
      opts.on_knockback(session, label .. "_door_kb")
    elseif ctrl.is_morph(st.pose) then
      ctrl.unmorph(session)
    else
      local names, reason
      if ctrl.y(st) > opts.lip_y then
        shoot_i = 0
        names, reason = opts.remount(st), label .. "_remount"
      else
        names = opts.door_action(st, shoot_i)
        shoot_i = shoot_i + 1
        reason = label .. "_door"
      end
      if names and names[1] then
        ctrl.hold(session, 1, names, reason)
      else
        ctrl.hold(session, 1, {}, label .. "_hurt")
      end
    end
  end
  if ctrl.num(session.state.room_id) ~= dest then
    ctrl.timeout(label .. ": ceiling door missed: " .. ctrl.brief(session.state))
  end
end

function M.settle_ceiling_dest(session, dest_room, opts)
  opts = opts or {}
  ctrl.wait_ordinary_room(session, dest_room, {
    settle_frames = opts.settle_frames or 200,
    label = opts.label or "ceiling",
  })
  local i
  for i = 1, opts.land_frames or 90 do
    local st = session.state
    if LAND[ctrl.num(st.pose)] and math.abs(ctrl.num(st.velocity_y)) <= 1 then
      break
    end
    ctrl.hold(session, 1, {}, (opts.label or "ceiling") .. "_land")
  end
  return session.state
end

return M
