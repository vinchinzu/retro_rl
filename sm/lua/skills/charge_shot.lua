-- Charge-release beam shot: position, aim, hold X, release.

local ram = require("ram")

local charge_shot = {}

charge_shot.ADDR_BEAM_CHARGE = 0x0CD0
charge_shot.CHARGE_FULL = 60
charge_shot.JUMP_LEAD = 12
charge_shot.MOVEMENT_TURNING = 14
charge_shot.FIRE_RANGE_PX = 48
charge_shot.UNDER_BLOB_DX = 16
charge_shot.AIM_UP_DY = -40
charge_shot.AIM_DOWN_DY = 40
charge_shot.JUMP_SHOT_DY = -16

function charge_shot.is_turning(movement_type)
  return tonumber(movement_type) == charge_shot.MOVEMENT_TURNING
end

function charge_shot.beam_charge_counter()
  return ram.u16(charge_shot.ADDR_BEAM_CHARGE)
end

function charge_shot.session_beam_charge(session)
  return charge_shot.beam_charge_counter()
end

function charge_shot.in_shot_seat(samus_x, samus_y, target_x, target_y, opts)
  opts = opts or {}
  local fire_range_px = opts.fire_range_px or charge_shot.FIRE_RANGE_PX
  local clamp_slack = opts.clamp_slack or 8
  local sx, tx = tonumber(samus_x) or 0, tonumber(target_x) or 0
  local dx = tx - sx
  if dx < 0 then
    dx = -dx
  end
  if dx <= fire_range_px then
    return true
  end
  if opts.approach_x_min ~= nil and tx < sx and sx <= opts.approach_x_min + clamp_slack then
    return true
  end
  if opts.approach_x_max ~= nil and tx > sx and sx >= opts.approach_x_max - clamp_slack then
    return true
  end
  return false
end

local function unique_names(names)
  local seen, out = {}, {}
  for i = 1, #names do
    local n = names[i]
    if n and not seen[n] then
      seen[n] = true
      out[#out + 1] = n
    end
  end
  return out
end

function charge_shot.aim_shot_buttons(dx, dy, opts)
  opts = opts or {}
  local names = {}
  if opts.include_face then
    if dx < 0 then
      names[#names + 1] = "LEFT"
    elseif dx > 0 then
      names[#names + 1] = "RIGHT"
    end
  end
  local adx = dx
  if adx < 0 then
    adx = -adx
  end
  if dy <= charge_shot.AIM_UP_DY and adx <= charge_shot.UNDER_BLOB_DX then
    names[#names + 1] = "UP"
  elseif dy <= charge_shot.AIM_UP_DY then
    names[#names + 1] = "R"
  elseif dy >= charge_shot.AIM_DOWN_DY then
    names[#names + 1] = "L"
  end
  if opts.jump then
    names[#names + 1] = "A"
  end
  if opts.fire then
    names[#names + 1] = "X"
  end
  return unique_names(names)
end

function charge_shot.position_then_charge_action(samus_x, samus_y, facing, target_x, target_y, opts)
  opts = opts or {}
  local sx, sy = tonumber(samus_x) or 0, tonumber(samus_y) or 0
  local tx, ty = tonumber(target_x) or 0, tonumber(target_y) or 0
  local dx, dy = tx - sx, ty - sy
  local face_left = dx < 0 or (dx == 0 and tonumber(facing) == ram.FACING_LEFT)
  local want_facing = face_left and ram.FACING_LEFT or ram.FACING_RIGHT
  local face_btn = face_left and "LEFT" or "RIGHT"
  local movement_type = opts.movement_type or 0
  local charge = opts.charge or 0
  local velocity_y = opts.velocity_y or 0

  if charge_shot.is_turning(movement_type) or tonumber(facing) ~= want_facing then
    return { face_btn }
  end

  local seated = charge_shot.in_shot_seat(sx, sy, tx, ty, {
    fire_range_px = opts.fire_range_px or charge_shot.FIRE_RANGE_PX,
    approach_x_min = opts.approach_x_min,
    approach_x_max = opts.approach_x_max,
  })
  if not seated then
    local walk = { face_btn, "B" }
    if opts.approach_x_min ~= nil and sx <= opts.approach_x_min and face_left then
      walk = {}
    end
    if opts.approach_x_max ~= nil and sx >= opts.approach_x_max and not face_left then
      walk = {}
    end
    if #walk > 0 then
      walk[#walk + 1] = "X"
      return unique_names(walk)
    end
    seated = true
  end

  local need_jump = dy <= charge_shot.JUMP_SHOT_DY
  local airborne = tonumber(velocity_y) ~= 0
  local jumping = need_jump and (charge >= charge_shot.CHARGE_FULL - charge_shot.JUMP_LEAD or airborne)
  local firing = charge < charge_shot.CHARGE_FULL or (need_jump and not airborne)
  return charge_shot.aim_shot_buttons(dx, dy, {
    jump = jumping,
    fire = firing,
    include_face = false,
  })
end

return charge_shot
