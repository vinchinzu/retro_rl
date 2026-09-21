-- Main Shaft per-region actions: pit two-hop, grate lip, shaft hops, attic door.

local geom = require("wrecked_ship.ws_main_geometry")
local ceiling = require("wrecked_ship.ws_ceiling_door")
local takeoff = require("takeoff")
local ctrl = require("wrecked_ship.ctrl")

local M = {}
local CHARGE_FULL = 60
local FACING_LEFT, FACING_RIGHT = ctrl.FACING_LEFT, ctrl.FACING_RIGHT

local function spin_jump(dir)
  if takeoff.spin_jump then return takeoff.spin_jump(dir) end
  return {dir, "B", "A"}
end
local function walk_toward_x(x, target, slack)
  if takeoff.walk_toward_x then return takeoff.walk_toward_x(x, target, slack) end
  if x < target - slack then return {"RIGHT"} end
  if x > target + slack then return {"LEFT"} end
  return {}
end

function M.at_take02_departure(x, y, vy)
  return 1221 <= x and x <= 1232 and 1851 <= y and y <= 1862 and math.abs(vy) <= 1
end

function M.pit_exit_action(x, y, pose, facing, movement_type, vy)
  if pose == 137 or pose == 138 then return {} end
  local turning = movement_type == geom.TURNING_MOVEMENT
  local airborne = geom.AIR[pose] or math.abs(vy) > 1
  if geom.at_ws_main_first_jump_land(x, y, pose, vy) then
    if x < geom.FIRST_JUMP_LAND_TARGET_X - 4 then return {"RIGHT"} end
    return {}
  end
  if x <= geom.PIT_EXIT_RIGHT_X then
    if airborne or facing ~= FACING_RIGHT or turning then return {"RIGHT"} end
    return {"RIGHT", "B"}
  end
  if airborne then
    if x >= geom.FIRST_JUMP_TAKEOFF_X[2] and y >= geom.WS_MAIN_STAIR_Y then return {"LEFT"} end
    if x > geom.FIRST_JUMP_LAND_X[2] then return {"LEFT"} end
    if y >= geom.WS_MAIN_STAIR_Y then
      if x < geom.FIRST_JUMP_TAKEOFF_X[1] then
        return (facing == FACING_RIGHT) and {"RIGHT", "A"} or {"A"}
      end
      return {"A"}
    end
    if x < geom.FIRST_JUMP_LAND_TARGET_X then return {"RIGHT", "A"} end
    return {}
  end
  if geom.LIP_SHOT_Y[1] <= y and y <= geom.LIP_SHOT_Y[2] and x < geom.FIRST_JUMP_LAND_X[1] then
    if facing ~= FACING_RIGHT or turning then return {"RIGHT"} end
    return {"A"}
  end
  if geom.SHORT_HOP_X[1] <= x and x <= geom.SHORT_HOP_X[2] and y >= geom.WS_MAIN_FLOOR_Y then
    if facing ~= FACING_LEFT or turning then return {"LEFT"} end
    return {"A"}
  end
  if x > geom.FIRST_JUMP_TAKEOFF_X[2] then return {"LEFT"} end
  if x < geom.FIRST_JUMP_TAKEOFF_X[1] then
    if facing ~= FACING_RIGHT or turning then return {"RIGHT"} end
    return {"RIGHT", "B"}
  end
  if math.abs(x - geom.FIRST_JUMP_TAKEOFF_TARGET_X) > 6 then
    return walk_toward_x(x, geom.FIRST_JUMP_TAKEOFF_TARGET_X, 6)
  end
  if facing ~= FACING_RIGHT or turning then return {"RIGHT"} end
  return {"A"}
end

function M.three_shot_action(x, y, pose, facing, frame, charge, movement_type, vy)
  if pose == 137 or pose == 138 then return {} end
  if ctrl.is_morph(pose) then
    return (y < geom.WS_MAIN_FLOOR_Y) and {"LEFT"} or {"UP"}
  end
  if y >= geom.WS_MAIN_PIT_Y then
    return M.pit_exit_action(x, y, pose, facing, movement_type, vy)
  end
  if movement_type == geom.TURNING_MOVEMENT or facing ~= FACING_LEFT then
    return {"LEFT"}
  end
  if x > geom.THREE_SHOT_X_MAX then return {"LEFT", "B"} end
  if x < geom.THREE_SHOT_X_MIN then return walk_toward_x(x, geom.THREE_SHOT_X_MIN, 6) end
  local phase = frame % 80
  local charged = charge >= CHARGE_FULL or phase >= 62
  if charged then
    if phase < 70 then return {"A"} end
    return {"LEFT", "A"}
  end
  return {"X", "A"}
end

function M.grate_lip_action(pose, lip_hit, facing, x, y, vy, charge)
  if geom.CROUCH[pose] then return {"UP"} end
  if not lip_hit then
    if x < geom.LIP_FIRE_X[1] then
      if facing ~= FACING_LEFT then return {"LEFT"} end
      if charge >= geom.POCKET_RELEASE_CHARGE then return {} end
      return {"X"}
    end
    local walk = walk_toward_x(x, geom.FIRST_JUMP_LAND_TARGET_X, 0)
    if walk and walk[1] then return walk end
    if x > geom.LIP_FIRE_X[2] then return {"LEFT"} end
    if facing ~= FACING_RIGHT then return {"RIGHT"} end
    if charge >= CHARGE_FULL then return {"UP"} end
    return {"UP", "X"}
  end
  if ctrl.is_morph(pose) then return {"LEFT"} end
  if not M.at_take02_departure(x, y, vy) then return {} end
  local hi = 1231
  if x < hi then
    if x == hi - 1 and (pose == 15 or pose == 16) then return {"UP"} end
    return {"UP", "RIGHT"}
  end
  if x > hi then return {"UP", "LEFT"} end
  if 1852 <= y and y <= 1856 and geom.GROUNDED[pose] and math.abs(vy) <= 1 then
    return {"LEFT", "A"}
  end
  return (math.abs(vy) <= 1) and {"UP"} or {}
end

function M.attic_door_action(x, y, pose, frame)
  local names = ceiling.ceiling_door_action(x, y, pose, frame, {
    seat_x = geom.WS_MAIN_ATTIC_DOOR_X, lip_y = 160, shaft_y = 50,
    slack = 12, hold_charge = false, fire_phase = 8, wait_phase = 18,
  })
  if names ~= nil then return names end
  return M.climb_action(x, y, pose, FACING_RIGHT, 0, 0, frame, false, 0, geom.SHAFT)
end

function M.wall_up_shot_action(shot_frame, charge)
  if shot_frame < 5 then return {"UP", "X"} end
  if shot_frame < 11 then return {"UP"} end
  return {"LEFT", "A"}
end

function M.climb_action(x, y, pose, facing, vy, movement_type, frame, lip_hit, charge, region)
  if pose == 137 or pose == 138 then return {} end
  facing = facing or FACING_RIGHT
  vy = vy or 0
  movement_type = movement_type or 0
  frame = frame or 0
  charge = charge or 0
  local turning = movement_type == geom.TURNING_MOVEMENT
  region = region or geom.classify_region_xy(x, y, pose, vy)
  if x >= geom.WS_MAIN_SAVE_X - 16 and y >= geom.SAVE_LEDGE_Y[1] and region ~= geom.GRATE_SEAT then
    return {"LEFT", "B"}
  end
  if x < 1040 then return {"RIGHT", "B"} end
  if region == geom.PIT then
    return M.pit_exit_action(x, y, pose, facing, movement_type, vy)
  end
  if region == geom.GRATE_SEAT then
    return M.grate_lip_action(pose, lip_hit, facing, x, y, vy, charge)
  end
  if region == geom.ATTIC_SEAT then
    return M.attic_door_action(x, y, pose, frame)
  end
  -- Generic shaft: hop toward next upper platform.
  local hops, i, hop = geom.UPPER_SHAFT_HOPS, 1, nil
  for i = 1, #hops do
    if y > hops[i].y - 20 then
      hop = hops[i]
      break
    end
  end
  if hop then
    local lo, hi, side = hop.take0, hop.take1, hop.side
    local want = (side == "LEFT") and FACING_LEFT or FACING_RIGHT
    if lo <= x and x <= hi and facing == want and not turning then
      return spin_jump(side)
    end
    return walk_toward_x(x, math.floor((lo + hi) / 2), 0)
  end
  if x < geom.WS_MAIN_SHAFT_CENTER - 24 then return {"RIGHT", "A"} end
  if x > geom.WS_MAIN_SHAFT_CENTER + 24 then return {"LEFT", "A"} end
  return spin_jump((facing == FACING_LEFT) and "LEFT" or "RIGHT")
end

return M
