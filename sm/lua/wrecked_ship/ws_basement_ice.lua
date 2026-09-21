-- Powered Basement Ice keepaway (Atomics) and Workrobot avoid.

local enemies_mod = require("enemies")

local M = {}
M.ATOMIC_ID = (enemies_mod.ATOMIC_ID) or 0xE9FF
M.COVERN_ID = (enemies_mod.COVERN_ID) or 0xEA3F
M.WORKROBOT_ID = (enemies_mod.WORKROBOT_ID) or 0xE8FF
M.BASEMENT_ICE_X = {400, 1100}
M.BASEMENT_ICE_RANGE = 400
M.BASEMENT_TAKEOFF_X = 720

function M.basement_overlay_targets(samus_x, samus_y, enemies)
  local out, i = {}, 1
  local lo, hi = M.BASEMENT_ICE_X[1], M.BASEMENT_ICE_X[2]
  for i = 1, #(enemies or {}) do
    local e = enemies[i]
    local hp = tonumber(e.hp) or 0
    local x, y = tonumber(e.x) or 0, tonumber(e.y) or 0
    if hp > 0 and lo <= x and x <= hi then
      local dx, dy = x - samus_x, y - samus_y
      if math.sqrt(dx * dx + dy * dy) <= M.BASEMENT_ICE_RANGE then
        out[#out + 1] = e
      end
    end
  end
  return out
end

function M.ice_keepaway_action(samus_x, samus_y, facing, enemies, opts)
  opts = opts or {}
  if not enemies_mod.choose then
    return nil
  end
  local choice = enemies_mod.choose(
    samus_x, samus_y, facing,
    M.basement_overlay_targets(samus_x, samus_y, enemies),
    {engage = {[M.ATOMIC_ID]=true}, absorb = {[M.COVERN_ID]=true}, ignore = {[M.WORKROBOT_ID]=true}},
    opts
  )
  return choice and choice.buttons
end

function M.workrobot_avoid_action(samus_x, samus_y, enemies, takeoff_x_min)
  if not enemies_mod.choose then
    return nil
  end
  local choice = enemies_mod.choose(
    samus_x, samus_y, 0, enemies,
    {avoid = {[M.WORKROBOT_ID]=true}, ignore = {[M.ATOMIC_ID]=true,[M.COVERN_ID]=true}},
    {takeoff_x_min = takeoff_x_min or M.BASEMENT_TAKEOFF_X}
  )
  return choice and choice.buttons
end

return M
