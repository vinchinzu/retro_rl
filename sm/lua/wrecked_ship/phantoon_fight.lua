-- Phantoon room fight wrapper. Natural 0xCD13 → wiki missile doppler.

local rooms = require("rooms")
local ctrl = require("wrecked_ship.ctrl")
local phan = require("combat.phantoon")

local M = {}
M.ADDR_WS_BOSS_BITS = phan.ADDR_WS_BOSS_BITS
M.PHANTOON_BOSS_BIT = phan.PHANTOON_BOSS_BIT
M.phantoon_boss_bit_set = phan.phantoon_boss_bit_set

function M.require_phantoon_defeated(session)
  if M.phantoon_boss_bit_set(session) then
    return
  end
  ctrl.timeout("phantoon_fight: Wrecked Ship $D82B bit 0 not set: " .. ctrl.brief(session.state))
end

function M.play_phantoon_room_fight(session)
  local doppler = require("combat.phantoon_doppler")
  ctrl.require_room(session, rooms.ROOM_PHANTOON or 0xCD13, "phantoon_room_fight")
  if M.phantoon_boss_bit_set(session) and ctrl.num(session.state.enemy0_hp) == 0 then
    return session.state
  end
  local evidence = doppler.play_phantoon_doppler_fight(session)
  if evidence.outcome ~= "phantoon_defeated" then
    ctrl.timeout("phantoon_room_fight: fight failed (" .. tostring(evidence.outcome) .. "): " .. ctrl.brief(session.state))
  end
  return session.state
end

return M
