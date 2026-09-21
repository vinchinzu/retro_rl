-- Ice → Gravity product spine (K5–K6). No DoorEdge tables.

local rooms = require("rooms")
local ctrl = require("wrecked_ship.ctrl")
local red = require("red_tower")
local alpha = require("wrecked_ship.alpha_pb_escape")
local moat = require("wrecked_ship.moat")
local west = require("wrecked_ship.west_ocean")
local entrance = require("wrecked_ship.ws_entrance")
local main = require("wrecked_ship.ws_main")
local basement = require("wrecked_ship.ws_basement")
local fight = require("wrecked_ship.phantoon_fight")
local leave = require("wrecked_ship.phantoon_leave")
local ret = require("wrecked_ship.ws_basement_return")
local climb = require("wrecked_ship.ws_main_climb")
local grav = require("wrecked_ship.gravity_collect")

local M = {}
M.ROOM_WS_BASEMENT = rooms.ROOM_WS_BASEMENT or 0xCC6F
M.ROOM_GRAVITY = rooms.ROOM_GRAVITY or 0xCE40
M.GRAVITY_MASK = ctrl.GRAVITY_MASK

local function call_ice_return(session)
  local ok, ice = pcall(require, "ice.spine")
  if ok and ice then
    if ice.play_ice_return then
      ice.play_ice_return(session)
      return
    end
    if ice.play_ice_to_warehouse then
      ice.play_ice_to_warehouse(session)
      return
    end
  end
  ok, ice = pcall(require, "norfair.spine")
  if ok and ice and ice.play_ice_return then
    ice.play_ice_return(session)
  end
end

function M.play_ice_to_phantoon_leave(session)
  call_ice_return(session)
  local ok_wh, wh = pcall(require, "kraid.return_hijump")
  if ok_wh and wh and wh.play_business_to_warehouse then
    wh.play_business_to_warehouse(session)
  else
    local ok_nf, nf = pcall(require, "norfair.business_climb")
    if ok_nf and nf and nf.play_business_to_warehouse then
      nf.play_business_to_warehouse(session)
    end
  end
  red.play_warehouse_to_east(session)
  red.play_east_to_glass(session)
  red.play_glass_to_west(session)
  red.play_west_to_below(session)
  red.play_below_to_bat(session)
  red.play_bat_to_red(session)
  red.play_red_to_hellway(session)
  red.play_hellway_to_caterpillar(session)
  red.play_caterpillar_to_alpha_pb(session)
  if ctrl.num(session.state.max_power_bombs) < 5 then
    ctrl.timeout("Alpha PB not collected: max_power_bombs=" .. tostring(session.state.max_power_bombs))
  end
  alpha.play_alpha_pb_to_caterpillar(session)
  alpha.play_caterpillar_to_elevator(session)
  alpha.play_elevator_to_kihunter(session)
  alpha.play_kihunter_to_moat(session)
  moat.play_moat_cross(session)
  west.play_west_ocean_over_ocean_spark(session)
  entrance.play_ws_entrance_to_main(session)
  main.play_ws_main_to_basement(session)
  basement.play_ws_basement_to_phantoon(session)
  fight.play_phantoon_room_fight(session)
  leave.play_phantoon_loot_exit(session)
  leave.require_phantoon_left(session)
  local st = session.state
  if ctrl.num(st.room_id) ~= M.ROOM_WS_BASEMENT or ctrl.num(st.game_state) ~= 8 then
    ctrl.timeout("phantoon leave: expected WS Basement 0xCC6F gs=8, got " .. ctrl.brief(st))
  end
  return st
end

function M.play_phantoon_to_gravity(session)
  ret.play_ws_basement_to_main(session)
  climb.play_ws_main_to_attic(session)
  grav.play_attic_to_west_ocean(session)
  grav.play_west_ocean_to_pancakes(session)
  grav.play_pancakes_to_homing_geemer(session)
  grav.play_homing_geemer_to_bowling(session)
  grav.play_bowling_to_gravity(session)
  return session.state
end

function M.play_gravity_collect(session)
  grav.play_gravity_collect(session)
  grav.require_gravity_collected(session)
  local st = session.state
  if ctrl.num(st.room_id) ~= M.ROOM_GRAVITY or ctrl.num(st.game_state) ~= 8 then
    ctrl.timeout("gravity collect: expected 0xCE40 gs=8, got " .. ctrl.brief(st))
  end
  if ctrl.band(ctrl.num(st.collected_items), M.GRAVITY_MASK) == 0 then
    ctrl.timeout(string.format("gravity collect: items 0x%04X missing GRAVITY_MASK", ctrl.num(st.collected_items)))
  end
  return st
end

function M.play(session)
  M.play_ice_to_phantoon_leave(session)
  M.play_phantoon_to_gravity(session)
  return M.play_gravity_collect(session)
end

return M
