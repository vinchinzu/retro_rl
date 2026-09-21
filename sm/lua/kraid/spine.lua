-- Red Tower → Varia → Business composer (K2–K3).
-- play_to_business(session) starts in Red Tower 0xA253 and leaves Business 0xA7DE.

local geo = require("red_tower.ctrl")
local red = require("red_tower.red_stack")
local hj = require("kraid.collect_hijump")
local ret = require("kraid.return_hijump")
local to = require("kraid.to_kraid")
local from = require("kraid.from_kraid")
local varia = require("kraid.varia_return")
local wh = require("kraid.warehouse_stack")

local M = {}

M.ROOM_RED_TOWER = geo.ROOM_RED_TOWER
M.ROOM_BUSINESS = geo.ROOM_BUSINESS -- 0xA7DE

-- Ordered hops matching POST_SUPERS_SPINE bat → business (K2–K3).
M.HOPS = {
  {id = "red_tower_to_bat", play = red.play_red_tower_to_bat, to = geo.ROOM_BAT},
  {id = "bat_to_below_spazer", play = red.play_bat_to_below_spazer, to = geo.ROOM_BELOW_SPAZER},
  {id = "below_spazer_to_west", play = red.play_below_spazer_to_west, to = geo.ROOM_WEST_TUNNEL},
  {id = "west_to_glass", play = red.play_west_to_glass, to = geo.ROOM_GLASS},
  {id = "glass_to_east", play = red.play_glass_to_east, to = geo.ROOM_EAST_TUNNEL},
  {id = "east_to_warehouse", play = red.play_east_to_warehouse, to = geo.ROOM_WAREHOUSE},
  {id = "warehouse_to_business", play = wh.play_warehouse_to_business, to = geo.ROOM_BUSINESS},
  {id = "business_to_hj_shaft", play = hj.play_business_to_hj_shaft, to = geo.ROOM_HJ_SHAFT},
  {id = "hj_shaft_to_hj_room", play = hj.play_hj_shaft_to_hj_room, to = geo.ROOM_HJ},
  {id = "hijump_collected", play = hj.play_hj_room_collect, to = geo.ROOM_HJ},
  {id = "hj_room_to_shaft", play = ret.play_hj_room_to_shaft, to = geo.ROOM_HJ_SHAFT},
  {id = "hj_shaft_to_business", play = ret.play_hj_shaft_to_business, to = geo.ROOM_BUSINESS},
  {id = "business_to_warehouse", play = ret.play_business_to_warehouse, to = geo.ROOM_WAREHOUSE},
  {id = "warehouse_to_zeela", play = to.play_warehouse_to_zeela_with_hijump, to = geo.ROOM_ZEELA},
  {id = "zeela_to_kihunter", play = to.play_zeela_to_kihunter, to = geo.ROOM_WAREHOUSE_KIHUNTER},
  {id = "kihunter_to_baby_kraid", play = to.play_kihunter_to_baby_kraid, to = geo.ROOM_BABY_KRAID},
  {id = "baby_kraid_to_eye", play = to.play_baby_kraid_to_eye, to = geo.ROOM_KRAID_EYE},
  {id = "eye_to_kraid", play = to.play_eye_to_kraid, to = geo.ROOM_KRAID},
  {id = "kraid_to_varia", play = to.play_kraid_entry_to_varia, to = geo.ROOM_VARIA},
  {id = "varia_to_kraid_return", play = varia.play_varia_to_kraid, to = geo.ROOM_KRAID},
  {id = "kraid_to_eye_return", play = varia.play_kraid_to_eye_return, to = geo.ROOM_KRAID_EYE},
  {id = "eye_to_baby_return", play = from.play_eye_to_baby_return, to = geo.ROOM_BABY_KRAID},
  {id = "baby_to_kihunter_return", play = from.play_baby_to_kihunter_return, to = geo.ROOM_WAREHOUSE_KIHUNTER},
  {id = "kihunter_to_zeela_return", play = from.play_kihunter_to_zeela_return, to = geo.ROOM_ZEELA},
  {id = "zeela_to_warehouse_return", play = from.play_zeela_to_warehouse_return, to = geo.ROOM_WAREHOUSE},
  {id = "warehouse_to_business_return", play = wh.play_warehouse_to_business, to = geo.ROOM_BUSINESS},
}

function M.play_to_business(session)
  geo.require_room(session, geo.ROOM_RED_TOWER, "play_to_business")
  local state = session.state
  for i = 1, #M.HOPS do
    local hop = M.HOPS[i]
    state = hop.play(session)
  end
  if session.state.room_id ~= geo.ROOM_BUSINESS then
    error("play_to_business: expected leave room "
      .. geo.fmt_hex(geo.ROOM_BUSINESS) .. ", got "
      .. geo.fmt_hex(session.state.room_id or 0) .. " " .. geo.fmt_state(session.state))
  end
  if session.state.game_state ~= geo.GS_ORDINARY then
    error("play_to_business: Business not ordinary gs=8: " .. geo.fmt_state(session.state))
  end
  if not geo.has_varia(session.state) then
    error("play_to_business: Varia not collected: " .. geo.fmt_state(session.state))
  end
  if not geo.has_hi_jump(session.state) then
    error("play_to_business: Hi-Jump not collected: " .. geo.fmt_state(session.state))
  end
  return state
end

return M
