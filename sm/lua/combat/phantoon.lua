-- Phantoon seat/window helpers. Product fight is phantoon_doppler.

local ram = require("ram")
local rooms = require("rooms")
local ctrl = require("wrecked_ship.ctrl")

local M = {}
M.ROOM_PHANTOON = rooms.ROOM_PHANTOON or 0xCD13
M.WEAPON_BEAM, M.WEAPON_MISSILES, M.WEAPON_SUPERS = 0, 1, 2
M.ADDR_WS_BOSS_BITS = 0xD82B
M.PHANTOON_BOSS_BIT = 0x01
M.SEAT_X, M.SEAT_X_MIN, M.SEAT_X_MAX = 32, 16, 56
M.FLOOR_Y_MIN, M.FLOOR_Y_MAX = 160, 210
M.CHARGE_FULL = 60
M.ADDR_BEAM_CHARGE = 0x0CD0
M.ADDR_ENEMY0_ID = 0x0F78
M.ADDR_ENEMY0_ILIST = 0x0F92
M.ADDR_ENEMY0_FUNC = 0x0FB2
M.ADDR_EYE_ILIST = 0x0F92 + 0x40
M.VULNERABLE_SPRITEMAPS = {[0xDEF1]=true,[0xDEE7]=true}
M.FUNC_FIG8_VULN, M.FUNC_FIG8_SWOOP_TRIG = 0xD60D, 0xD65C
M.FUNC_SWOOP_OPAQUE, M.FUNC_RAIN_MAKE_VULN, M.FUNC_RAIN_VULN = 0xD678, 0xD767, 0xD788
M.VULNERABLE_FUNCS = {
  [0xD60D]=true,[0xD65C]=true,[0xD678]=true,[0xD767]=true,[0xD788]=true,
}
M.RAIN_VULN_FUNCS = {[0xD767]=true,[0xD788]=true}
M.RAIN_PHASE_FUNCS = {
  [0xD767]=true,[0xD788]=true,[0xD82A]=true,[0xD73F]=true,[0xD7D5]=true,[0xD7F7]=true,
}
M.HURT_POSES = {[83]=true,[84]=true,[109]=true,[143]=true,[158]=true,[159]=true,[160]=true}
M.IL_EYE_OPEN = {0xCC53, 0xCC7B}
M.IL_EYE_LOOK = {0xCC9D, 0xCCD7}
M.IL_BODY_EYE_HIT = {0xCC4D, 0xCC53}
M.RAIN_FIRE_X_MAX, M.RAIN_FIRE_Y_MIN, M.RAIN_FIRE_Y_MAX = 56, 88, 104

function M.strategy(overrides)
  local s = {
    seat_x = M.SEAT_X, seat_x_min = M.SEAT_X_MIN, seat_x_max = M.SEAT_X_MAX,
    floor_y_min = M.FLOOR_Y_MIN, floor_y_max = M.FLOOR_Y_MAX,
    weapon = M.WEAPON_MISSILES, skip_enemy_x = 155,
    fire_close_x = 16, kite_x_max = 130, max_fight_frames = 20000,
    boss_bit_grace_frames = 1200, window_timeout = 480,
  }
  if type(overrides) == "table" then
    local k, v
    for k, v in pairs(overrides) do
      s[k] = v
    end
  end
  return s
end

local function wram_u16(session, addr)
  if ram.u16 then
    return ram.u16(session, addr)
  end
  return ctrl.u16(ctrl.wram(session), addr)
end

local function in_il(value, span)
  return span[1] <= value and value < span[2]
end

function M.enemy_extra(session)
  local func = wram_u16(session, M.ADDR_ENEMY0_FUNC)
  local body_il = wram_u16(session, M.ADDR_ENEMY0_ILIST)
  local eye_il = wram_u16(session, M.ADDR_EYE_ILIST)
  return {
    func = func, body_il = body_il, eye_il = eye_il,
    charge = wram_u16(session, M.ADDR_BEAM_CHARGE),
    eye_x = wram_u16(session, 0x0F7A + 0x40),
    eye_y = wram_u16(session, 0x0F7E + 0x40),
    func_vuln = M.VULNERABLE_FUNCS[func] == true,
    eye_il_open = in_il(eye_il, M.IL_EYE_OPEN) or in_il(eye_il, M.IL_EYE_LOOK),
    body_eye_hit = in_il(body_il, M.IL_BODY_EYE_HIT),
  }
end

function M.eye_open(state, session)
  if session then
    local extra = M.enemy_extra(session)
    return extra.func_vuln or extra.eye_il_open or extra.body_eye_hit
  end
  return M.VULNERABLE_SPRITEMAPS[ctrl.num(state.enemy0_spritemap)] == true
end

function M.seated(state, strat)
  strat = strat or M.strategy()
  if ctrl.is_morph(state.pose) then
    return false
  end
  local pose = ctrl.num(state.pose)
  if pose == 81 or pose == 82 or pose == 164 then
    return false
  end
  local x, y = ctrl.x(state), ctrl.y(state)
  return strat.seat_x_min <= x and x <= strat.seat_x_max
    and strat.floor_y_min <= y and y <= strat.floor_y_max
end

function M.func_vulnerable(func)
  return M.VULNERABLE_FUNCS[ctrl.num(func)] == true
end

function M.rain_vulnerable(func)
  return M.RAIN_VULN_FUNCS[ctrl.num(func)] == true
end

function M.rain_phase(func)
  return M.RAIN_PHASE_FUNCS[ctrl.num(func)] == true
end

function M.rain_charge_ok(enemy_x, enemy_y, max_x)
  max_x = max_x or M.RAIN_FIRE_X_MAX
  enemy_y = enemy_y or 96
  return 0 < enemy_x and enemy_x <= max_x
    and M.RAIN_FIRE_Y_MIN <= enemy_y and enemy_y <= M.RAIN_FIRE_Y_MAX
end

function M.right_park(enemy_x, skip_x)
  skip_x = skip_x or 155
  return ctrl.num(enemy_x) >= skip_x
end

function M.charge_window_ok(func, enemy_x, enemy_y, skip_x)
  skip_x = skip_x or 155
  enemy_y = enemy_y or 96
  if M.right_park(enemy_x, skip_x) then
    return false
  end
  if M.rain_phase(func) then
    return M.rain_vulnerable(func) and M.rain_charge_ok(enemy_x, enemy_y)
  end
  return 100 <= enemy_x and enemy_x < skip_x
end

function M.phantoon_boss_bit_set(session)
  local bitv = wram_u16(session, M.ADDR_WS_BOSS_BITS)
  -- $D82B is a byte; u16 of D82B still has bit 0.
  if ram.u8 then
    bitv = ram.u8(session, M.ADDR_WS_BOSS_BITS)
  end
  return ctrl.band(ctrl.num(bitv), M.PHANTOON_BOSS_BIT) ~= 0
end

local function dead(session)
  local st = session.state
  return ctrl.num(st.health) == 0 or ctrl.num(st.game_state) == 26 or ctrl.num(st.game_state) == 36
end

function M.go_to_seat(session, strat)
  strat = strat or M.strategy()
  if ctrl.y(session.state) < strat.floor_y_min then
    local i
    for i = 1, 180 do
      if ctrl.y(session.state) >= strat.floor_y_min or dead(session) then
        break
      end
      ctrl.hold(session, 1, {}, "phan_fall_in")
    end
  end
  local i
  for i = 1, 80 do
    if ctrl.num(session.state.pose) == 1 or ctrl.num(session.state.pose) == 2 then
      break
    end
    if ctrl.y(session.state) < strat.floor_y_min then
      break
    end
    ctrl.hold(session, 1, {}, "phan_land")
  end
  if ctrl.is_morph(session.state.pose) then
    pcall(ctrl.unmorph, session)
  end
  if M.seated(session.state, strat) or dead(session) or ctrl.num(session.state.enemy0_hp) == 0 then
    if M.seated(session.state, strat) then
      if ctrl.num(session.state.facing) == 8 then
        -- already right
      else
        ctrl.hold(session, 1, {"RIGHT"}, "phan_face")
      end
      pcall(ctrl.select_weapon, session, strat.weapon)
    end
    return
  end
  for i = 1, 90 do
    local st = session.state
    if dead(session) or M.seated(st, strat) or ctrl.num(st.enemy0_hp) == 0 then
      break
    end
    if ctrl.y(st) < strat.floor_y_min then
      ctrl.hold(session, 1, {}, "phan_fall_in")
    elseif ctrl.is_morph(st.pose) then
      pcall(ctrl.unmorph, session)
    elseif ctrl.x(st) < strat.seat_x_min then
      ctrl.hold(session, 1, {"RIGHT"}, "phan_seat_right")
    elseif ctrl.x(st) > strat.seat_x_max then
      ctrl.hold(session, 1, {"LEFT"}, "phan_seat_left")
    else
      ctrl.hold(session, 1, {}, "phan_seat_idle")
    end
  end
  if not dead(session) and ctrl.num(session.state.enemy0_hp) > 0 then
    if ctrl.num(session.state.facing) ~= 8 then
      ctrl.hold(session, 1, {"RIGHT"}, "phan_face")
    end
    pcall(ctrl.select_weapon, session, strat.weapon)
  end
end

function M.flame_snipe_tap(session, strat)
  strat = strat or M.strategy()
  local st = session.state
  if ctrl.is_morph(st.pose) then
    pcall(ctrl.unmorph, session)
    return
  end
  if ctrl.num(st.selected_item) ~= M.WEAPON_BEAM then
    pcall(ctrl.select_weapon, session, M.WEAPON_BEAM)
    return
  end
  if ctrl.y(st) < strat.floor_y_min then
    ctrl.hold(session, 1, {}, "phan_fall_in")
    return
  end
  if M.HURT_POSES[ctrl.num(st.pose)] then
    ctrl.hold(session, 1, {}, "phan_hurt")
    return
  end
  local pose = ctrl.num(st.pose)
  if pose == 21 or pose == 22 or pose == 25 or pose == 43 or pose == 44 or pose == 81 or pose == 82 then
    ctrl.hold(session, 1, {}, "phan_farm_land")
    return
  end
  if ctrl.x(st) > strat.seat_x_max then
    ctrl.hold(session, 1, {"LEFT"}, "phan_farm_left")
    return
  end
  local names = {((ctrl.num(st.facing) ~= 8) and "RIGHT" or "UP")}
  if (ctrl.num(session.frame) % 8) < 2 then
    names[#names + 1] = "X"
  end
  ctrl.hold(session, 1, names, "phan_farm_snipe")
end

function M.rain_corner_wait(session, strat)
  M.flame_snipe_tap(session, strat)
end

return M
