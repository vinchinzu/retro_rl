-- Wiki missile-doppler Phantoon (PRKD recipe on KPDR inventory). 2-2-N, Super if HP ≤ 600.

local rooms = require("rooms")
local ctrl = require("wrecked_ship.ctrl")
local phan = require("combat.phantoon")

local M = {}
M.PAIR_SIZE = 2
M.MAX_DOPPLER_EXTRA = 6
M.PAIR_WAIT_FRAMES = 45
M.GAP_FRAMES = 40
M.DOPPLER_SPACING = 10
M.SUPER_KILL_HP = 600
M.KITE_X_MAX = 130
M.FIRE_CLOSE_X = 32

local function dead(session)
  local st = session.state
  return ctrl.num(st.health) == 0 or ctrl.num(st.game_state) == 26 or ctrl.num(st.game_state) == 36
end

function M.missile_spacing_ok(frames_since_fire, spacing)
  return ctrl.num(frames_since_fire) >= (spacing or M.DOPPLER_SPACING)
end

function M.should_fire_missile(missiles, hittable, frames_since_fire, spacing)
  return ctrl.num(missiles) > 0 and hittable and M.missile_spacing_ok(frames_since_fire, spacing)
end

function M.should_fire_super(hp, supers, kill_hp)
  kill_hp = kill_hp or M.SUPER_KILL_HP
  return ctrl.num(supers) > 0 and 0 < ctrl.num(hp) and ctrl.num(hp) <= kill_hp
end

function M.barrage_phase(pair1, pair2, extra, wait_frames, gap_frames)
  if pair1 < M.PAIR_SIZE then
    return "pair1"
  end
  if wait_frames < M.PAIR_WAIT_FRAMES then
    return "wait1"
  end
  if pair2 < M.PAIR_SIZE then
    return "pair2"
  end
  if gap_frames < M.GAP_FRAMES then
    return "gap"
  end
  if extra >= M.MAX_DOPPLER_EXTRA then
    return "done"
  end
  return "doppler"
end

local function hittable(session)
  local extra = phan.enemy_extra(session)
  return extra.func_vuln or extra.eye_il_open or extra.body_eye_hit
end

local function body_func(session)
  return phan.enemy_extra(session).func
end

local function face(state)
  if ctrl.num(state.enemy0_x) >= ctrl.x(state) then
    return "RIGHT"
  end
  return "LEFT"
end

local function close_enough(state, strategy)
  return math.abs(ctrl.x(state) - ctrl.num(state.enemy0_x)) <= strategy.fire_close_x
end

local function fire_names(state)
  if ctrl.y(state) > ctrl.num(state.enemy0_y) + 10 then
    return {"UP", "X"}
  end
  return {"X"}
end

local function chase_names(state, strategy)
  local x, ex = ctrl.x(state), ctrl.num(state.enemy0_x)
  local dx = ex - x
  if ex <= 56 then
    if x > strategy.seat.seat_x_max then
      return {"LEFT"}
    elseif ctrl.num(state.facing) ~= 8 then
      return {"RIGHT"}
    end
    local names = {}
    if ctrl.y(state) - ctrl.num(state.enemy0_y) > 40 and not phan.HURT_POSES[ctrl.num(state.pose)] then
      names[#names + 1] = "A"
    end
    if ctrl.y(state) > ctrl.num(state.enemy0_y) + 10 then
      names[#names + 1] = "UP"
    end
    return names
  end
  if x >= strategy.kite_x_max and dx > 0 then
    return {"LEFT"}
  end
  if math.abs(dx) > strategy.fire_close_x then
    local names = {(dx > 0) and "RIGHT" or "LEFT"}
    if ctrl.y(state) >= strategy.seat.floor_y_min - 4 then
      names[#names + 1] = "B"
    end
    return names
  end
  local names = {face(state)}
  if ctrl.y(state) - ctrl.num(state.enemy0_y) > 40 and not phan.HURT_POSES[ctrl.num(state.pose)] then
    names[#names + 1] = "A"
  end
  if ctrl.y(state) > ctrl.num(state.enemy0_y) + 10 then
    names[#names + 1] = "UP"
  end
  return names
end

local function unique_names(names)
  local seen, out, i = {}, {}, 1
  for i = 1, #names do
    local n = names[i]
    if n and not seen[n] then
      seen[n] = true
      out[#out + 1] = n
    end
  end
  return out
end

local function park_ok(session, park_x, strategy)
  return phan.charge_window_ok(body_func(session), park_x, ctrl.num(session.state.enemy0_y), strategy.seat.skip_enemy_x)
end

local function skip_park(session, park_x, strategy)
  local skip = strategy.seat.skip_enemy_x
  return not park_ok(session, park_x, strategy) and (
    phan.rain_phase(body_func(session)) or phan.right_park(park_x, skip) or (0 < park_x and park_x < 100)
  )
end

local function window_ready(session, park_x, strategy)
  local st = session.state
  return hittable(session) and ctrl.y(st) >= strategy.seat.floor_y_min and park_ok(session, park_x, strategy)
end

local function select_w(session, weapon)
  if ctrl.num(session.state.selected_item) ~= weapon then
    pcall(ctrl.select_weapon, session, weapon)
  end
end

local function retreat(session, strategy)
  if dead(session) or ctrl.num(session.state.enemy0_hp) <= 0 then
    return
  end
  local i
  for i = 1, 24 do
    if not phan.HURT_POSES[ctrl.num(session.state.pose)] then
      break
    end
    ctrl.hold(session, 1, {}, "phan_doppler_land")
  end
  phan.go_to_seat(session, strategy.seat)
end

local function fire_window(session, strategy)
  if ctrl.is_morph(session.state.pose) then
    pcall(ctrl.unmorph, session)
    local hp = ctrl.num(session.state.enemy0_hp)
    return {missiles_spent = 0, super_spent = 0, hp_drop = 0, halt_miss = true, pre_hp = hp, post_hp = hp}
  end
  local pre_hp = ctrl.num(session.state.enemy0_hp)
  local want_super = M.should_fire_super(pre_hp, ctrl.num(session.state.super_missiles))
  if not want_super then
    if strategy.flame_eat and ctrl.num(session.state.selected_item) == phan.WEAPON_BEAM then
      if not phan.HURT_POSES[ctrl.num(session.state.pose)] and not ctrl.is_morph(session.state.pose) then
        ctrl.hold(session, 2, {"X"}, "phan_doppler_flame_eat")
        ctrl.hold(session, 2, {}, "phan_doppler_flame_eat_release")
      end
    end
  end
  select_w(session, want_super and phan.WEAPON_SUPERS or phan.WEAPON_MISSILES)
  local pair1, pair2, extra, wait_frames, gap_frames = 0, 0, 0, 0, 0
  local missiles_spent, super_spent, last_spend, last_hp = 0, 0, -99, pre_hp
  local pending, halt_miss, seen_open = nil, false, false
  local i
  for i = 1, strategy.window_timeout do
    local st = session.state
    if dead(session) or ctrl.num(st.enemy0_hp) == 0 or ctrl.num(st.health) <= 20 then
      break
    end
    local hit = hittable(session)
    seen_open = seen_open or hit
    if pending ~= nil then
      if ctrl.num(st.enemy0_hp) < last_hp then
        pending, last_hp = nil, ctrl.num(st.enemy0_hp)
      elseif ctrl.num(session.frame) - pending >= 48 then
        halt_miss = true
        break
      end
    end
    local spent = missiles_spent + super_spent
    if (not hit) and seen_open and spent >= 1 and pending == nil then
      break
    end
    if phan.rain_phase(body_func(session)) and not park_ok(session, ctrl.num(st.enemy0_x), strategy) then
      break
    end
    if phan.HURT_POSES[ctrl.num(st.pose)] then
      ctrl.hold(session, 1, {}, "phan_doppler_hurt")
    else
      local use_super = want_super or M.should_fire_super(ctrl.num(st.enemy0_hp), ctrl.num(st.super_missiles))
      local phase = use_super and "super" or M.barrage_phase(pair1, pair2, extra, wait_frames, gap_frames)
      if phase == "done" or (phase == "super" and super_spent >= 1 and pending == nil) then
        break
      end
      if phase == "wait1" or phase == "gap" then
        if phase == "wait1" then wait_frames = wait_frames + 1 else gap_frames = gap_frames + 1 end
        ctrl.hold(session, 1, {}, phase == "wait1" and "phan_doppler_wait" or "phan_doppler_gap")
      else
        local ammo, attr, reason, fire
        if phase == "super" then
          want_super = true
          select_w(session, phan.WEAPON_SUPERS)
          ammo, attr, reason = ctrl.num(st.super_missiles), "super_missiles", "phan_doppler_super"
          fire = hit and ammo > 0 and M.missile_spacing_ok(ctrl.num(session.frame) - last_spend) and close_enough(st, strategy)
        else
          ammo, attr, reason = ctrl.num(st.missiles), "missiles", "phan_doppler_shot"
          fire = M.should_fire_missile(ammo, hit, ctrl.num(session.frame) - last_spend) and close_enough(st, strategy)
        end
        local names = fire and fire_names(st) or (chase_names(st, strategy))
        if not names or not names[1] then
          names = {face(st)}
        end
        ctrl.hold(session, 1, unique_names(names), reason)
        if ctrl.num(session.state[attr]) < ammo then
          last_spend, pending = ctrl.num(session.frame), ctrl.num(session.frame)
          if phase == "super" then
            super_spent = super_spent + 1
          elseif phase == "pair1" then
            pair1, missiles_spent = pair1 + 1, missiles_spent + 1
          elseif phase == "pair2" then
            pair2, missiles_spent = pair2 + 1, missiles_spent + 1
          else
            extra, missiles_spent = extra + 1, missiles_spent + 1
          end
        end
      end
    end
  end
  local post_hp = ctrl.num(session.state.enemy0_hp)
  local hp_drop = math.max(0, pre_hp - post_hp)
  if missiles_spent + super_spent >= 1 and hp_drop <= 0 then
    halt_miss = true
  end
  retreat(session, strategy)
  return {
    missiles_spent = missiles_spent, super_spent = super_spent, hp_drop = hp_drop,
    pair1 = pair1, pair2 = pair2, extra = extra, halt_miss = halt_miss,
    pre_hp = pre_hp, post_hp = post_hp,
  }
end

local function wait_tick(session, park_x, strategy)
  local st = session.state
  if M.should_fire_super(ctrl.num(st.enemy0_hp), ctrl.num(st.super_missiles)) then
    select_w(session, phan.WEAPON_SUPERS)
    ctrl.hold(session, 1, {"DOWN"}, "phan_doppler_wait_super")
  elseif skip_park(session, park_x, strategy) then
    phan.rain_corner_wait(session, strategy.seat)
  elseif not phan.seated(st, strategy.seat) then
    phan.go_to_seat(session, strategy.seat)
  elseif ctrl.is_morph(st.pose) then
    pcall(ctrl.unmorph, session)
  else
    select_w(session, phan.WEAPON_MISSILES)
    ctrl.hold(session, 1, {"DOWN"}, "phan_doppler_wait_eye")
  end
end

function M.play_phantoon_doppler_fight(session, opts)
  opts = opts or {}
  local strategy = {
    seat = phan.strategy(),
    flame_eat = true,
    fire_close_x = M.FIRE_CLOSE_X,
    kite_x_max = M.KITE_X_MAX,
    max_fight_frames = 20000,
    window_timeout = 720,
    boss_bit_grace_frames = 1200,
  }
  local require_boss_bit = opts.require_boss_bit ~= false
  if ctrl.num(session.state.room_id) ~= phan.ROOM_PHANTOON then
    ctrl.timeout(string.format(
      "Phantoon doppler expected room 0x%04X, got 0x%04X",
      phan.ROOM_PHANTOON, ctrl.num(session.state.room_id)
    ))
  end
  local start = ctrl.num(session.frame)
  select_w(session, phan.WEAPON_MISSILES)
  phan.go_to_seat(session, strategy.seat)
  local peak_hp = ctrl.num(session.state.enemy0_hp)
  local min_hp, start_hp, prev_hp = peak_hp, peak_hp, peak_hp
  local body_zero_frame, boss_bit_frame = nil, nil
  if peak_hp == 0 then
    body_zero_frame = start
  end
  local missiles, supers, halt = 0, 0, false
  local park_x, last_func = ctrl.num(session.state.enemy0_x), nil
  local i
  for i = 1, strategy.max_fight_frames do
    local state = session.state
    if ctrl.num(state.room_id) ~= phan.ROOM_PHANTOON or dead(session) then
      break
    end
    local hp = ctrl.num(state.enemy0_hp)
    if hp > peak_hp then peak_hp = hp end
    if hp < min_hp then min_hp = hp end
    if body_zero_frame == nil and hp == 0 and prev_hp > 0 then
      body_zero_frame, min_hp = ctrl.num(session.frame), 0
    end
    prev_hp = hp
    if body_zero_frame ~= nil then
      if phan.phantoon_boss_bit_set(session) then
        boss_bit_frame = ctrl.num(session.frame)
        break
      end
      if not require_boss_bit or ctrl.num(session.frame) - body_zero_frame >= strategy.boss_bit_grace_frames then
        break
      end
      ctrl.hold(session, 1, {}, "phantoon_death_anim")
    else
      local func_now = body_func(session)
      if func_now ~= last_func then
        park_x, last_func = ctrl.num(state.enemy0_x), func_now
      end
      if not window_ready(session, park_x, strategy) then
        wait_tick(session, park_x, strategy)
      else
        local got = fire_window(session, strategy)
        missiles = missiles + got.missiles_spent
        supers = supers + got.super_spent
        if got.halt_miss and got.hp_drop <= 0 then
          halt = true
          break
        end
        if got.missiles_spent + got.super_spent <= 0 then
          ctrl.hold(session, 1, {}, "phan_doppler_wait_eye")
        end
      end
    end
  end
  local boss_set = phan.phantoon_boss_bit_set(session)
  local outcome
  if dead(session) then
    outcome = "died"
  elseif halt then
    outcome = "halt_miss"
  elseif boss_set then
    outcome = "phantoon_defeated"
  elseif body_zero_frame ~= nil then
    outcome = "phantoon_body_zero_no_boss_bit"
  else
    outcome = "timeout"
  end
  return {
    start_frame = start, body_zero_frame = body_zero_frame, boss_bit_frame = boss_bit_frame,
    end_frame = ctrl.num(session.frame), peak_body_hp = peak_hp, min_body_hp = min_hp,
    action_frames = ctrl.num(session.frame) - start,
    final_body_hp = ctrl.num(session.state.enemy0_hp),
    boss_bit_set = boss_set, outcome = outcome,
    missiles_spent = missiles, super_spent = supers,
  }
end

return M
