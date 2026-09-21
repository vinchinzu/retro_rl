-- Clean Spore Spawn left-ledge fight. No energy/ammo/item/boss/door writes.
-- Python: snes/super_metroid/combat/spore_spawn.py play_spore_spawn_fight.

local ctrl = require("brinstar.ctrl")

local M = {}

local num = ctrl.num
local hold = ctrl.hold
local band = ctrl.band
local brief = ctrl.brief

M.ROOM_SPORE_SPAWN = 0x9DC7
M.WEAPON_MISSILES = 1
M.SEAT_X = 21
M.SEAT_Y = 697
M.LEDGE_Y_MIN = 668
M.LEDGE_Y_MAX = 705
M.FLOOR_Y = 710
M.BOSS_AREA = 1
M.BOSS_BIT = 0x02

M.VULNERABLE_SPRITEMAPS = {
  [0xEE79] = true, [0xEE8B] = true, [0xEE9D] = true, [0xEEAF] = true,
  [0xEEC1] = true, [0xEED3] = true, [0xEEE5] = true, [0xEF3D] = true,
  [0xEF4F] = true, [0xEF61] = true,
}
M.FULLY_OPEN_SPRITEMAPS = {
  [0xEF3D] = true, [0xEF4F] = true, [0xEF61] = true,
}

local HURT = {
  [81] = true, [83] = true, [84] = true, [109] = true,
  [143] = true, [158] = true, [159] = true, [160] = true,
}

local function default_strategy(overrides)
  local s = {
    seat_x = M.SEAT_X,
    seat_x_max = 28,
    missiles_per_window = 2,
    min_missiles_to_fire = 1,
    farm_until = 2,
    jump_hold_frames = 48,
    seat_hop_frames = 8,
    hop_x_min = 62,
    hop_x_max = 78,
    missile_cooldown = 10,
    fire_hold_frames = 2,
    max_fight_frames = 24000,
    farm_sweep_frames = 1600,
    boss_bit_grace_frames = 1200,
    fire_enemy_x_min = 120,
    fire_enemy_x_max = 200,
    fire_enemy_y_min = 540,
    fire_enemy_y_max = 640,
    fire_airborne_y = 660,
    fire_x_slop = 10,
    fire_x_left = 4,
    fire_x_right = 12,
    fire_y_max = 624,
    takeoff_behind = 16,
    fire_x_bias = 6,
    fire_standoff = 32,
    fire_close_x = 64,
    fire_close_y = 70,
  }
  if type(overrides) == "table" then
    local k, v
    for k, v in pairs(overrides) do
      s[k] = v
    end
  end
  return s
end

function M.mouth_open(state)
  return M.VULNERABLE_SPRITEMAPS[num(state.enemy0_spritemap)] == true
end

function M.eye_fully_open(state)
  return M.FULLY_OPEN_SPRITEMAPS[num(state.enemy0_spritemap)] == true
end

function M.on_left_ledge(state)
  local y = num(state.samus_y)
  return M.LEDGE_Y_MIN <= y and y <= M.LEDGE_Y_MAX and num(state.samus_x) <= 80
end

function M.under_eye(state, left, right, slop)
  local dx = num(state.samus_x) - num(state.enemy0_x)
  if slop ~= nil then
    return math.abs(dx) <= slop
  end
  left = left or 4
  right = right or 12
  return -left <= dx and dx <= right
end

function M.in_fire_height(state, below)
  below = below or 30
  local eye_y = num(state.enemy0_y)
  local samus_y = num(state.samus_y)
  if eye_y >= 650 then
    return eye_y <= samus_y and samus_y <= eye_y + 20
  end
  return eye_y <= samus_y and samus_y <= eye_y + below
end

local function low_floor_park(state)
  return num(state.enemy0_y) >= 650
end

local function high_right_park(state)
  return num(state.enemy0_x) >= 170 and num(state.enemy0_y) < 595
end

local function fire_target_x(state, bias)
  if high_right_park(state) then
    return 188
  end
  return num(state.enemy0_x) + bias
end

function M.seated(state, strategy)
  strategy = strategy or default_strategy()
  return ctrl.is_morph(state.pose)
    and num(state.samus_x) <= strategy.seat_x_max
    and M.LEDGE_Y_MIN <= num(state.samus_y)
    and num(state.samus_y) <= M.LEDGE_Y_MAX
end

function M.fight_spore_spawn_action(state, frame_index, strategy)
  strategy = default_strategy(strategy)
  if num(state.enemy0_hp) == 0 then
    return {}
  end
  if M.seated(state, strategy) then
    if M.mouth_open(state)
        and num(state.missiles) >= strategy.min_missiles_to_fire
        and num(state.enemy0_x) >= strategy.fire_enemy_x_min then
      return {"UP"}
    end
    return {}
  end
  if M.on_left_ledge(state) and num(state.samus_x) > strategy.seat_x_max then
    return {"LEFT"}
  end
  if num(state.samus_y) >= M.FLOOR_Y then
    if num(state.samus_x) > 80 then
      return {"LEFT", "B"}
    end
    return {"LEFT", "A"}
  end
  if M.mouth_open(state) and num(state.missiles) >= 1 and not ctrl.is_morph(state.pose) then
    local names = {}
    if num(state.samus_y) > strategy.fire_airborne_y then
      names[#names + 1] = "A"
    end
    if num(state.enemy0_y) + 20 < num(state.samus_y) then
      names[#names + 1] = "UP"
    end
    local period = strategy.missile_cooldown + strategy.fire_hold_frames
    if (frame_index % period) < strategy.fire_hold_frames then
      names[#names + 1] = "X"
    end
    return names
  end
  return {}
end

local function dead(session)
  return ctrl.is_dead(session.state)
end

local function boss_defeated(state)
  return band(ctrl.area_boss_bits(state, M.BOSS_AREA), M.BOSS_BIT) ~= 0
end

local function morph_in_corner(session, strategy)
  local i
  for i = 1, 40 do
    local st = session.state
    if dead(session) or M.seated(st, strategy) then
      return
    end
    if not M.on_left_ledge(st) then
      return
    end
    if num(st.samus_x) <= strategy.seat_x_max then
      break
    end
    hold(session, 1, {"LEFT"}, "spore_ledge_left")
  end
  if dead(session) or M.seated(session.state, strategy) then
    return
  end
  if M.on_left_ledge(session.state) and not ctrl.is_morph(session.state.pose) then
    local ok = pcall(ctrl.ensure_morph, session)
    if not ok then
      return
    end
  end
end

local function go_to_seat(session, strategy)
  if num(session.state.samus_y) < 500 then
    local i
    for i = 1, 180 do
      if num(session.state.samus_y) >= M.LEDGE_Y_MIN or dead(session) then
        break
      end
      hold(session, 1, {}, "spore_fall_in")
    end
  end
  local s
  for s = 1, 80 do
    local st = session.state
    local y_ok = num(st.samus_y) >= M.LEDGE_Y_MIN
    if y_ok and not HURT[num(st.pose)] then
      break
    end
    hold(session, 1, {}, "spore_land")
  end
  if M.seated(session.state, strategy) or dead(session) or num(session.state.enemy0_hp) == 0 then
    return
  end
  if M.on_left_ledge(session.state) then
    morph_in_corner(session, strategy)
    return
  end

  local i
  for i = 1, 90 do
    local st = session.state
    if dead(session) or M.seated(st, strategy) or num(st.enemy0_hp) == 0 then
      return
    end
    if M.on_left_ledge(st) then
      morph_in_corner(session, strategy)
      return
    end
    if num(st.samus_y) < M.LEDGE_Y_MIN then
      hold(session, 1, {}, "spore_fall_in")
    elseif num(st.samus_y) >= M.FLOOR_Y
        and strategy.hop_x_min <= num(st.samus_x)
        and num(st.samus_x) <= strategy.hop_x_max then
      break
    elseif num(st.samus_x) < strategy.hop_x_min then
      hold(session, 1, {"RIGHT"}, "spore_off_wall")
    else
      hold(session, 1, {"LEFT", "B"}, "spore_floor_left")
    end
  end

  local st = session.state
  if M.seated(st, strategy) or dead(session) or num(st.enemy0_hp) == 0 then
    return
  end
  if M.on_left_ledge(st) then
    morph_in_corner(session, strategy)
    return
  end
  if num(st.samus_y) >= M.FLOOR_Y and 60 <= num(st.samus_x) and num(st.samus_x) <= 80 then
    hold(session, strategy.seat_hop_frames, {"LEFT", "A"}, "spore_ledge_hop")
    local j
    for j = 1, 28 do
      st = session.state
      if dead(session) or M.on_left_ledge(st) then
        break
      end
      if num(st.samus_x) < 50 then
        hold(session, 1, {}, "spore_hop_idle")
      else
        hold(session, 1, {"LEFT"}, "spore_ledge_left")
      end
    end
  end
  if M.on_left_ledge(session.state) then
    morph_in_corner(session, strategy)
  end
end

local function keep_seat(session, strategy)
  local st = session.state
  if M.seated(st, strategy) then
    return
  end
  if ctrl.is_morph(st.pose) and M.on_left_ledge(st) and num(st.samus_x) > strategy.seat_x then
    hold(session, 4, {"LEFT"}, "spore_seat_nudge")
    return
  end
  if M.on_left_ledge(st) then
    if num(st.samus_x) > strategy.seat_x_max then
      hold(session, 4, {"LEFT"}, "spore_seat_nudge")
    end
    if not ctrl.is_morph(session.state.pose) then
      pcall(ctrl.ensure_morph, session)
    end
    return
  end
  go_to_seat(session, strategy)
end

local function fire_window(session, strategy)
  if ctrl.is_morph(session.state.pose) then
    ctrl.unmorph(session)
  end
  if num(session.state.missiles) <= 0 then
    return 0
  end
  if not ctrl.select_weapon(session, M.WEAPON_MISSILES) then
    return 0
  end

  local left = strategy.fire_x_left
  local right = strategy.fire_x_right
  local bias = strategy.fire_x_bias
  local i
  for i = 1, 100 do
    local st = session.state
    if dead(session) or num(st.enemy0_hp) == 0 then
      return 0
    end
    if num(st.samus_x) < 55 then
      hold(session, 1, {"RIGHT"}, "spore_off_wall")
    else
      local target = fire_target_x(st, bias)
      local close
      if high_right_park(st) then
        close = 40
      elseif low_floor_park(st) then
        close = 12
      else
        close = 18
      end
      if num(st.samus_y) >= M.FLOOR_Y and math.abs(num(st.samus_x) - target) <= close then
        break
      end
      local face
      if num(st.samus_x) < target then
        face = "RIGHT"
      else
        face = "LEFT"
      end
      if num(st.samus_y) >= M.FLOOR_Y then
        local dash = high_right_park(st) or math.abs(num(st.samus_x) - target) > 50
        if dash then
          hold(session, 1, {face, "B"}, "spore_close")
        else
          hold(session, 1, {face}, "spore_close")
        end
      else
        hold(session, 1, {face}, "spore_close")
      end
    end
  end

  local shots = 0
  local jump_hold = 0
  local last_spend = -99
  local index
  for index = 1, 220 do
    local st = session.state
    if num(st.enemy0_hp) == 0 or dead(session) then
      break
    end
    local on_floor = num(st.samus_y) >= M.FLOOR_Y
    if num(st.samus_x) < 55 then
      jump_hold = 0
      hold(session, 1, {"RIGHT"}, "spore_off_wall")
    else
      local aim = num(st.enemy0_x)
      local dx = num(st.samus_x) - aim
      local jump_dir = ""
      if dx < -left then
        jump_dir = "RIGHT"
      elseif dx > right then
        jump_dir = "LEFT"
      end
      if on_floor and jump_hold == 0 then
        if low_floor_park(st) then
          if M.mouth_open(st) then
            jump_hold = 12
          else
            jump_hold = 0
          end
        elseif M.mouth_open(st) then
          jump_hold = 52
        else
          jump_hold = 36
        end
      end
      local hold_jump = jump_hold > 0
      if not on_floor and num(st.samus_y) <= num(st.enemy0_y) + 12 then
        hold_jump = false
      end
      jump_hold = math.max(0, jump_hold - 1)
      local fire = M.under_eye(st, left, right)
        and M.in_fire_height(st)
        and M.mouth_open(st)
        and not HURT[num(st.pose)]
        and num(st.missiles) > 0
        and shots < strategy.missiles_per_window
      local tap_x = fire and (num(session.frame) - last_spend) >= 10
      local names_list
      if fire and tap_x then
        names_list = {"UP", "A", "X"}
      elseif fire then
        names_list = {"UP", "A"}
      elseif on_floor then
        if low_floor_park(st) then
          names_list = {jump_dir ~= "" and jump_dir or "RIGHT"}
          if hold_jump then
            names_list[#names_list + 1] = "A"
          end
        else
          names_list = {jump_dir ~= "" and jump_dir or "RIGHT", "A"}
          if hold_jump then
            names_list[#names_list + 1] = "B"
          end
        end
      else
        names_list = {"UP"}
        if jump_dir ~= "" then
          table.insert(names_list, 1, jump_dir)
        end
        if hold_jump then
          names_list[#names_list + 1] = "A"
          if not low_floor_park(st) then
            names_list[#names_list + 1] = "B"
          end
        end
      end
      local ms_before = num(st.missiles)
      hold(session, 1, ctrl.unique_names(names_list), "spore_eye_shot")
      if num(session.state.missiles) < ms_before then
        shots = shots + 1
        last_spend = num(session.frame)
      end
      if shots >= strategy.missiles_per_window then
        break
      end
    end
  end

  if not dead(session) and num(session.state.enemy0_hp) > 0 then
    local k
    for k = 1, 24 do
      if not HURT[num(session.state.pose)] then
        break
      end
      hold(session, 1, {}, "spore_land")
    end
    go_to_seat(session, strategy)
    if not M.seated(session.state, strategy) and not dead(session) then
      go_to_seat(session, strategy)
    end
  end
  return shots
end

local function walk_toward_x(session, target_x, frames)
  local i
  for i = 1, frames do
    local st = session.state
    if math.abs(num(st.samus_x) - target_x) <= 4 then
      return
    end
    local face
    if num(st.samus_x) < target_x then
      face = "RIGHT"
    else
      face = "LEFT"
    end
    hold(session, 1, {face}, "spore_farm_walk")
  end
end

local function farm_drops(session, strategy)
  local start = num(session.frame)
  local want = math.min(strategy.farm_until, num(session.state.max_missiles))
  if want <= 0 then
    want = strategy.farm_until
  end
  if ctrl.is_morph(session.state.pose) then
    ctrl.unmorph(session)
  end
  ctrl.try_select(session, 0)
  local walk_right = true
  local index
  for index = 0, strategy.farm_sweep_frames - 1 do
    local st = session.state
    if dead(session) or num(st.enemy0_hp) == 0 or num(st.missiles) >= want then
      break
    end
    if num(st.samus_y) < 500 then
      hold(session, 1, {}, "spore_farm_fall")
    else
      if num(st.samus_x) >= 200 then
        walk_right = false
      elseif num(st.samus_x) <= 50 then
        walk_right = true
      end
      local face
      if walk_right then
        face = "RIGHT"
      else
        face = "LEFT"
      end
      local names = {face, "UP"}
      if (index % 40) < 18 then
        names[#names + 1] = "A"
      end
      if (index % 8) < 2 then
        names[#names + 1] = "X"
      end
      hold(session, 1, names, "spore_farm_shoot")
    end
  end
  if num(session.state.missiles) > 0 then
    ctrl.select_weapon(session, M.WEAPON_MISSILES)
  end
  go_to_seat(session, strategy)
  return num(session.frame) - start
end

function M.play_spore_spawn_fight(session, strategy, require_boss_bit)
  if require_boss_bit == nil then
    require_boss_bit = true
  end
  strategy = default_strategy(strategy)
  local start = num(session.frame)
  if num(session.state.room_id) ~= M.ROOM_SPORE_SPAWN then
    error(string.format(
      "Spore Spawn fight expected room 0x%04X, got 0x%04X",
      M.ROOM_SPORE_SPAWN,
      num(session.state.room_id)
    ))
  end

  if num(session.state.missiles) > 0 then
    ctrl.select_weapon(session, M.WEAPON_MISSILES)
  end
  go_to_seat(session, strategy)

  local peak_hp = num(session.state.enemy0_hp)
  local min_hp = peak_hp
  local activation_seen = peak_hp >= 900 or M.mouth_open(session.state)
  local defeat_frame = nil
  if peak_hp == 0 then
    defeat_frame = start
  end
  local boss_bit_frame = nil
  local shots_fired = 0
  local farm_frames = 0
  local windows = 0
  local seen = {}
  local prev_hp = num(session.state.enemy0_hp)

  local _
  for _ = 1, strategy.max_fight_frames do
    local state = session.state
    if num(state.room_id) ~= M.ROOM_SPORE_SPAWN then
      break
    end
    if dead(session) then
      break
    end
    peak_hp = math.max(peak_hp, num(state.enemy0_hp))
    min_hp = math.min(min_hp, num(state.enemy0_hp))
    if M.mouth_open(state) then
      seen[num(state.enemy0_spritemap)] = true
      activation_seen = true
    end

    if defeat_frame == nil and num(state.enemy0_hp) == 0 and prev_hp > 0 then
      defeat_frame = num(session.frame)
      min_hp = 0
    end
    prev_hp = num(state.enemy0_hp)

    if defeat_frame ~= nil then
      if boss_defeated(session.state) then
        boss_bit_frame = num(session.frame)
        break
      end
      if not require_boss_bit then
        break
      end
      if num(session.frame) - defeat_frame >= strategy.boss_bit_grace_frames then
        break
      end
      hold(session, 1, {}, "spore_death_anim")
    elseif num(state.missiles) < strategy.min_missiles_to_fire then
      farm_frames = farm_frames + farm_drops(session, strategy)
    else
      local ready = M.mouth_open(state)
        and num(state.enemy0_x) >= 120
        and num(state.missiles) >= strategy.min_missiles_to_fire
        and (M.seated(state, strategy) or M.on_left_ledge(state))
      if ready then
        windows = windows + 1
        shots_fired = shots_fired + fire_window(session, strategy)
      elseif not M.seated(state, strategy) then
        keep_seat(session, strategy)
      else
        hold(session, 1, {}, "spore_wait_eye")
      end
    end
  end

  local outcome
  if dead(session) then
    outcome = "died"
  elseif defeat_frame ~= nil and (not require_boss_bit or boss_bit_frame ~= nil) then
    outcome = "spore_spawn_defeated"
  elseif defeat_frame ~= nil then
    outcome = "boss_bit_timeout"
  else
    outcome = "timeout"
  end

  local maps = {}
  local k
  for k in pairs(seen) do
    maps[#maps + 1] = k
  end
  table.sort(maps)

  return {
    start_frame = start,
    activation_seen = activation_seen,
    defeat_frame = defeat_frame,
    boss_bit_frame = boss_bit_frame,
    end_frame = num(session.frame),
    peak_hp = peak_hp,
    min_enemy_hp = min_hp,
    action_frames = num(session.frame) - start,
    final_enemy_hp = num(session.state.enemy0_hp),
    shots_fired = shots_fired,
    farm_frames = farm_frames,
    windows = windows,
    outcome = outcome,
    vulnerable_spritemaps = maps,
  }
end

M.strategy = default_strategy

return M
