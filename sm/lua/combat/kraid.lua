-- Full-knowledge Kraid Super-spray + rear door + Varia PLM.
-- Port of snes/super_metroid/combat/kraid.py. Clean: no energy/ammo writes.

local geo = require("red_tower.ctrl")

local M = {}

M.ROOM_KRAID = geo.ROOM_KRAID
M.ROOM_VARIA = geo.ROOM_VARIA
M.VARIA_MASK = geo.ITEM_VARIA
M.ADDR_BRINSTAR_BOSS_BITS = 0xD829
M.KRAID_BOSS_BIT = 0x01
M.WEAPON_SUPERS = 2
M.WEAPON_BEAM = 0
M.KRAID_MAX_HP = 1000

local DEFAULT_STRATEGY = {
  min_x = 50,
  max_x = 260,
  jump_hold_frames = 10,
  jump_period = 50,
  fire_hold_frames = 6,
  fire_period = 12,
  max_fight_frames = 15000,
  boss_bit_grace_frames = 1200,
}

local function merge_strategy(strategy)
  if not strategy then
    return DEFAULT_STRATEGY
  end
  local s = {}
  for k, v in pairs(DEFAULT_STRATEGY) do
    s[k] = strategy[k] or v
  end
  return s
end

local function read_u8(session, addr)
  local ram_ok, rammod = pcall(require, "ram")
  if ram_ok and type(rammod) == "table" then
    if rammod.u8 then
      local ok, val = pcall(rammod.u8, addr)
      if ok and val ~= nil then
        return val
      end
    end
    if rammod.read_u8 then
      local ok, val = pcall(rammod.read_u8, addr)
      if ok and val ~= nil then
        return val
      end
    end
  end
  local st = session.state
  if st.brinstar_boss_bits ~= nil then
    return st.brinstar_boss_bits
  end
  local bits = st.boss_bits
  if type(bits) == "table" then
    return bits[2] or bits[1] or 0
  end
  return bits or 0
end

function M.brinstar_boss_bits(session)
  return read_u8(session, M.ADDR_BRINSTAR_BOSS_BITS)
end

function M.kraid_defeated(session)
  return geo.band(M.brinstar_boss_bits(session), M.KRAID_BOSS_BIT) ~= 0
end

function M.body_hp(state)
  return tonumber(state.enemy0_hp) or 0
end

local function lane_hold_action(samus_x, min_x, max_x, face, dash)
  if samus_x > 60000 then
    return {}
  end
  if samus_x > max_x then
    if dash then
      return {"LEFT", "B"}
    end
    return {"LEFT"}
  end
  if samus_x < min_x then
    if dash then
      return {"RIGHT", "B"}
    end
    return {"RIGHT"}
  end
  if face and face ~= "" then
    return {face}
  end
  return {}
end

local function spray_action(frame_index, face, fire_period, fire_hold, jump_period, jump_hold, dash)
  local names = {}
  local seen = {}
  local function add(n)
    if n and n ~= "" and not seen[n] then
      seen[n] = true
      names[#names + 1] = n
    end
  end
  add(face)
  if jump_period > 0 and frame_index % jump_period < jump_hold then
    add("A")
  end
  if fire_period > 0 and frame_index % fire_period < fire_hold then
    add("X")
  end
  if dash and not seen["A"] then
    add("B")
  end
  return names
end

function M.fight_kraid_action(state, frame_index, strategy, body_dead)
  strategy = merge_strategy(strategy)
  local x = state.samus_x
  if x > 60000 then
    return {}
  end
  if body_dead then
    if state.pose == 137 or state.pose == 138 then
      if math.floor(frame_index / 30) % 2 == 0 then
        return {"LEFT"}
      end
      return {"RIGHT", "A", "B"}
    end
    return {"RIGHT", "B", "A"}
  end
  if x > strategy.max_x or x < strategy.min_x then
    return lane_hold_action(x, strategy.min_x, strategy.max_x, "RIGHT", true)
  end
  return spray_action(
    frame_index,
    "RIGHT",
    strategy.fire_period,
    strategy.fire_hold_frames,
    strategy.jump_period,
    strategy.jump_hold_frames,
    true
  )
end

local function ensure_weapon(session, weapon)
  if session.state.selected_item == weapon then
    return
  end
  if weapon == 1 and (session.state.max_missiles or 0) <= 0 then
    return
  end
  if weapon == 2 and (session.state.max_super_missiles or 0) <= 0 then
    return
  end
  if weapon == 3 and (session.state.max_power_bombs or 0) <= 0 then
    return
  end
  geo.select_weapon(session, weapon)
end

local function settle_standing(session, min_y, bad_poses, max_frames, reason)
  max_frames = max_frames or 60
  reason = reason or "combat_settle"
  for _ = 1, max_frames do
    local st = session.state
    local y_ok = (min_y == nil) or st.samus_y >= min_y
    if y_ok and not bad_poses[st.pose] then
      return st
    end
    geo.hold(session, 1, {}, reason)
  end
  return session.state
end

function M.play_kraid_fight(session, opts)
  opts = opts or {}
  local strategy = merge_strategy(opts.strategy)
  local require_boss_bit = true
  if opts.require_boss_bit == false then
    require_boss_bit = false
  end
  local start = session.frame or 0
  if session.state.room_id ~= geo.ROOM_KRAID then
    error("Kraid fight expected room " .. geo.fmt_hex(geo.ROOM_KRAID)
      .. ", got " .. geo.fmt_hex(session.state.room_id or 0))
  end
  if (session.state.max_super_missiles or 0) > 0 then
    ensure_weapon(session, M.WEAPON_SUPERS)
  end
  settle_standing(session, 390, geo.set(81, 164), 60, "fight_kraid_land")

  local peak_hp = 0
  local min_hp = M.KRAID_MAX_HP
  local body_zero_frame = nil
  local boss_bit_frame = nil
  local prev_hp = M.body_hp(session.state)
  if prev_hp > 0 and prev_hp <= M.KRAID_MAX_HP then
    peak_hp = prev_hp
    min_hp = prev_hp
  end

  for index = 0, strategy.max_fight_frames - 1 do
    local state = session.state
    if state.room_id ~= geo.ROOM_KRAID then
      break
    end
    local names = M.fight_kraid_action(state, index, strategy, body_zero_frame ~= nil)
    if #names > 0 then
      geo.hold(session, 1, names, "fight_kraid")
    else
      geo.hold(session, 1, {}, "fight_kraid_idle")
    end
    local post = session.state
    local hp = M.body_hp(post)
    if hp >= 0 and hp <= M.KRAID_MAX_HP then
      if hp > peak_hp then
        peak_hp = hp
      end
      if hp < min_hp then
        min_hp = hp
      end
    end
    if body_zero_frame == nil and hp == 0 and prev_hp > 0 then
      body_zero_frame = session.frame
      min_hp = 0
    end
    if boss_bit_frame == nil and M.kraid_defeated(session) then
      boss_bit_frame = session.frame
    end
    if require_boss_bit then
      if boss_bit_frame ~= nil then
        break
      end
      if body_zero_frame ~= nil
          and (session.frame - body_zero_frame) > strategy.boss_bit_grace_frames then
        break
      end
    elseif body_zero_frame ~= nil then
      break
    end
    prev_hp = hp
  end

  local final_hp = M.body_hp(session.state)
  local boss_set = M.kraid_defeated(session)
  local outcome
  if boss_set then
    outcome = "kraid_defeated"
  elseif body_zero_frame ~= nil then
    outcome = "kraid_body_zero_no_boss_bit"
  elseif session.state.room_id ~= geo.ROOM_KRAID then
    outcome = "left_room"
  else
    outcome = "timeout"
  end
  return {
    start_frame = start,
    body_zero_frame = body_zero_frame,
    boss_bit_frame = boss_bit_frame,
    end_frame = session.frame,
    peak_body_hp = peak_hp,
    min_body_hp = min_hp,
    action_frames = (session.frame or 0) - start,
    final_body_hp = final_hp,
    boss_bit_set = boss_set,
    outcome = outcome,
  }
end

function M.play_kraid_rear_exit(session, max_frames)
  max_frames = max_frames or 1200
  if session.state.room_id == geo.ROOM_VARIA then
    return session.state
  end
  if session.state.room_id ~= geo.ROOM_KRAID then
    error("kraid rear exit expected " .. geo.fmt_hex(geo.ROOM_KRAID)
      .. " or " .. geo.fmt_hex(geo.ROOM_VARIA)
      .. ", got " .. geo.fmt_hex(session.state.room_id or 0))
  end
  if not M.kraid_defeated(session) then
    error("kraid rear exit: Brinstar boss bit 0 not set")
  end
  if session.state.selected_item ~= M.WEAPON_BEAM then
    geo.try_select_weapon(session, M.WEAPON_BEAM)
  end

  local left = false
  local air = geo.set(81, 19, 20, 25, 26, 27, 28)
  for index = 0, max_frames - 1 do
    local state = session.state
    if state.room_id == geo.ROOM_VARIA then
      left = true
      break
    end
    if state.room_id ~= geo.ROOM_KRAID then
      break
    end
    if state.samus_y < 360 and air[state.pose] then
      geo.hold(session, 1, {"RIGHT"}, "kraid_rear_fall")
    elseif state.pose == 137 or state.pose == 138 then
      geo.hold(session, 1, {"RIGHT", "B", "A"}, "kraid_rear_knockback")
    elseif state.samus_x < 400 then
      geo.hold(session, 1, {"RIGHT", "B"}, "kraid_rear_run")
    else
      local phase = index % 24
      if phase < 3 then
        geo.hold(session, 1, {"RIGHT"}, "kraid_rear_face")
      elseif phase < 6 then
        geo.hold(session, 1, {"X"}, "kraid_rear_shot")
      elseif phase < 14 then
        geo.hold(session, 1, {"RIGHT", "A", "B"}, "kraid_rear_jump")
      else
        geo.hold(session, 1, {"RIGHT", "B"}, "kraid_rear_push")
      end
    end
  end
  if not left and session.state.room_id ~= geo.ROOM_VARIA then
    error("kraid rear exit: door failed @ frame " .. tostring(session.frame)
      .. ": " .. geo.fmt_state(session.state))
  end
  if session.state.room_id ~= geo.ROOM_VARIA then
    error("kraid rear exit: expected Varia " .. geo.fmt_hex(geo.ROOM_VARIA)
      .. ", got " .. geo.fmt_hex(session.state.room_id or 0))
  end
  for frame = 0, 399 do
    local state = geo.hold(session, 1, {}, "kraid_rear_settle")
    if state.room_id == geo.ROOM_VARIA
        and state.game_state == 8
        and (state.door_transition or 0) == 0
        and frame > 10 then
      break
    end
  end
  for _ = 1, 30 do
    local st = session.state
    if st.samus_y >= 130 and st.pose ~= 81 then
      break
    end
    geo.hold(session, 1, {}, "kraid_rear_land")
  end
  return session.state
end

function M.play_varia_collect(session, max_frames, fanfare_frames)
  max_frames = max_frames or 1200
  fanfare_frames = fanfare_frames or 480
  if session.state.room_id ~= geo.ROOM_VARIA then
    error("varia collect expected room " .. geo.fmt_hex(geo.ROOM_VARIA)
      .. ", got " .. geo.fmt_hex(session.state.room_id or 0))
  end
  if geo.has_varia(session.state) then
    return session.frame
  end
  geo.unmorph(session)
  if session.state.selected_item ~= M.WEAPON_BEAM then
    geo.try_select_weapon(session, M.WEAPON_BEAM)
  end

  local collect_frame = nil
  for index = 0, max_frames - 1 do
    local state = session.state
    if geo.has_varia(state) then
      collect_frame = session.frame
      break
    end
    if state.room_id ~= geo.ROOM_VARIA then
      break
    end
    if state.pose == 137 or state.pose == 138 or state.pose == 9 or state.pose == 10 then
      local m = index % 20
      if m < 8 then
        geo.hold(session, 1, {"UP"}, "varia_recover")
      elseif m < 14 then
        geo.hold(session, 1, {"A"}, "varia_recover")
      else
        local dir = (state.samus_x > 90) and "LEFT" or "RIGHT"
        geo.hold(session, 1, {dir}, "varia_recover")
      end
    else
      local x = state.samus_x
      if x < 45 then
        geo.hold(session, 1, {"RIGHT", "B"}, "varia_approach")
      elseif x > 150 then
        geo.hold(session, 1, {"LEFT", "B"}, "varia_backoff")
      else
        local phase = index % 50
        if phase < 3 then
          geo.hold(session, 1, {"RIGHT"}, "varia_face")
        elseif phase < 5 then
          geo.hold(session, 1, {}, "varia_face_release")
        elseif phase < 8 then
          geo.hold(session, 1, {"X"}, "varia_statue_shot")
        elseif phase < 16 then
          geo.hold(session, 1, {"A"}, "varia_jump")
        elseif phase < 20 then
          geo.hold(session, 1, {"RIGHT", "X"}, "varia_air_shot")
        elseif phase < 38 then
          if x < 112 then
            geo.hold(session, 1, {"RIGHT", "B"}, "varia_touch")
          elseif x > 125 then
            geo.hold(session, 1, {"LEFT"}, "varia_touch")
          else
            geo.hold(session, 1, {"RIGHT"}, "varia_touch")
          end
        else
          geo.hold(session, 1, {}, "varia_wait")
        end
      end
    end
    if geo.has_varia(session.state) then
      collect_frame = session.frame
      break
    end
  end
  if collect_frame == nil or not geo.has_varia(session.state) then
    error("varia collect: PLM not collected @ frame " .. tostring(session.frame)
      .. ": " .. geo.fmt_state(session.state))
  end
  for _ = 1, fanfare_frames do
    geo.hold(session, 1, {}, "varia_fanfare")
  end
  return collect_frame
end

local function varia_evidence(start, varia_room_frame, collect_frame, session, outcome)
  return {
    start_frame = start,
    varia_room_frame = varia_room_frame,
    collect_frame = collect_frame,
    end_frame = session.frame,
    final_items = session.state.collected_items,
    final_room_id = session.state.room_id,
    samus_x = session.state.samus_x,
    samus_y = session.state.samus_y,
    outcome = outcome,
  }
end

function M.play_kraid_to_varia(session, max_exit_frames, max_collect_frames)
  max_exit_frames = max_exit_frames or 1200
  max_collect_frames = max_collect_frames or 1200
  local start = session.frame or 0
  if geo.has_varia(session.state) then
    local room_frame = (session.state.room_id == geo.ROOM_VARIA) and start or nil
    return varia_evidence(start, room_frame, start, session, "varia_collected")
  end
  local varia_room_frame = nil
  if session.state.room_id ~= geo.ROOM_VARIA then
    M.play_kraid_rear_exit(session, max_exit_frames)
  end
  if session.state.room_id == geo.ROOM_VARIA then
    varia_room_frame = session.frame
  end
  local collect_frame = nil
  if session.state.room_id == geo.ROOM_VARIA and not geo.has_varia(session.state) then
    collect_frame = M.play_varia_collect(session, max_collect_frames)
  end
  local outcome
  if geo.has_varia(session.state) then
    outcome = "varia_collected"
  elseif session.state.room_id == geo.ROOM_VARIA then
    outcome = "varia_room_no_item"
  else
    outcome = "no_varia_room"
  end
  return varia_evidence(start, varia_room_frame, collect_frame, session, outcome)
end

function M.play_kraid_fight_to_varia(session, opts)
  opts = opts or {}
  local fight = M.play_kraid_fight(session, opts)
  if fight.outcome ~= "kraid_defeated" then
    return {
      fight = fight,
      varia = varia_evidence(
        session.frame, nil, nil, session, "skipped_fight_failed"
      ),
    }
  end
  return {fight = fight, varia = M.play_kraid_to_varia(session)}
end

return M
