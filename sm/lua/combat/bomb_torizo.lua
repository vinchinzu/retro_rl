-- Bomb Torizo one-frame kite + play_fight. Clean: never poke missiles.
-- Natural 10-missile pack + drops. SELECT cycles the HUD; no ammo writes.
--
-- Port of snes/super_metroid/combat/bomb_torizo.py. Defeat is enemy0_hp == 0;
-- do not idle on open-bus boss_bits.

local ram = require("ram")
local rooms = require("rooms")

local M = {}

M.ROOM_BOMB_TORIZO = rooms.ROOM_BOMB_TORIZO or 0x9804
M.STATUE_SPRITEMAP = 0x87D0
M.SPAWN_SPRITEMAP = 0x804F
M.BOMB_TORIZO_BOSS_BIT = 0x04
M.MAX_HP = 800
M.HITBOX_W = 73
M.HITBOX_H = 90
M.MAX_ENEMY_SLOTS = 4
M.INACTIVE = {
  [0x87D0] = true,
  [0x804F] = true,
}

-- Clean kite (natural ~3/10 missiles at continuous entry).
M.MIN_RANGE = 100
M.MAX_RANGE = 160
M.JUMP_RANGE = 100
M.JUMP_HOLD_FRAMES = 28
M.JUMP_PERIOD = 40
M.FIRE_PERIOD = 2
M.MAX_FIGHT_FRAMES = 12000
M.BOSS_BIT_GRACE_FRAMES = 1200

local GS_DEAD = { [26] = true, [36] = true }

local function num(v, default)
  v = tonumber(v)
  if v == nil then
    return default or 0
  end
  return v
end

local function has_mask(value, mask)
  return math.floor(num(value) / mask) % 2 == 1
end

local function timeout(msg)
  local ok, runtime = pcall(require, "runtime")
  if ok and runtime and runtime.timeout then
    runtime.timeout(msg)
  end
  error("TimeoutError: " .. msg)
end

local function brief(state)
  if type(state) ~= "table" then
    return tostring(state)
  end
  return string.format(
    "room=0x%04X gs=%s xy=%s,%s pose=%s hp=%s sm=0x%04X items=0x%04X",
    num(state.room_id),
    tostring(state.game_state),
    tostring(state.samus_x or state.x),
    tostring(state.samus_y or state.y),
    tostring(state.pose),
    tostring(state.enemy0_hp),
    num(state.enemy0_spritemap),
    num(state.collected_items)
  )
end

local function frame_of(session)
  local st = session.state
  if st and st.frame ~= nil then
    return num(st.frame)
  end
  return num(session.frame)
end

local function hold(session, frames, names, reason)
  names = names or {}
  if frames <= 0 then
    return session.state
  end
  if session.hold then
    session:hold(frames, names, reason)
    return session.state
  end
  local i
  for i = 1, frames do
    session:step(names, reason)
  end
  return session.state
end

function M.crateria_boss_bits(state)
  if state and state.crateria_boss_bits ~= nil then
    return num(state.crateria_boss_bits)
  end
  local bits = state and state.boss_bits
  if type(bits) == "table" then
    if bits[0] ~= nil then
      return num(bits[0])
    end
    return num(bits[1])
  end
  if bits ~= nil then
    return num(bits)
  end
  -- ram.read() may omit boss_bits; $7E:D828 is a read, not a poke.
  if ram.u8 and ram.ADDR_BOSS_BITS then
    return ram.u8(ram.ADDR_BOSS_BITS)
  end
  return 0
end

function M.boss_bit_set(state)
  local bits = M.crateria_boss_bits(state)
  if ram.band then
    return ram.band(bits, M.BOMB_TORIZO_BOSS_BIT) ~= 0
  end
  return has_mask(bits, M.BOMB_TORIZO_BOSS_BIT)
end

function M.enemy_active(state)
  if num(state.room_id) ~= M.ROOM_BOMB_TORIZO then
    return false
  end
  local hp = num(state.enemy0_hp)
  if not (hp > 0 and hp <= M.MAX_HP) then
    return false
  end
  local sm = num(state.enemy0_spritemap)
  if sm == 0 or M.INACTIVE[sm] then
    return false
  end
  local n = state.num_enemies
  if n ~= nil and num(n) > M.MAX_ENEMY_SLOTS then
    return false
  end
  return true
end

function M.features(state)
  local sx = num(state.samus_x or state.x)
  local sy = num(state.samus_y or state.y)
  local ex = num(state.enemy0_x)
  local ey = num(state.enemy0_y)
  local dx = ex - sx
  local dy = ey - sy
  local hp = num(state.enemy0_hp)
  return {
    room_id = num(state.room_id),
    samus_x = sx,
    samus_y = sy,
    enemy_x = ex,
    enemy_y = ey,
    enemy_hp = hp,
    enemy_spritemap = num(state.enemy0_spritemap),
    dx = dx,
    dy = dy,
    missiles = num(state.missiles),
    max_missiles = num(state.max_missiles),
    selected_item = num(state.selected_item),
    enemy_active = M.enemy_active(state),
    enemy_defeated = hp == 0 and num(state.room_id) == M.ROOM_BOMB_TORIZO,
  }
end

function M.strategy(overrides)
  local s = {
    min_range = M.MIN_RANGE,
    max_range = M.MAX_RANGE,
    jump_range = M.JUMP_RANGE,
    jump_hold_frames = M.JUMP_HOLD_FRAMES,
    jump_period = M.JUMP_PERIOD,
    fire_period = M.FIRE_PERIOD,
    max_fight_frames = M.MAX_FIGHT_FRAMES,
    boss_bit_grace_frames = M.BOSS_BIT_GRACE_FRAMES,
  }
  if type(overrides) == "table" then
    local k, v
    for k, v in pairs(overrides) do
      s[k] = v
    end
  end
  return s
end

-- One-frame button names. frame_index is 0-based like the Python policy.
function M.fight_action(state, frame_index, strategy)
  strategy = M.strategy(strategy)
  if num(state.enemy0_hp) == 0 then
    return {}
  end
  local feat = M.features(state)
  local sm = num(state.enemy0_spritemap)
  if not feat.enemy_active and (sm == M.STATUE_SPRITEMAP or sm == M.SPAWN_SPRITEMAP) then
    return { "RIGHT" }
  end
  local dx = feat.dx
  local abs_dx
  if dx >= 0 then
    abs_dx = dx
  else
    abs_dx = -dx
  end
  local face
  if dx >= 0 then
    face = "RIGHT"
  else
    face = "LEFT"
  end
  local move
  if abs_dx < strategy.min_range then
    if dx >= 0 then
      move = "LEFT"
    else
      move = "RIGHT"
    end
  else
    move = face
  end
  local names = { move }
  if abs_dx < strategy.jump_range
      and (num(frame_index) % strategy.jump_period) < strategy.jump_hold_frames then
    names[#names + 1] = "A"
  end
  -- Fire X on the period even with 0 missiles (beam fallback). Never poke ammo.
  if (num(frame_index) % strategy.fire_period) == 0 then
    names[#names + 1] = "X"
  end
  return names
end

M.fight_bomb_torizo_action = M.fight_action

-- SELECT-cycle to missiles (1). HUD only — no WRAM ammo write.
function M.select_weapon(session, target, max_cycles)
  target = target or 1
  max_cycles = max_cycles or 8
  local i
  for i = 1, max_cycles do
    if num(session.state.selected_item) == target then
      return
    end
    hold(session, 1, {"SELECT"}, "select_weapon")
    hold(session, 25, {}, "select_weapon_settle")
  end
  if num(session.state.selected_item) ~= target then
    timeout("could not select weapon " .. tostring(target)
      .. ", still " .. tostring(session.state.selected_item))
  end
end

function M.fight_ready(state)
  if num(state.room_id) ~= M.ROOM_BOMB_TORIZO then
    return false
  end
  local items = num(state.collected_items)
  local bombs = ram.BOMBS_MASK or 0x1000
  local has_bombs
  if ram.band then
    has_bombs = ram.band(items, bombs) ~= 0
  else
    has_bombs = has_mask(items, bombs)
  end
  if not has_bombs then
    return false
  end
  local sm = num(state.enemy0_spritemap)
  if sm == M.STATUE_SPRITEMAP or sm == M.SPAWN_SPRITEMAP or sm == 0 then
    return false
  end
  return num(state.enemy0_hp) >= 800
end

function M.play_fight(session, strategy, require_active, require_boss_bit)
  strategy = M.strategy(strategy)
  if require_active == nil then
    require_active = true
  end
  if require_boss_bit == nil then
    require_boss_bit = true
  end
  local start = frame_of(session)
  local st = session.state
  if num(st.room_id) ~= M.ROOM_BOMB_TORIZO then
    timeout(string.format(
      "Bomb Torizo fight expected room 0x%04X, got 0x%04X",
      M.ROOM_BOMB_TORIZO,
      num(st.room_id)
    ))
  end

  -- Natural pack only. SELECT to missiles when capacity exists.
  if num(st.selected_item) ~= 1 and num(st.max_missiles) > 0 then
    M.select_weapon(session, 1)
  end

  local peak_hp = num(session.state.enemy0_hp)
  local min_hp = peak_hp
  local activation_seen = num(session.state.enemy0_spritemap) ~= M.STATUE_SPRITEMAP
  local defeat_frame = nil
  local boss_bit_frame = nil
  local prev_hp = num(session.state.enemy0_hp)
  local outcome = "timeout"
  local index
  for index = 0, strategy.max_fight_frames - 1 do
    st = session.state
    if GS_DEAD[num(st.game_state)] then
      timeout("Bomb Torizo death during fight: " .. brief(st))
    end
    local hp = num(st.enemy0_hp)
    if hp > peak_hp then
      peak_hp = hp
    end
    if hp < min_hp then
      min_hp = hp
    end
    if M.enemy_active(st) then
      activation_seen = true
    end
    local names = M.fight_action(st, index, strategy)
    if names[1] then
      hold(session, 1, names, "fight_bomb_torizo")
    else
      hold(session, 1, {}, "fight_bomb_torizo_idle")
    end
    st = session.state
    hp = num(st.enemy0_hp)
    if defeat_frame == nil and hp == 0 and prev_hp > 0 then
      defeat_frame = frame_of(session)
      min_hp = 0
      if not require_boss_bit then
        break
      end
    end
    prev_hp = hp
    if defeat_frame ~= nil and require_boss_bit and M.boss_bit_set(st) then
      boss_bit_frame = frame_of(session)
      break
    end
    if defeat_frame ~= nil and require_boss_bit
        and (frame_of(session) - defeat_frame) >= strategy.boss_bit_grace_frames then
      break
    end
  end

  st = session.state
  if defeat_frame ~= nil and (
      (not require_boss_bit)
      or boss_bit_frame ~= nil
      or M.boss_bit_set(st)
  ) then
    if boss_bit_frame == nil and M.boss_bit_set(st) then
      boss_bit_frame = frame_of(session)
    end
    outcome = "bomb_torizo_defeated"
  elseif require_active and not activation_seen then
    outcome = "torizo_inactive_statue"
  elseif defeat_frame ~= nil and require_boss_bit then
    outcome = "boss_bit_timeout"
  else
    outcome = "timeout"
  end

  return {
    start_frame = start,
    activation_seen = activation_seen,
    defeat_frame = defeat_frame,
    boss_bit_frame = boss_bit_frame,
    end_frame = frame_of(session),
    peak_hp = peak_hp,
    min_enemy_hp = min_hp,
    action_frames = frame_of(session) - start,
    final_enemy_hp = num(session.state.enemy0_hp),
    outcome = outcome,
  }
end

M.play_bomb_torizo_fight = M.play_fight

return M
