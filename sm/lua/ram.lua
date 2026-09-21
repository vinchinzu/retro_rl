-- Super Metroid WRAM snapshot for lsnes Lua 5.1.
-- Offsets match snes/super_metroid/ram.py. Moonwalk $09E4 is the only write.

local ram = {}

ram.WRAM = 0x7E0000
local WRAM = ram.WRAM

ram.ADDR_ROOM_ID = 0x079B
ram.ADDR_AREA_INDEX = 0x079F
ram.ADDR_DOOR_TRANSITION = 0x0797
ram.ADDR_TRANSITION_DIRECTION = 0x0791
ram.ADDR_GAME_STATE = 0x0998
ram.ADDR_EQUIPPED_ITEMS = 0x09A2
ram.ADDR_COLLECTED_ITEMS = 0x09A4
ram.ADDR_EQUIPPED_BEAMS = 0x09A6
ram.ADDR_COLLECTED_BEAMS = 0x09A8
ram.ADDR_HEALTH = 0x09C2
ram.ADDR_MAX_HEALTH = 0x09C4
ram.ADDR_MISSILES = 0x09C6
ram.ADDR_MAX_MISSILES = 0x09C8
ram.ADDR_SUPER_MISSILES = 0x09CA
ram.ADDR_MAX_SUPER_MISSILES = 0x09CC
ram.ADDR_POWER_BOMBS = 0x09CE
ram.ADDR_MAX_POWER_BOMBS = 0x09D0
ram.ADDR_SELECTED_ITEM = 0x09D2
ram.ADDR_MOONWALK = 0x09E4
ram.ADDR_MAX_RESERVE_HEALTH = 0x09D4
ram.ADDR_RESERVE_HEALTH = 0x09D6
ram.ADDR_SAMUS_POSE = 0x0A1C
ram.ADDR_SAMUS_FACING = 0x0A1E
ram.ADDR_MOVEMENT_TYPE = 0x0A1F
ram.ADDR_SHINESPARK_TIMER = 0x0A68
ram.ADDR_SAMUS_X = 0x0AF6
ram.ADDR_SAMUS_X_SUB = 0x0AF8
ram.ADDR_SAMUS_Y = 0x0AFA
ram.ADDR_SAMUS_Y_SUB = 0x0AFC
ram.ADDR_VELOCITY_Y_SUB = 0x0B2C
ram.ADDR_VELOCITY_Y = 0x0B2E
ram.ADDR_VERTICAL_DIRECTION = 0x0B36
ram.ADDR_SPEED_FLAG = 0x0B3C
ram.ADDR_SPEED_COUNTER = 0x0B3E
ram.ADDR_VELOCITY_X = 0x0B42
ram.ADDR_VELOCITY_X_SUB = 0x0B44
ram.ADDR_MOMENTUM_X = 0x0B46
ram.ADDR_MOMENTUM_X_SUB = 0x0B48
ram.ADDR_TIMER_TYPE = 0x0943
ram.ADDR_ESCAPE_TIMER_FRAMES = 0x0945
ram.ADDR_ESCAPE_TIMER_SECONDS = 0x0946
ram.ADDR_ESCAPE_TIMER_MINUTES = 0x0947
ram.ADDR_EVENT_FLAGS = 0xD820
ram.ADDR_BOSS_BITS = 0xD828
ram.ADDR_NUM_ENEMIES = 0x0E4E
ram.ADDR_ENEMIES_KILLED = 0x0E50
ram.ADDR_ENEMY0_X = 0x0F7A
ram.ADDR_ENEMY0_Y = 0x0F7E
ram.ADDR_ENEMY0_HP = 0x0F8C
ram.ADDR_ENEMY0_SPRITEMAP = 0x0F8E
ram.ADDR_DOOR_DEF_PTR = 0x078D
ram.ADDR_INVINCIBILITY_TIMER = 0x18A8
ram.ADDR_KNOCKBACK_TIMER = 0x18AA
ram.ADDR_RNG = 0x05E5
ram.RNG_BOOT_SEED = 0x0061

ram.FACING_LEFT = 0x04
ram.FACING_RIGHT = 0x08

ram.GS_ORDINARY = 8
ram.GS_DEAD = { [26] = true, [36] = true }
ram.GS_CERES_LEAVE = { [32] = true, [33] = true, [34] = true }

ram.MORPH_BALL_MASK = 0x0004
ram.BOMBS_MASK = 0x1000
ram.VARIA_MASK = 0x0001
ram.HI_JUMP_MASK = 0x0100
ram.GRAVITY_MASK = 0x0020
ram.EVENT_MOTHER_BRAIN_DEFEATED = 0x0E

ram.AREA_NAMES = {
  [0] = "Crateria",
  "Brinstar",
  "Norfair",
  "Wrecked Ship",
  "Maridia",
  "Tourian",
  "Ceres",
}

local function _band16(a, b)
  a = tonumber(a) or 0
  b = tonumber(b) or 0
  if a < 0 then a = a + 0x10000 end
  if b < 0 then b = b + 0x10000 end
  a = a % 0x10000
  b = b % 0x10000
  local r, p = 0, 1
  for _ = 1, 16 do
    local aa, bb = a % 2, b % 2
    if aa == 1 and bb == 1 then
      r = r + p
    end
    a = (a - aa) / 2
    b = (b - bb) / 2
    p = p * 2
  end
  return r
end

local band
if bit and bit.band then
  band = function(a, b)
    return bit.band(tonumber(a) or 0, tonumber(b) or 0)
  end
else
  band = _band16
end
ram.band = band

function ram.u8(addr)
  return memory.readbyte(WRAM + addr) % 256
end

function ram.u16(addr)
  if memory.readword then
    return memory.readword(WRAM + addr) % 0x10000
  end
  return ram.u8(addr) + ram.u8(addr + 1) * 256
end

function ram.i16(addr)
  local v = ram.u16(addr)
  if band(v, 0x8000) ~= 0 then
    return v - 0x10000
  end
  return v
end

local function write_u8(addr, value)
  value = band(value, 0xFF)
  if memory.writebyte then
    memory.writebyte(WRAM + addr, value)
    return
  end
  if memory.write then
    memory.write(WRAM + addr, value)
    return
  end
  error("ram: no memory.writebyte")
end

local function write_u16(addr, value)
  value = band(value, 0xFFFF)
  if memory.writeword then
    memory.writeword(WRAM + addr, value)
    return
  end
  write_u8(addr, value % 256)
  write_u8(addr + 1, math.floor(value / 256) % 256)
end

-- File option only. Never write energy / ammo / items / pose / xy.
function ram.set_moonwalk(on)
  local want = on and 1 or 0
  if ram.u16(ram.ADDR_MOONWALK) == want then
    return false
  end
  write_u16(ram.ADDR_MOONWALK, want)
  return true
end

function ram.phase_for_game_state(game_state, door_transition)
  door_transition = door_transition or 0
  local gs = tonumber(game_state) or 0
  if gs <= 6 or gs == 30 or gs == 31 or (gs >= 40 and gs <= 44) then
    return "boot_or_menu"
  end
  if gs == ram.GS_ORDINARY then
    if door_transition ~= 0 then
      return "room_transition"
    end
    return "ordinary_gameplay"
  end
  if gs == 7 or gs == 9 or gs == 10 or gs == 11 then
    return "room_transition"
  end
  if gs >= 12 and gs <= 18 then
    return "pause_or_inventory"
  end
  if (gs >= 19 and gs <= 26) or gs == 29 or gs == 35 or gs == 36 or gs == 37 then
    return "death_or_game_over"
  end
  if gs == 27 or gs == 32 or gs == 33 or gs == 34 then
    return "scripted_sequence"
  end
  if gs == 38 or gs == 39 then
    return "ending_or_credits"
  end
  return "unknown"
end

local State = {}
State.__index = State
ram.State = State

function State:morph_ball()
  return band(self.collected_items, ram.MORPH_BALL_MASK) ~= 0
end

function State:bombs()
  return band(self.collected_items, ram.BOMBS_MASK) ~= 0
end

function State:varia()
  return band(self.collected_items, ram.VARIA_MASK) ~= 0
end

function State:hi_jump()
  return band(self.collected_items, ram.HI_JUMP_MASK) ~= 0
end

function State:gravity()
  return band(self.collected_items, ram.GRAVITY_MASK) ~= 0
end

function State:dead()
  return self.phase == "death_or_game_over" or ram.GS_DEAD[self.game_state] or false
end

function State:moonwalk_enabled()
  return self.moonwalk ~= 0
end

function State:controllable()
  return self.phase == "ordinary_gameplay"
end

function State:facing_left()
  return self.facing == ram.FACING_LEFT
end

function State:facing_right()
  return self.facing == ram.FACING_RIGHT
end

function State:speed_boosting()
  return self.speed_counter >= 4
end

function State:shinesparking()
  return self.shinespark_timer > 0
end

function State:area_name()
  local name = ram.AREA_NAMES[self.area_index]
  if name then
    return name
  end
  return "Unknown " .. tostring(self.area_index)
end

function State:__tostring()
  return string.format(
    "room=0x%04X gs=%d pose=%d xy=(%d,%d) items=0x%04X",
    self.room_id or 0,
    self.game_state or 0,
    self.pose or 0,
    self.samus_x or 0,
    self.samus_y or 0,
    self.collected_items or 0
  )
end

ram.morph_ball = function(st) return State.morph_ball(st) end
ram.gravity = function(st) return State.gravity(st) end
ram.dead = function(st) return State.dead(st) end
ram.moonwalk_enabled = function(st) return State.moonwalk_enabled(st) end

local function current_frame()
  if movie and movie.currentframe then
    return movie.currentframe()
  end
  return 0
end

function ram.read()
  local game_state = ram.u16(ram.ADDR_GAME_STATE)
  local door_transition = ram.u16(ram.ADDR_DOOR_TRANSITION)
  local speed_word = ram.u16(ram.ADDR_SPEED_COUNTER)
  local st = {
    frame = current_frame(),
    game_state = game_state,
    phase = ram.phase_for_game_state(game_state, door_transition),
    room_id = ram.u16(ram.ADDR_ROOM_ID),
    area_index = ram.u16(ram.ADDR_AREA_INDEX),
    door_transition = door_transition,
    transition_direction = ram.u16(ram.ADDR_TRANSITION_DIRECTION),
    door_def_ptr = ram.u16(ram.ADDR_DOOR_DEF_PTR),
    samus_x = ram.u16(ram.ADDR_SAMUS_X),
    samus_x_sub = ram.u16(ram.ADDR_SAMUS_X_SUB),
    samus_y = ram.u16(ram.ADDR_SAMUS_Y),
    samus_y_sub = ram.u16(ram.ADDR_SAMUS_Y_SUB),
    velocity_x = ram.i16(ram.ADDR_VELOCITY_X),
    velocity_x_sub = ram.u16(ram.ADDR_VELOCITY_X_SUB),
    velocity_y = ram.i16(ram.ADDR_VELOCITY_Y),
    velocity_y_sub = ram.u16(ram.ADDR_VELOCITY_Y_SUB),
    momentum_x = ram.i16(ram.ADDR_MOMENTUM_X),
    momentum_x_sub = ram.u16(ram.ADDR_MOMENTUM_X_SUB),
    speed_counter = math.floor(speed_word / 256) % 256,
    speed_flag = ram.u16(ram.ADDR_SPEED_FLAG),
    vertical_direction = ram.u16(ram.ADDR_VERTICAL_DIRECTION),
    facing = ram.u8(ram.ADDR_SAMUS_FACING),
    movement_type = ram.u8(ram.ADDR_MOVEMENT_TYPE),
    shinespark_timer = ram.u16(ram.ADDR_SHINESPARK_TIMER),
    moonwalk = ram.u16(ram.ADDR_MOONWALK),
    invincibility_timer = ram.u16(ram.ADDR_INVINCIBILITY_TIMER),
    knockback_timer = ram.u16(ram.ADDR_KNOCKBACK_TIMER),
    pose = ram.u16(ram.ADDR_SAMUS_POSE),
    health = ram.u16(ram.ADDR_HEALTH),
    max_health = ram.u16(ram.ADDR_MAX_HEALTH),
    reserve_health = ram.u16(ram.ADDR_RESERVE_HEALTH),
    max_reserve_health = ram.u16(ram.ADDR_MAX_RESERVE_HEALTH),
    missiles = ram.u16(ram.ADDR_MISSILES),
    max_missiles = ram.u16(ram.ADDR_MAX_MISSILES),
    super_missiles = ram.u16(ram.ADDR_SUPER_MISSILES),
    max_super_missiles = ram.u16(ram.ADDR_MAX_SUPER_MISSILES),
    power_bombs = ram.u16(ram.ADDR_POWER_BOMBS),
    max_power_bombs = ram.u16(ram.ADDR_MAX_POWER_BOMBS),
    selected_item = ram.u16(ram.ADDR_SELECTED_ITEM),
    equipped_items = ram.u16(ram.ADDR_EQUIPPED_ITEMS),
    collected_items = ram.u16(ram.ADDR_COLLECTED_ITEMS),
    equipped_beams = ram.u16(ram.ADDR_EQUIPPED_BEAMS),
    collected_beams = ram.u16(ram.ADDR_COLLECTED_BEAMS),
    timer_type = ram.u8(ram.ADDR_TIMER_TYPE),
    escape_timer_frames = ram.u8(ram.ADDR_ESCAPE_TIMER_FRAMES),
    escape_timer_seconds = ram.u8(ram.ADDR_ESCAPE_TIMER_SECONDS),
    escape_timer_minutes = ram.u8(ram.ADDR_ESCAPE_TIMER_MINUTES),
    num_enemies = ram.u16(ram.ADDR_NUM_ENEMIES),
    enemies_killed = ram.u16(ram.ADDR_ENEMIES_KILLED),
    enemy0_x = ram.u16(ram.ADDR_ENEMY0_X),
    enemy0_y = ram.u16(ram.ADDR_ENEMY0_Y),
    enemy0_hp = ram.u16(ram.ADDR_ENEMY0_HP),
    enemy0_spritemap = ram.u16(ram.ADDR_ENEMY0_SPRITEMAP),
  }
  return setmetatable(st, State)
end

return ram
