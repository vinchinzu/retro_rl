-- Enemy slot scan ($0F78 + i*$40) plus Ceres steam/door and WS species.

local ram = require("ram")

local enemies = {}

enemies.ENEMY_BASE = 0x0F78
enemies.STRIDE = 0x40
enemies.ENEMY_STRIDE = 0x40
enemies.MAX_ENEMY_SLOTS = 32
enemies.OFF_MAP = 0xFE00

enemies.ATOMIC_ID = 0xE9FF
enemies.WORKROBOT_ID = 0xE8FF
enemies.COVERN_ID = 0xEA3F
enemies.CERES_STEAM_ID = 0xE1FF
enemies.CERES_DOOR_ID = 0xE23F
enemies.STEAM_HIDDEN_BIT = 0x04
enemies.STEAM_IDLE_SPRITEMAPS = {
  [0] = true,
  [0xF142] = true,
  [0x804F] = true,
}

local function u16(addr)
  return ram.u16(addr)
end

function enemies.slot_base(slot)
  return enemies.ENEMY_BASE + slot * enemies.STRIDE
end

function enemies.from_wram()
  local out = {}
  for slot = 0, enemies.MAX_ENEMY_SLOTS - 1 do
    local base = enemies.slot_base(slot)
    local enemy_id = u16(base)
    if enemy_id ~= 0 then
      local hp = u16(base + 0x14)
      if hp > 0 then
        local x = u16(base + 0x02)
        local y = u16(base + 0x06)
        if x < enemies.OFF_MAP and y < enemies.OFF_MAP then
          out[#out + 1] = {
            slot = slot,
            enemy_id = enemy_id,
            x = x,
            y = y,
            hp = hp,
            freeze_timer = u16(base + 0x26),
            extra_props = u16(base + 0x10),
            x_radius = u16(base + 0x0A),
            y_radius = u16(base + 0x0C),
            spritemap = u16(base + 0x16),
          }
        end
      end
    end
  end
  return out
end

-- session_or_nil is ignored: lsnes reads live WRAM.
function enemies.list(session_or_nil)
  return enemies.from_wram()
end

function enemies.steam_jet_shown(e)
  if not e or tonumber(e.enemy_id) ~= enemies.CERES_STEAM_ID then
    return false
  end
  return not enemies.STEAM_IDLE_SPRITEMAPS[tonumber(e.spritemap) or 0]
end

function enemies.steam_is_burning(e)
  if not e or tonumber(e.enemy_id) ~= enemies.CERES_STEAM_ID then
    return false
  end
  return ram.band(tonumber(e.extra_props) or 0, enemies.STEAM_HIDDEN_BIT) == 0
end

function enemies.overlaps(e, x, y, samus_r)
  samus_r = samus_r or 8
  if not e then
    return false
  end
  local xr = tonumber(e.x_radius) or 0
  local yr = tonumber(e.y_radius) or 0
  if xr == 0 then
    xr = 8
  end
  if yr == 0 then
    yr = 8
  end
  local dx = (tonumber(e.x) or 0) - (tonumber(x) or 0)
  local dy = (tonumber(e.y) or 0) - (tonumber(y) or 0)
  if dx < 0 then
    dx = -dx
  end
  if dy < 0 then
    dy = -dy
  end
  return dx <= xr + samus_r and dy <= yr + samus_r
end

enemies.enemy_overlaps = enemies.overlaps

return enemies
