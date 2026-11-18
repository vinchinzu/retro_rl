-- Bounded Botwoon spray. Clean: enemy slots are read off session.state only.
-- sm/lua/combat and sm/lua/ram.lua record dead bits for Bomb Torizo ($D828
-- bit 0x04), Spore Spawn (area 1 bit 0x02), Kraid ($D829 bit 0x01), and
-- Phantoon ($D82B bit 0x01). None of those is Botwoon's bit. enemy0_hp == 0
-- is not that condition, and this file does not write HP.

local M = {}

M.proven = false
M.ROOM_BOTWOON = 0xD95E
M.GS_ORDINARY = 8
M.MAX_FIGHT_FRAMES = 2400

local function num(v)
  return tonumber(v) or 0
end

-- Same fields a Phantoon loop reads from session.state (slot 0).
local function enemy_slot(state)
  return {
    x = num(state.enemy0_x),
    y = num(state.enemy0_y),
    hp = num(state.enemy0_hp),
    spritemap = num(state.enemy0_spritemap),
  }
end

local function on_map(v)
  return v > 0 and v < 0xFE00
end

local function shot_names(state)
  local slot = enemy_slot(state)
  local names = { "X" }
  if slot.spritemap == 0 and slot.hp == 0 and slot.x == 0 then
    return names
  end
  local sx = num(state.samus_x)
  local sy = num(state.samus_y)
  if on_map(slot.x) then
    if sx + 32 < slot.x then
      names[#names + 1] = "right"
    elseif slot.x + 32 < sx then
      names[#names + 1] = "left"
    end
  end
  if on_map(slot.y) then
    if sy > slot.y + 24 then
      names[#names + 1] = "up"
    elseif slot.y > sy + 48 then
      names[#names + 1] = "down"
    end
  end
  return names
end

function M.play_botwoon_fight(session)
  if type(session) ~= "table" or type(session.state) ~= "table" or type(session.step) ~= "function" then
    error("botwoon fight requires a session")
  end
  local room = num(session.state.room_id)
  local gs = num(session.state.game_state)
  if room ~= M.ROOM_BOTWOON or gs ~= M.GS_ORDINARY then
    error(string.format(
      "botwoon fight requires room 0xD95E gs=8, got room 0x%04X gs=%d",
      room, gs
    ))
  end
  local i
  for i = 1, M.MAX_FIGHT_FRAMES do
    local st = session.state
    if type(st) ~= "table" then
      break
    end
    if num(st.room_id) ~= M.ROOM_BOTWOON or num(st.game_state) ~= M.GS_ORDINARY then
      break
    end
    if num(st.health) == 0 then
      break
    end
    session:step(shot_names(st), "botwoon_fight")
  end
  error("UNPROVEN: botwoon dead bit")
end

return M
