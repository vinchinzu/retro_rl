-- Combat package. Kraid is this thread; other bosses live in sibling modules.

local kraid = require("combat.kraid")
local M = {
  kraid = kraid,
  play_kraid_fight = kraid.play_kraid_fight,
  play_kraid_fight_to_varia = kraid.play_kraid_fight_to_varia,
  play_kraid_to_varia = kraid.play_kraid_to_varia,
  play_varia_collect = kraid.play_varia_collect,
}
local extras = {
  bomb_torizo = "combat.bomb_torizo",
  spore_spawn = "combat.spore_spawn",
  ridley = "combat.ridley",
  phantoon = "combat.phantoon",
}
for name, mod in pairs(extras) do
  local ok, pkg = pcall(require, mod)
  if ok then
    M[name] = pkg
  end
end
return M

