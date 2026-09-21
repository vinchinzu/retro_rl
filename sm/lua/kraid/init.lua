-- Kraid hop controllers (warehouse, Hi-Jump, Varia) + Red→Business spine.

local spine = require("kraid.spine")
local M = {
  play_to_business = spine.play_to_business,
  HOPS = spine.HOPS,
  ROOM_BUSINESS = spine.ROOM_BUSINESS,
}

local mods = {
  require("kraid.warehouse_stack"),
  require("kraid.collect_hijump"),
  require("kraid.return_hijump"),
  require("kraid.to_kraid"),
  require("kraid.from_kraid"),
  require("kraid.varia_return"),
}
for i = 1, #mods do
  for k, v in pairs(mods[i]) do
    if M[k] == nil then
      M[k] = v
    end
  end
end

return M
