-- lsnes Lua 5.1 package.path for sm/lua.
-- Every entry script dofiles this first.

local src = debug.getinfo(1, "S").source:gsub("^@", "")
local root = src:match("^(.*)/[^/]+$") or src:match("^(.*)\\[^\\]+$") or "."
package.path = root .. "/?.lua;" .. root .. "/?/init.lua;" .. (package.path or "")
return root
