-- After Gravity. Not a live proof. proven stays false.
-- Hops error UNPROVEN themselves; a false proven flag does not skip a call.

local maridia = require("maridia")
local botwoon = require("botwoon")
local draygon = require("draygon")
local lower_norfair = require("lower_norfair")
local tourian = require("tourian")

local M = {}
M.proven = false

local function call_hop(mod, name, session)
  local fn = mod and mod[name]
  if type(fn) ~= "function" then
    error("missing " .. tostring(name))
  end
  return fn(session)
end

function M.play_after_gravity(session)
  call_hop(maridia, "play_gravity_to_main_street", session)
  call_hop(botwoon, "play_main_street_to_botwoon", session)
  call_hop(botwoon, "play_botwoon_fight", session)
  call_hop(draygon, "play_botwoon_to_space_jump", session)
  call_hop(lower_norfair, "play_space_jump_to_ridley", session)
  call_hop(tourian, "play_ridley_to_credits", session)
end

return M
