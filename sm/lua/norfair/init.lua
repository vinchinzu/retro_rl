-- Norfair Cathedral / Rising Tide / Business–Frog / Speed hop package.

local cathedral = require("norfair.cathedral")
local rising_tide = require("norfair.rising_tide")
local business_frog = require("norfair.business_frog")
local business_climb = require("norfair.business_climb")
local to_bat_cave = require("norfair.to_bat_cave")
local to_speed = require("norfair.to_speed")
local speed_return = require("norfair.speed_return")

return {
  play_business_to_cathedral_entrance = cathedral.play_business_to_cathedral_entrance,
  play_cathedral_entrance_to_cathedral = cathedral.play_cathedral_entrance_to_cathedral,
  play_cathedral_to_rising_tide = cathedral.play_cathedral_to_rising_tide,
  play_rising_tide_to_bubble = rising_tide.play_rising_tide_to_bubble,
  play_business_to_frog_save = business_frog.play_business_to_frog_save,
  play_frog_save_to_speedway = business_frog.play_frog_save_to_speedway,
  play_speedway_to_farm = business_frog.play_speedway_to_farm,
  play_farm_to_bubble = business_frog.play_farm_to_bubble,
  play_business_to_warehouse = business_climb.play_business_to_warehouse,
  play_bubble_to_bat_cave = to_bat_cave.play_bubble_to_bat_cave,
  play_bat_cave_to_speed_hall = to_speed.play_bat_cave_to_speed_hall,
  play_speed_hall_to_speed = to_speed.play_speed_hall_to_speed,
  play_speed_return_to_bubble = speed_return.play_speed_return_to_bubble,
}
