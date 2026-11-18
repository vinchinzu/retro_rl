-- Main Street 0xCFC9. Reads the room we are in. Does not walk there.
-- A RAM match is not a leave: play always errors UNPROVEN. proven stays false.

local M = {}

local MAIN_STREET = 0xCFC9

local function num(v)
  return tonumber(v) or 0
end

local function snap(st)
  st = st or {}
  return {
    room = num(st.room_id),
    gs = num(st.game_state),
    x = num(st.samus_x or st.x),
    y = num(st.samus_y or st.y),
    pose = num(st.pose),
    door = num(st.door_transition),
  }
end

function M.at(state, ids)
  local want = (ids and ids.MAIN_STREET) or MAIN_STREET
  local s = snap(state)
  return s.room == want and s.gs == 8 and s.door == 0
end

function M.play(session, ids)
  local want = (ids and ids.MAIN_STREET) or MAIN_STREET
  local s = snap(session.state)
  if s.room ~= want or s.gs ~= 8 or s.door ~= 0 then
    error(string.format(
      "UNPROVEN: not in Main Street 0x%04X; room=0x%04X xy=(%d,%d) pose=%d gs=%d frame=%s",
      want, s.room, s.x, s.y, s.pose, s.gs, tostring(session.frame)
    ), 2)
  end
  session:idle("main_street_seen")
  s = snap(session.state)
  error(string.format(
    "UNPROVEN: room 0x%04X xy=(%d,%d) pose=%d gs=%d is not a measured tube entry and not a live leave (Main Street 0x%04X); frame=%s",
    s.room, s.x, s.y, s.pose, s.gs, want, tostring(session.frame)
  ), 2)
end

return M
