-- Ceres Ridley tail-tank. Pure fight_action; play_fight drives the session.
-- Countdown (timer_type == 3, health < 30) is the win. Knockback poses 137/138.

local M = {}

M.ENERGY_LEAVE = 30
M.WALL_X_MIN = 220
M.WALL_X_MAX = 244
M.NUDGE_X = 214
M.JUMP_AFTER_HITS = 3
M.NUDGE_AFTER_HITS = 4
M.JUMP_HOLD_FRAMES = 24
M.DOOR_X = 48
M.ROOM_CERES_RIDLEY = 0xE0B5
M.POSE_KNOCKBACK = { [137] = true, [138] = true }
M.GS_DEAD = { [26] = true, [36] = true }

local function num(v, default)
  v = tonumber(v)
  if v == nil then
    return default or 0
  end
  return v
end

local function samus_x(state)
  return num(state.samus_x or state.x)
end

local function has_name(names, want)
  local i
  for i = 1, #names do
    if names[i] == want then
      return true
    end
  end
  return false
end

local function merge_strategy(strategy)
  local s = {
    policy = "tail_tank",
    wall_x_min = M.WALL_X_MIN,
    wall_x_max = M.WALL_X_MAX,
    nudge_x = M.NUDGE_X,
    jump_after_hits = M.JUMP_AFTER_HITS,
    nudge_after_hits = M.NUDGE_AFTER_HITS,
    energy_leave = M.ENERGY_LEAVE,
    door_x = M.DOOR_X,
    max_fight_frames = 6000,
    jump_hold_frames = M.JUMP_HOLD_FRAMES,
    fresh_fifth_jump = true,
  }
  if type(strategy) == "table" then
    local k, v
    for k, v in pairs(strategy) do
      s[k] = v
    end
  end
  return s
end

function M.strategy(overrides)
  return merge_strategy(overrides)
end

function M.countdown_started(state, energy_leave)
  energy_leave = energy_leave or M.ENERGY_LEAVE
  return num(state.room_id) == M.ROOM_CERES_RIDLEY
    and num(state.timer_type) == 3
    and num(state.health) < energy_leave
end

function M.is_knockback(state, knockback_timer)
  local pose = num(state.pose)
  return M.POSE_KNOCKBACK[pose] == true or num(knockback_timer) > 0
end

function M.fight_action(state, hits_taken, frames_since_hit, invuln, knockback_timer, strategy)
  hits_taken = num(hits_taken)
  frames_since_hit = num(frames_since_hit)
  invuln = num(invuln)
  knockback_timer = num(knockback_timer)
  strategy = merge_strategy(strategy)

  if strategy.policy == "wait" or M.countdown_started(state, strategy.energy_leave) then
    return {}
  end
  if M.is_knockback(state, knockback_timer) then
    return {}
  end

  local x = samus_x(state)
  if x > 60000 then
    return {}
  end

  if hits_taken < strategy.jump_after_hits then
    if x < strategy.wall_x_min then
      return { "RIGHT", "B" }
    end
    if x > strategy.wall_x_max then
      return { "LEFT" }
    end
    return {}
  end

  if hits_taken < strategy.nudge_after_hits then
    local names = {}
    if x < strategy.wall_x_min then
      names[#names + 1] = "RIGHT"
    elseif x > strategy.wall_x_max then
      names[#names + 1] = "LEFT"
    end
    if frames_since_hit < strategy.jump_hold_frames or invuln == 0 then
      names[#names + 1] = "A"
    end
    return names
  end

  if num(state.health) < strategy.energy_leave then
    if x > strategy.door_x then
      return { "LEFT", "B" }
    end
    return {}
  end

  local names = {}
  if x < strategy.wall_x_min then
    names[#names + 1] = "RIGHT"
  elseif x > strategy.wall_x_max then
    names[#names + 1] = "LEFT"
  end
  if not strategy.fresh_fifth_jump then
    names[#names + 1] = "A"
    return names
  end
  local mt = num(state.movement_type)
  local vd = num(state.vertical_direction)
  local air = (mt == 2 or mt == 3 or mt == 6 or mt == 23) or (vd == 1 or vd == 2)
  if air then
    names[#names + 1] = "A"
    return names
  end
  if invuln > 55 and frames_since_hit < 50 then
    return names
  end
  if not has_name(names, "RIGHT") and not has_name(names, "LEFT") then
    names[#names + 1] = "RIGHT"
  end
  names[#names + 1] = "B"
  names[#names + 1] = "A"
  return names
end

function M.fight_terminal(state, energy_leave)
  energy_leave = energy_leave or M.ENERGY_LEAVE
  if M.countdown_started(state, energy_leave) then
    return "ceres_ridley_countdown"
  end
  if M.GS_DEAD[num(state.game_state)] then
    return "death"
  end
  if num(state.room_id) ~= M.ROOM_CERES_RIDLEY then
    return "left_room"
  end
  return nil
end

function M.require_countdown(evidence)
  local outcome = evidence and evidence.outcome
  if outcome ~= "ceres_ridley_countdown" then
    error("Ceres Ridley did not start escape (" .. tostring(outcome) .. ")")
  end
end

local function brief(state)
  if not state then
    return "nil"
  end
  return string.format(
    "room=0x%04X gs=%s xy=%s,%s pose=%s health=%s timer_type=%s",
    num(state.room_id),
    tostring(state.game_state),
    tostring(state.samus_x or state.x),
    tostring(state.samus_y or state.y),
    tostring(state.pose),
    tostring(state.health),
    tostring(state.timer_type)
  )
end

function M.play_fight(session, strategy)
  strategy = merge_strategy(strategy)
  local start = num(session.frame)
  local start_health = num(session.state.health)
  local hits = 0
  local hit_frames = {}
  local prev_health = start_health
  local frames_since_hit = 0
  local outcome = "timeout"
  local st = session.state

  if num(st.room_id) ~= M.ROOM_CERES_RIDLEY then
    error(string.format(
      "Ceres Ridley expected room 0x%04X, got 0x%04X",
      M.ROOM_CERES_RIDLEY,
      num(st.room_id)
    ))
  end

  local i
  for i = 1, strategy.max_fight_frames do
    st = session.state
    local terminal = M.fight_terminal(st, strategy.energy_leave)
    if terminal == "ceres_ridley_countdown" then
      outcome = terminal
      break
    end
    if terminal == "death" then
      error("Ceres Ridley death during fight: " .. brief(st))
    end
    if terminal == "left_room" then
      outcome = terminal
      break
    end

    -- ram.lua snapshot; do not reach through session.env.
    local invuln = num(st.invincibility_timer)
    local kb = num(st.knockback_timer)
    local names = M.fight_action(st, hits, frames_since_hit, invuln, kb, strategy)
    local reason
    if strategy.policy == "wait" then
      reason = "ceres_ridley_natural_countdown"
    else
      reason = "ceres_ridley_tail_tank"
    end
    session:step(names, reason)

    local post = session.state
    local health = num(post.health)
    frames_since_hit = frames_since_hit + 1
    if health > 0 and health < prev_health then
      hits = hits + 1
      hit_frames[#hit_frames + 1] = num(session.frame)
      frames_since_hit = 0
    end
    prev_health = health
    local post_term = M.fight_terminal(post, strategy.energy_leave)
    if post_term == "ceres_ridley_countdown" then
      outcome = post_term
      break
    end
    if post_term == "death" then
      error("Ceres Ridley death during fight: " .. brief(post))
    end
  end

  local end_state = session.state
  return {
    start_frame = start,
    end_frame = num(session.frame),
    action_frames = num(session.frame) - start,
    policy = strategy.policy,
    start_health = start_health,
    end_health = num(end_state.health),
    hits = hits,
    hit_frames = hit_frames,
    timer_type = num(end_state.timer_type),
    escape_timer_seconds = num(end_state.escape_timer_seconds),
    final_x = samus_x(end_state),
    final_y = num(end_state.samus_y or end_state.y),
    outcome = outcome,
  }
end

return M
