-- Shinespark charge / crouch-store / activate. Dash=B, jump/activate=A, store=DOWN.

local ram = require("ram")
local plans = require("skills.shinespark_plans")

local shinespark = {}

shinespark.ECHOES_FULL = 4
shinespark.TYPICAL_ARM_TIMER = 179
shinespark.TYPICAL_CHARGE_FRAMES = 90
shinespark.DEFAULT_STORE_MAX_FRAMES = 24
shinespark.DEFAULT_PRE_STAND_FRAMES = 4
shinespark.SPARK_POSES = {
  [199] = true, [200] = true, [201] = true, [202] = true,
  [203] = true, [204] = true, [205] = true, [206] = true,
}
shinespark.STORE_OK_POSES = {
  [1] = true, [2] = true, [5] = true, [6] = true, [9] = true, [10] = true,
}
shinespark.STORE_WIPE_POSES = {
  [25] = true, [26] = true, [27] = true, [28] = true, [166] = true, [167] = true,
}
shinespark.KNOCKBACK_POSES = { [137] = true, [138] = true }

shinespark.NTSC_MAGIC_DASH_FRAMES = plans.NTSC_MAGIC_DASH_FRAMES
shinespark.PAL_MAGIC_DASH_FRAMES = plans.PAL_MAGIC_DASH_FRAMES
shinespark.NTSC_STUTTER_MIN_PX = plans.NTSC_STUTTER_MIN_PX
shinespark.PAL_STUTTER_MIN_PX = plans.PAL_STUTTER_MIN_PX
shinespark.NTSC_STUTTER_FULL_STOP_PX = plans.NTSC_STUTTER_FULL_STOP_PX
shinespark.NTSC_SHORT_CHARGE_FRAMES = plans.NTSC_SHORT_CHARGE_FRAMES
shinespark.PAL_SHORT_CHARGE_FRAMES = plans.PAL_SHORT_CHARGE_FRAMES
shinespark.magic_dash_frames = plans.magic_dash_frames
shinespark.stutter_forward_mask = plans.stutter_forward_mask
shinespark.stutter_dash_mask = plans.stutter_dash_mask
shinespark.short_charge_plan = plans.short_charge_plan

function shinespark.read_spark_wram()
  local sc_word = ram.u16(ram.ADDR_SPEED_COUNTER)
  return {
    spark_timer = ram.u16(ram.ADDR_SHINESPARK_TIMER),
    speed_flag = ram.u16(ram.ADDR_SPEED_FLAG),
    speed_counter_word = sc_word,
    speed_echoes = math.floor(sc_word / 256) % 256,
    speed_anim = sc_word % 256,
  }
end

function shinespark.session_spark_wram(session)
  return shinespark.read_spark_wram()
end

function shinespark.spark_snapshot(frame)
  local st = ram.read()
  local w = shinespark.read_spark_wram()
  return {
    frame = frame or st.frame,
    room = st.room_id,
    room_hex = string.format("0x%04X", st.room_id),
    x = st.samus_x,
    y = st.samus_y,
    pose = st.pose,
    facing = st.facing,
    vx = st.velocity_x,
    vy = st.velocity_y,
    gs = st.game_state,
    door_trans = st.door_transition,
    health = st.health,
    spark_timer = w.spark_timer,
    speed_flag = w.speed_flag,
    speed_counter_word = w.speed_counter_word,
    speed_echoes = w.speed_echoes,
    speed_anim = w.speed_anim,
    speed_boosting = st:speed_boosting(),
    shinesparking = st:shinesparking(),
  }
end

local function snap(session)
  return shinespark.spark_snapshot(session and session.frame)
end

function shinespark.is_spark_pose(pose)
  return shinespark.SPARK_POSES[tonumber(pose) or -1] or false
end

function shinespark.store_pose_ok(pose)
  local p = tonumber(pose) or -1
  if shinespark.STORE_WIPE_POSES[p] or shinespark.KNOCKBACK_POSES[p] then
    return false
  end
  return shinespark.STORE_OK_POSES[p] or p == 39 or p == 40 or p == 53 or p == 54
end

local function echoes_of(session)
  local ok, w = pcall(shinespark.session_spark_wram, session)
  if ok and w then
    return w.speed_echoes
  end
  return tonumber(session.state.speed_counter) or 0
end

local function timer_of(session)
  local ok, w = pcall(shinespark.session_spark_wram, session)
  if ok and w then
    return w.spark_timer
  end
  return tonumber(session.state.shinespark_timer) or 0
end

function shinespark.charge_by_plan(session, plan, opts)
  opts = opts or {}
  local require_grounded = opts.require_grounded
  if require_grounded == nil then
    require_grounded = true
  end
  local label = opts.label or "short_charge"
  local stop_on_boost = opts.stop_on_boost
  if stop_on_boost == nil then
    stop_on_boost = true
  end
  local first_echo, boost_row = nil, nil
  local start_frame = tonumber(session.frame) or 0
  local start_x = tonumber(session.state.samus_x) or 0
  local buttons_log, dash_frames = {}, {}
  local function finish(ok, i, early)
    local st = session.state
    local end_x = tonumber(st.samus_x) or 0
    local dir = "RIGHT"
    if plan[1] then
      for b = 1, #plan[1] do
        if plan[1][b] == "LEFT" then
          dir = "LEFT"
        end
      end
    end
    return {
      ok = ok,
      frames = i,
      elapsed = (tonumber(session.frame) or 0) - start_frame,
      direction = dir,
      first_echo = first_echo,
      boost = boost_row or snap(session),
      plan_len = #plan,
      start_x = start_x,
      end_x = end_x,
      delta_x = end_x - start_x,
      dash_frames = dash_frames,
      buttons_log = buttons_log,
      early_stop = early and true or false,
    }
  end
  for i = 1, #plan do
    local st = session.state
    local echoes = echoes_of(session)
    if first_echo == nil and echoes >= 1 then
      first_echo = snap(session)
    end
    local grounded = true
    if require_grounded then
      grounded = tonumber(st.velocity_y) == 0
    end
    local boosting = st.speed_boosting and st:speed_boosting() or (echoes >= shinespark.ECHOES_FULL)
    if stop_on_boost and boosting and grounded and not shinespark.KNOCKBACK_POSES[tonumber(st.pose)] then
      boost_row = snap(session)
      return finish(true, i - 1, true)
    end
    local btn_t = plan[i]
    session:hold(1, btn_t, label .. "_" .. tostring(i - 1))
    local has_dash = false
    for b = 1, #btn_t do
      if btn_t[b] == "B" or btn_t[b] == "Y" then
        has_dash = true
      end
    end
    if has_dash then
      dash_frames[#dash_frames + 1] = i - 1
    end
    if has_dash or i == 1 or i == #plan then
      buttons_log[#buttons_log + 1] = { f = i - 1, buttons = btn_t }
    end
  end
  local st = session.state
  local echoes = echoes_of(session)
  local grounded = true
  if require_grounded then
    grounded = tonumber(st.velocity_y) == 0
  end
  local boosting = st.speed_boosting and st:speed_boosting() or (echoes >= shinespark.ECHOES_FULL)
  if first_echo == nil and echoes >= 1 then
    first_echo = snap(session)
  end
  local ok = boosting and grounded and not shinespark.KNOCKBACK_POSES[tonumber(st.pose)]
  local out = finish(ok, #plan, false)
  if not ok then
    out.error = string.format(
      "short charge plan finished without echoes>=4 (echoes=%s grounded=%s pose=%s)",
      tostring(echoes),
      tostring(grounded),
      tostring(st.pose)
    )
  end
  return out
end

function shinespark.short_charge_until_boost(session, direction, opts)
  opts = opts or {}
  direction = direction or "RIGHT"
  local region = opts.region or "NTSC"
  local plan = plans.short_charge_plan(region, {
    stutter = opts.stutter,
    store_on_last = opts.store_on_last,
    direction = direction,
    dash_button = opts.dash_button or "B",
  })
  local stop_on_boost = opts.stop_on_boost
  if stop_on_boost == nil then
    stop_on_boost = true
  end
  if opts.store_on_last then
    stop_on_boost = false
  end
  local require_grounded = opts.require_grounded
  if require_grounded == nil then
    require_grounded = true
  end
  local report = shinespark.charge_by_plan(session, plan, {
    require_grounded = require_grounded,
    label = opts.label or "short_charge",
    stop_on_boost = stop_on_boost,
  })
  report.region = region
  report.stutter = opts.stutter and true or false
  report.store_on_last = opts.store_on_last and true or false
  report.mode = opts.stutter and "stutter" or "short"
  report.magic_frames = plans.magic_dash_frames(region)
  if opts.store_on_last then
    local timer = timer_of(session)
    local echoes = echoes_of(session)
    report.store_armed = timer > 0
    report.spark_timer = timer
    if timer > 0 or echoes >= shinespark.ECHOES_FULL then
      report.ok = true
      report.error = nil
    end
  end
  return report
end

function shinespark.charge_until_boost(session, direction, opts)
  opts = opts or {}
  direction = direction or "RIGHT"
  local mode = opts.mode or "full"
  if mode == "short" or mode == "stutter" then
    return shinespark.short_charge_until_boost(session, direction, {
      region = opts.region or "NTSC",
      stutter = mode == "stutter",
      store_on_last = opts.store_on_last,
      dash_button = opts.dash_button or "B",
      require_grounded = opts.require_grounded,
      label = opts.label or "charge",
    })
  end
  local dir_btn = direction == "LEFT" and "LEFT" or "RIGHT"
  local require_grounded = opts.require_grounded
  if require_grounded == nil then
    require_grounded = true
  end
  local budget = opts.budget or 500
  local dash_button = opts.dash_button or "B"
  local extra = opts.extra_buttons or {}
  local label = opts.label or "charge"
  local first_echo, boost_row = nil, nil
  local start_frame = tonumber(session.frame) or 0
  local start_x = tonumber(session.state.samus_x) or 0
  local run = { dir_btn, dash_button }
  for i = 1, #extra do
    run[#run + 1] = extra[i]
  end
  for i = 0, budget - 1 do
    local st = session.state
    local echoes = echoes_of(session)
    if first_echo == nil and echoes >= 1 then
      first_echo = snap(session)
    end
    local grounded = true
    if require_grounded then
      grounded = tonumber(st.velocity_y) == 0
    end
    local boosting = st.speed_boosting and st:speed_boosting() or (echoes >= shinespark.ECHOES_FULL)
    if boosting and grounded and not shinespark.KNOCKBACK_POSES[tonumber(st.pose)] then
      boost_row = snap(session)
      local end_x = tonumber(st.samus_x) or 0
      return {
        ok = true,
        frames = i,
        elapsed = (tonumber(session.frame) or 0) - start_frame,
        direction = dir_btn,
        first_echo = first_echo,
        boost = boost_row,
        mode = "full",
        start_x = start_x,
        end_x = end_x,
        delta_x = end_x - start_x,
      }
    end
    session:hold(1, run, label .. "_run")
  end
  local end_x = tonumber(session.state.samus_x) or 0
  return {
    ok = false,
    frames = budget,
    elapsed = (tonumber(session.frame) or 0) - start_frame,
    direction = dir_btn,
    first_echo = first_echo,
    boost = snap(session),
    mode = "full",
    start_x = start_x,
    end_x = end_x,
    delta_x = end_x - start_x,
    error = "never reached speed_boosting (echoes>=4)",
  }
end

function shinespark.crouch_store(session, opts)
  opts = opts or {}
  local max_frames = opts.max_frames or shinespark.DEFAULT_STORE_MAX_FRAMES
  local label = opts.label or "store"
  local armed = nil
  local peak = 0
  local start_pose = tonumber(session.state.pose)
  for i = 0, max_frames - 1 do
    session:hold(1, { "DOWN" }, label .. "_" .. tostring(i))
    local timer = timer_of(session)
    if timer > peak then
      peak = timer
    end
    if armed == nil and timer > 0 then
      armed = snap(session)
      armed.store_frame_index = i
      armed.start_pose = start_pose
      break
    end
  end
  return {
    ok = armed ~= nil,
    armed = armed,
    peak_timer_during_store = peak,
    after = snap(session),
    start_pose = start_pose,
    error = armed ~= nil and nil or string.format(
      "store never armed $0A68 (peak=%s, start_pose=%s)",
      tostring(peak),
      tostring(start_pose)
    ),
  }
end

function shinespark.wait_store_window(session, frames, opts)
  opts = opts or {}
  local hold_down = opts.hold_down
  local label = opts.label or "idle"
  local series = {}
  local alive = 0
  if frames < 0 then
    frames = 0
  end
  for i = 0, frames - 1 do
    if hold_down then
      session:hold(1, { "DOWN" }, label .. "_store_" .. tostring(i))
    else
      session:hold(1, {}, label .. "_" .. tostring(i))
    end
    local row = snap(session)
    series[#series + 1] = row
    local timer = tonumber(row.spark_timer or row.shinespark_timer) or 0
    if timer > 0 then
      alive = alive + 1
    elseif i > 0 and series[1] then
      local t0 = tonumber(series[1].spark_timer or series[1].shinespark_timer) or 0
      if t0 > 0 then
        break
      end
    end
  end
  local function t_of(row)
    return tonumber(row.spark_timer or row.shinespark_timer) or 0
  end
  local first_zero = nil
  for i = 1, #series do
    if t_of(series[i]) == 0 then
      first_zero = series[i].frame
      break
    end
  end
  local drain = nil
  if #series > 1 then
    drain = (t_of(series[1]) - t_of(series[#series])) / (#series - 1)
  end
  return {
    requested_frames = frames,
    hold_down = hold_down and true or false,
    frames_timer_gt0 = alive,
    timer_start = series[1] and t_of(series[1]) or 0,
    timer_end = series[#series] and t_of(series[#series]) or 0,
    first_zero_frame = first_zero,
    samples = series,
    sample_count = #series,
    drain_per_frame = drain,
  }
end

function shinespark.activate_shinespark(session, aim_buttons, opts)
  opts = opts or {}
  aim_buttons = aim_buttons or {}
  local activate_button = opts.activate_button or "A"
  local hold_frames = opts.hold_frames or 12
  local travel_budget = opts.travel_budget or 0
  local pre_stand_frames = opts.pre_stand_frames
  if pre_stand_frames == nil then
    pre_stand_frames = shinespark.DEFAULT_PRE_STAND_FRAMES
  end
  local pre_stand_buttons = opts.pre_stand_buttons
  if pre_stand_buttons == nil then
    pre_stand_buttons = { "UP" }
  end
  local label = opts.label or "activate"
  local pre_snap = nil
  if pre_stand_frames > 0 then
    for i = 0, pre_stand_frames - 1 do
      if pre_stand_buttons and #pre_stand_buttons > 0 then
        session:hold(1, pre_stand_buttons, label .. "_pre_" .. tostring(i))
      else
        session:hold(1, {}, label .. "_pre_" .. tostring(i))
      end
    end
    pre_snap = snap(session)
  end
  local activate_snap = nil
  local spark_seen = false
  local min_y = tonumber(session.state.samus_y) or 0
  local max_x = tonumber(session.state.samus_x) or 0
  local min_x = max_x
  local start_room = tonumber(session.state.room_id)
  local fire = {}
  for i = 1, #aim_buttons do
    fire[i] = aim_buttons[i]
  end
  fire[#fire + 1] = activate_button
  for i = 0, hold_frames - 1 do
    session:hold(1, fire, label .. "_" .. tostring(i))
    local st = session.state
    local y = tonumber(st.samus_y) or 0
    local x = tonumber(st.samus_x) or 0
    if y < min_y then
      min_y = y
    end
    if x < 60000 then
      if x > max_x then
        max_x = x
      end
      if x < min_x then
        min_x = x
      end
    end
    if shinespark.is_spark_pose(st.pose) then
      spark_seen = true
      if activate_snap == nil or not shinespark.is_spark_pose(activate_snap.pose) then
        activate_snap = snap(session)
        activate_snap.activate_frame_index = i
      end
    end
  end
  local travel_rows = {}
  if travel_budget > 0 then
    local hold_btns = opts.travel_hold or fire
    for i = 0, travel_budget - 1 do
      session:hold(1, hold_btns, label .. "_travel_" .. tostring(i))
      local st = session.state
      local y = tonumber(st.samus_y) or 0
      local x = tonumber(st.samus_x) or 0
      if y < min_y then
        min_y = y
      end
      if x < 60000 then
        if x > max_x then
          max_x = x
        end
        if x < min_x then
          min_x = x
        end
      end
      if i % 10 == 0 or shinespark.is_spark_pose(st.pose) then
        travel_rows[#travel_rows + 1] = snap(session)
      end
      local timer = timer_of(session)
      if timer == 0 and not shinespark.is_spark_pose(st.pose) and i > 8 then
        break
      end
    end
  end
  local final = snap(session)
  return {
    ok = spark_seen or shinespark.is_spark_pose(session.state.pose),
    aim = aim_buttons,
    pre_stand = pre_snap,
    pre_stand_frames = pre_stand_frames,
    pre_stand_buttons = pre_stand_buttons,
    activate = activate_snap or final,
    spark_pose_seen = spark_seen,
    final = final,
    min_y = min_y,
    max_x = max_x,
    min_x = min_x,
    start_room = start_room,
    end_room = tonumber(session.state.room_id),
    room_changed = tonumber(session.state.room_id) ~= start_room,
    travel_samples = travel_rows,
  }
end

function shinespark.store_then_spin_unspin_activate(session, opts)
  opts = opts or {}
  local stand_frames = opts.stand_frames or 8
  local hop_frames = opts.hop_frames or 13
  local hop_direction = opts.hop_direction or "RIGHT"
  local unspin_frames = opts.unspin_frames or 4
  local unspin_buttons = opts.unspin_buttons or { "UP" }
  local aim_buttons = opts.aim_buttons or { "RIGHT" }
  local micro_run_frames = opts.micro_run_frames or 0
  local activate_hold = opts.activate_hold or 12
  local travel_budget = opts.travel_budget or 300
  local label = opts.label or "hop_carry"
  local dir_btn = hop_direction == "LEFT" and "LEFT" or "RIGHT"
  local report = {
    params = {
      stand_frames = stand_frames,
      hop_frames = hop_frames,
      hop_direction = dir_btn,
      unspin_frames = unspin_frames,
      unspin_buttons = unspin_buttons,
      aim_buttons = aim_buttons,
      micro_run_frames = micro_run_frames,
    },
  }
  if stand_frames > 0 then
    session:hold(stand_frames, {}, label .. "_stand")
  end
  report.after_stand = snap(session)
  if micro_run_frames > 0 then
    session:hold(micro_run_frames, { dir_btn, "B" }, label .. "_micro_run")
    report.after_micro_run = snap(session)
  end
  session:hold(hop_frames, { dir_btn, "B", "A" }, label .. "_hop")
  report.after_hop = snap(session)
  if unspin_frames > 0 and unspin_buttons and #unspin_buttons > 0 then
    session:hold(unspin_frames, unspin_buttons, label .. "_unspin")
  end
  report.after_unspin = snap(session)
  report.activate = shinespark.activate_shinespark(session, aim_buttons, {
    hold_frames = activate_hold,
    travel_budget = travel_budget,
    pre_stand_frames = 0,
    label = label .. "_act",
  })
  report.ok = report.activate.ok and true or false
  return report
end

function shinespark.charge_store_activate(session, opts)
  opts = opts or {}
  local direction = opts.direction or "RIGHT"
  local label = opts.label or "spark"
  local charge_mode = opts.charge_mode or "full"
  local report = { label = label, charge_mode = charge_mode }
  report.charge = shinespark.charge_until_boost(session, direction, {
    budget = opts.charge_budget or 500,
    label = label .. "_charge",
    mode = charge_mode,
    region = opts.region or "NTSC",
    store_on_last = opts.store_on_last_magic and (charge_mode == "short" or charge_mode == "stutter"),
  })
  if not report.charge.ok then
    report.ok = false
    report.error = report.charge.error
    return report
  end
  if report.charge.store_armed then
    report.store = {
      ok = true,
      armed = report.charge.boost,
      peak_timer_during_store = tonumber(report.charge.spark_timer) or 0,
      after = snap(session),
      start_pose = tonumber(session.state.pose),
      via = "store_on_last_magic",
    }
  else
    report.store = shinespark.crouch_store(session, {
      max_frames = opts.store_max_frames or shinespark.DEFAULT_STORE_MAX_FRAMES,
      label = label .. "_store",
    })
    if not report.store.ok then
      report.ok = false
      report.error = report.store.error
      return report
    end
  end
  local idle_after_store = opts.idle_after_store or 0
  if idle_after_store > 0 then
    report.window = shinespark.wait_store_window(session, idle_after_store, {
      hold_down = opts.hold_down_idle,
      label = label .. "_idle",
    })
  else
    local armed = report.store.armed or {}
    report.window = {
      requested_frames = 0,
      frames_timer_gt0 = (tonumber(armed.spark_timer) or 0) > 0 and 1 or 0,
      timer_start = armed.spark_timer,
      timer_end = armed.spark_timer,
    }
  end
  report.activate = shinespark.activate_shinespark(session, opts.aim_buttons or { "RIGHT" }, {
    travel_budget = opts.travel_budget or 200,
    label = label .. "_act",
  })
  report.ok = report.activate.ok and true or false
  if not report.ok then
    report.error = "activate did not enter spark pose"
  end
  return report
end

local function btnset(row)
  local s = {}
  local buttons = row.buttons or {}
  for i = 1, #buttons do
    s[string.upper(tostring(buttons[i]))] = true
  end
  return s
end

function shinespark.diagnose_trace(trace)
  if not trace or #trace == 0 then
    return {
      ok = false,
      grade = "EMPTY",
      failures = { "no frames recorded" },
      cues = {},
      peaks = {},
      milestones = {},
    }
  end
  local peak_e, spark_n, down_n = 0, 0, 0
  local first_store, first_spark, start_i = nil, nil, nil
  local wins = {}
  for i = 1, #trace do
    local row = trace[i]
    local e = tonumber(row.speed_echoes) or 0
    local t = tonumber(row.spark_timer) or 0
    local pose = tonumber(row.pose) or 0
    if e > peak_e then
      peak_e = e
    end
    if first_store == nil and t > 0 then
      first_store = tonumber(row.frame) or (i - 1)
    end
    if shinespark.is_spark_pose(pose) then
      spark_n = spark_n + 1
      if first_spark == nil then
        first_spark = tonumber(row.frame) or (i - 1)
      end
    end
    if btnset(row).DOWN then
      down_n = down_n + 1
    end
    if e >= shinespark.ECHOES_FULL and start_i == nil then
      start_i = i
    elseif e < shinespark.ECHOES_FULL and start_i ~= nil then
      wins[#wins + 1] = { start_i, i - 1 }
      start_i = nil
    end
  end
  if start_i ~= nil then
    wins[#wins + 1] = { start_i, #trace }
  end
  local missed, kill_dir = 0, false
  for w = 1, #wins do
    local s, e_i = wins[w][1], wins[w][2]
    local stored = false
    for i = s, e_i do
      if btnset(trace[i]).DOWN then
        stored = true
        break
      end
    end
    if not stored then
      local later = false
      for i = e_i + 1, #trace do
        if btnset(trace[i]).DOWN then
          later = true
          break
        end
      end
      if later then
        missed = missed + 1
        local nxt = e_i + 1 <= #trace and btnset(trace[e_i + 1]) or {}
        if nxt.B and not nxt.RIGHT and not nxt.LEFT then
          kill_dir = true
        end
      end
    end
  end
  local late = first_store == nil and peak_e >= shinespark.ECHOES_FULL and missed > 0 and down_n > 0
  local crouch_walk = false
  if first_store ~= nil and first_spark == nil then
    local a_hold = 0
    for i = 1, #trace do
      local r = trace[i]
      if (tonumber(r.spark_timer) or 0) > 0 then
        local b = btnset(r)
        local pose = tonumber(r.pose) or 0
        if b.A and (b.RIGHT or b.LEFT or b.UP) then
          a_hold = a_hold + 1
        end
        if (pose == 39 or pose == 40 or pose == 53 or pose == 54) and b.A and b.RIGHT then
          crouch_walk = true
        end
      end
    end
    crouch_walk = crouch_walk or a_hold >= 8
  end
  local ok = first_spark ~= nil and spark_n >= 3
  local failures, cues = {}, {}
  if peak_e < shinespark.ECHOES_FULL then
    failures[#failures + 1] = "charge incomplete (peak echoes=" .. tostring(peak_e) .. ", need >=4)"
  elseif first_store == nil then
    failures[#failures + 1] = "never crouch-stored ($0A68 stayed 0)"
    if late then
      failures[#failures + 1] = "late crouch: charged but DOWN never during echoes=4"
      if kill_dir then
        failures[#failures + 1] = "boost killed by releasing LEFT/RIGHT while keeping B"
      end
      cues[#cues + 1] = "CRITICAL: ALSO press DOWN while still holding RIGHT+B"
    end
  elseif first_spark == nil then
    failures[#failures + 1] = "stored but never entered spark pose"
  end
  local grade
  if ok then
    grade = "GREEN"
  elseif peak_e >= shinespark.ECHOES_FULL and first_store ~= nil then
    grade = "YELLOW"
  elseif peak_e >= shinespark.ECHOES_FULL then
    grade = "ORANGE"
  else
    grade = "RED"
  end
  return {
    ok = ok,
    grade = grade,
    failures = failures,
    cues = cues,
    peaks = { echoes = peak_e, spark_travel_frames = spark_n, missed_store_windows = missed },
    milestones = {
      late_store_after_charge_died = late,
      activate_from_crouch_walk = crouch_walk,
    },
  }
end

return shinespark
