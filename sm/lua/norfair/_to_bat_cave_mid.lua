-- Bubble → Bat Cave mid-budget loop: save-door runway / lip + climb.
-- Lua 5.1. Session: step / hold / wait_until / span.

local geom = require("skills.geometry")
local runway = require("skills.runway")

local M = {}

function M.run_mid_loop(session, track, start, policy)
  start = start or "launch"
  local label = track.label
  local lip_lo, lip_hi = policy.LIP_X[1], policy.LIP_X[2]
  local lip_y_lo, lip_y_hi = policy.LIP_Y[1], policy.LIP_Y[2]
  local stand_lo, stand_hi = policy.MID_STAND_X[1], policy.MID_STAND_X[2]
  local phase = start
  local mid_i, frames_used = 0, 0
  local height_class = start == "climb"
  local ROOM_ID = policy.ROOM_ID
  if start == "climb" then
    track.launched = true
    track.mid_reached = true
    local st0 = session.state
    if geom.phase_c_usable_right_contact(st0, policy)
        or (st0.room_id == ROOM_ID and st0.samus_x >= policy.RIGHT_SHELF_X - 20
            and st0.samus_y <= policy.PHASE_C_Y_MAX + 40) then
      track.phase_c_hit = true
      height_class = true
    end
  end

  while frames_used < policy.MID_FRAMES do
    frames_used = frames_used + 1
    local state = session.state
    if state.room_id ~= ROOM_ID then
      break
    end
    geom.track_state(session, track, state, policy)
    if geom.phase_d_top_band(state, policy) then
      track.top_reached = true
      break
    end
    if geom.avoid_wrong_door(session, track, state, policy) then
      -- continue
    elseif state.pose == 137 or state.pose == 138 then
      for _ = 1, 10 do
        session:hold(1, {"RIGHT", "B", "A"}, label .. "_mid_kb")
      end
    else
      local x, y = state.samus_x, state.samus_y
      if y <= policy.HEIGHT_CLASS_Y then
        height_class = true
      end
      if x > policy.CAVITY_X_MAX and y > policy.TOP_Y then
        session:hold(1, {"LEFT", "B"}, label .. "_mid_cap")
      else
        mid_i = mid_i + 1
        if phase == "launch" and not track.launched then
          local human_lo, human_hi = policy.SAVE_HUMAN_SEAT_X[1], policy.SAVE_HUMAN_SEAT_X[2]
          local fire_lo, fire_hi = policy.SAVE_RUNWAY_FIRE_X[1], policy.SAVE_RUNWAY_FIRE_X[2]
          local on_runway = geom.on_save_runway(state, policy) and not geom.on_launch_lip(state, policy)
          local seated = on_runway and human_lo <= x and x <= human_hi
            and geom.is_true_ground(state, {poses = policy.TRUE_GROUND})
            and state.pose ~= 137 and state.pose ~= 138
          if seated then
            runway.save_runway_fire_recipe(session, track, {
              policy = policy, y_clear = true, crouch = false,
              arm_pump = policy.SAVE_ARM_PUMP, wj_count = 2, phase_wait = true,
            })
            state = session.state
            if track.min_y <= policy.HEIGHT_CLASS_Y or state.samus_y <= policy.HEIGHT_CLASS_Y then
              height_class = true
            end
            track.launched = true
            phase = "climb"
            mid_i = 0
            if track.top_reached or state.room_id ~= ROOM_ID then
              break
            end
          elseif on_runway and (x > human_hi or x < human_lo or x > fire_hi) then
            if x < fire_lo then
              session:hold(1, {"RIGHT", "B"}, label .. "_save_align_r")
            elseif mid_i < 8 then
              runway.seat_max_left_fire(session, track, {policy = policy})
            elseif geom.on_launch_lip(state, policy) then
              -- fall through to lip
            else
              session:hold(1, {"LEFT", "B"}, label .. "_to_lip_left")
            end
          elseif geom.on_launch_lip(state, policy) then
            if x < 70 then
              session:hold(1, {"RIGHT", "B"}, label .. "_lip_align")
            elseif x > 90 then
              session:hold(1, {"LEFT", "B"}, label .. "_lip_align_l")
            else
              for _ = 1, policy.LIP_CHARGE do
                session:hold(1, {"A"}, label .. "_lip_charge")
              end
              for _ = 1, policy.LIP_SPIN do
                state = session:hold(1, {"RIGHT", "B", "A"}, label .. "_lip_hj")
                geom.track_state(session, track, state, policy)
                if state.room_id ~= ROOM_ID then break end
                if state.samus_y <= policy.HEIGHT_CLASS_Y then height_class = true end
                if geom.phase_d_top_band(state, policy) then
                  track.top_reached = true
                  break
                end
              end
              track.launched = true
              phase = "climb"
              mid_i = 0
              if track.top_reached or state.room_id ~= ROOM_ID then
                break
              end
            end
          elseif geom.on_mid_iso_pin(state, policy) and not geom.on_launch_lip(state, policy)
              and y <= policy.SAVE_RUNWAY_Y[2] then
            local run_lo, run_hi = policy.SAVE_RUNWAY_X[1], policy.SAVE_RUNWAY_X[2]
            if x > run_hi then
              session:hold(1, {"LEFT", "B"}, label .. "_to_save_l")
            elseif x < run_lo then
              session:hold(1, {"RIGHT", "B"}, label .. "_to_save_r")
            else
              session:hold(1, {}, label .. "_save_wait")
            end
          elseif y <= lip_y_lo and stand_lo - 10 <= x and x <= stand_hi + 20 then
            if x > lip_hi then
              session:hold(1, {"LEFT", "B"}, label .. "_drop_left")
            else
              session:hold(1, {}, label .. "_drop_idle")
            end
          elseif x > 160 and y > lip_y_lo then
            session:hold(1, {"LEFT", "B"}, label .. "_to_lip_left")
          else
            local dir_h = (x > 100) and "LEFT" or "RIGHT"
            session:hold(1, {dir_h, "B", "A"}, label .. "_to_lip_air")
            if mid_i > 600 then
              phase = "climb"
              mid_i = 0
            end
          end
        elseif phase == "climb" then
          local grounded = geom.is_true_ground(state, {poses = policy.TRUE_GROUND})
          local phase_c_sticky = track.phase_c_hit or geom.phase_c_usable_right_contact(state, policy)
          if grounded then
            if height_class and geom.on_right_shelf(state, policy) then
              for _ = 1, 12 do session:hold(1, {"A"}, label .. "_shelf_charge") end
              for _ = 1, 56 do
                state = session:hold(1, {"LEFT", "B", "A"}, label .. "_shelf_hj")
                geom.track_state(session, track, state, policy)
                if state.room_id ~= ROOM_ID then break end
                if geom.phase_d_top_band(state, policy) then
                  track.top_reached = true
                  break
                end
              end
              if track.top_reached or state.room_id ~= ROOM_ID then break end
            elseif height_class and y >= policy.FLOOR_RECLIMB_Y and not phase_c_sticky then
              for _ = 1, policy.FLOOR_RECLIMB_CHARGE do
                session:hold(1, {"A"}, label .. "_floor_charge")
              end
              for _ = 1, policy.FLOOR_RECLIMB_SPIN do
                state = session:hold(1, {"RIGHT", "B", "A"}, label .. "_floor_hj")
                geom.track_state(session, track, state, policy)
                if state.room_id ~= ROOM_ID then break end
                if geom.phase_d_top_band(state, policy) then
                  track.top_reached = true
                  break
                end
              end
              if track.top_reached or state.room_id ~= ROOM_ID then break end
            else
              for _ = 1, 10 do session:hold(1, {"A"}, label .. "_climb_charge") end
              local dir_h
              if height_class then
                dir_h = (x < 340) and "RIGHT" or "LEFT"
              elseif x < 70 then
                dir_h = "RIGHT"
              elseif x > 130 then
                dir_h = "LEFT"
              else
                dir_h = (math.floor(mid_i / 40) % 2 == 0) and "RIGHT" or "LEFT"
              end
              if x > policy.CAVITY_X_MAX - 15 then dir_h = "LEFT" end
              for _ = 1, 44 do
                state = session:hold(1, {dir_h, "B", "A"}, label .. "_climb_hj")
                geom.track_state(session, track, state, policy)
                if state.room_id ~= ROOM_ID then break end
                if state.samus_y <= policy.HEIGHT_CLASS_Y then height_class = true end
                if geom.phase_d_top_band(state, policy) then
                  track.top_reached = true
                  break
                end
              end
              if track.top_reached or state.room_id ~= ROOM_ID then break end
            end
          elseif phase_c_sticky and height_class then
            if x > policy.CAVITY_X_MAX - 15 then
              session:hold(1, {"LEFT", "B"}, label .. "_pc_sc")
            else
              local dir_h
              if x < 300 then dir_h = "RIGHT"
              elseif x > 380 then dir_h = "LEFT"
              else dir_h = (x < 365) and "RIGHT" or "LEFT" end
              local wj_phase = mid_i % policy.RIGHT_WJ_PERIOD
              if wj_phase < policy.RIGHT_WJ_INTO then
                session:hold(1, {(x < 375) and "RIGHT" or "LEFT", "B"}, label .. "_pc_into")
              elseif wj_phase < (policy.RIGHT_WJ_INTO + policy.RIGHT_WJ_BOUNCE) then
                session:hold(1, {"LEFT", "A"}, label .. "_pc_wj")
              else
                session:hold(1, {dir_h, "B", "A"}, label .. "_pc_spin")
              end
            end
          elseif height_class and y <= policy.MIDHIGH_Y then
            session:hold(1, {"RIGHT", "B", "A"}, label .. "_ol_cross")
          else
            local dir_h = (x < 70) and "RIGHT" or ((x > 150 and y > 300) and "LEFT" or ((x < 120) and "RIGHT" or "LEFT"))
            if x > policy.CAVITY_X_MAX - 15 then dir_h = "LEFT" end
            session:hold(1, {dir_h, "B", "A"}, label .. "_climb_spin")
          end
        else
          session:hold(1, {"RIGHT", "B", "A"}, label .. "_mid_fallback")
        end
      end
    end
  end
end

M.run_bubble_mid_loop = M.run_mid_loop
return M
