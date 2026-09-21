-- Pure short-charge / stutter plan builders. No session.

local plans = {}

plans.NTSC_MAGIC_DASH_FRAMES = { 25, 50, 70, 85 }
plans.PAL_MAGIC_DASH_FRAMES = { 20, 40, 60, 70 }
plans.NTSC_STUTTER_MIN_PX = 163.1875
plans.PAL_STUTTER_MIN_PX = 157.668
plans.NTSC_STUTTER_FULL_STOP_PX = 164.1875
plans.NTSC_SHORT_CHARGE_FRAMES = 86
plans.PAL_SHORT_CHARGE_FRAMES = 71

function plans.magic_dash_frames(region)
  region = region or "NTSC"
  if region == "PAL" then
    return plans.PAL_MAGIC_DASH_FRAMES
  end
  if region == "NTSC" then
    return plans.NTSC_MAGIC_DASH_FRAMES
  end
  error("unknown region " .. tostring(region) .. "; expected NTSC or PAL")
end

local function extend_mask(out, n, hold_f)
  for _ = 1, n do
    out[#out + 1] = hold_f
  end
end

function plans.stutter_forward_mask(region)
  region = region or "NTSC"
  local segs
  if region == "NTSC" then
    segs = {
      { 3, true }, { 1, false }, { 4, true }, { 1, false }, { 4, true },
      { 1, false }, { 4, true }, { 1, false }, { 2, true }, { 3, true }, { 1, false },
    }
  elseif region == "PAL" then
    segs = {
      { 3, true }, { 1, false }, { 4, true }, { 1, false }, { 3, true },
      { 1, false }, { 2, true }, { 1, false }, { 3, true }, { 1, false },
    }
  else
    error("unknown region " .. tostring(region) .. "; expected NTSC or PAL")
  end
  local out = {}
  for i = 1, #segs do
    extend_mask(out, segs[i][1], segs[i][2])
  end
  return out
end

function plans.stutter_dash_mask(region)
  region = region or "NTSC"
  if region == "NTSC" then
    local mask = {}
    for i = 1, 25 do
      mask[i] = false
    end
    -- frames 21–24 (0-based) → Lua indices 22–25
    mask[22] = true
    mask[23] = true
    mask[24] = true
    mask[25] = true
    return mask
  end
  if region == "PAL" then
    local n = #plans.stutter_forward_mask("PAL")
    local mask = {}
    for i = 1, n do
      mask[i] = false
    end
    return mask
  end
  error("unknown region " .. tostring(region) .. "; expected NTSC or PAL")
end

function plans.short_charge_plan(region, opts)
  region = region or "NTSC"
  opts = opts or {}
  local stutter = opts.stutter
  local store_on_last = opts.store_on_last
  local direction = opts.direction or "RIGHT"
  local dash_button = opts.dash_button or "B"
  local dir_btn = direction == "LEFT" and "LEFT" or "RIGHT"
  local magics = plans.magic_dash_frames(region)
  local last = magics[#magics]
  local magic_set = {}
  for i = 1, #magics do
    magic_set[magics[i]] = true
  end
  local n = last + 1
  local hold_fwd, hold_dash = {}, {}
  for f = 0, last do
    hold_fwd[f] = true
    hold_dash[f] = false
  end
  if stutter then
    local fwd_prefix = plans.stutter_forward_mask(region)
    local dash_prefix = plans.stutter_dash_mask(region)
    if #fwd_prefix ~= magics[1] then
      error(string.format(
        "stutter prefix length %d != first magic %d for %s",
        #fwd_prefix,
        magics[1],
        region
      ))
    end
    for i = 1, #fwd_prefix do
      hold_fwd[i - 1] = fwd_prefix[i]
      hold_dash[i - 1] = dash_prefix[i]
    end
  end
  for i = 1, #magics do
    local f = magics[i]
    hold_fwd[f] = true
    hold_dash[f] = true
  end
  local plan = {}
  for f = 0, last do
    local buttons = {}
    if hold_fwd[f] then
      buttons[#buttons + 1] = dir_btn
    end
    if hold_dash[f] then
      buttons[#buttons + 1] = dash_button
    end
    if store_on_last and f == last then
      buttons[#buttons + 1] = "DOWN"
    end
    plan[#plan + 1] = buttons
  end
  return plan
end

return plans
