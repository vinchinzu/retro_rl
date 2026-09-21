-- Open-loop RLE tables for Early Spazer (guide-shaped).
-- Port of snes/super_metroid/routes/kpdr/spazer/scripts.py.

local M = {}

-- Floor→standing mid — guide floor_to_mid compressed phases (~205f).
M.FLOOR_MID_RLE = {
  {4, {"UP"}},
  {7, {"UP", "X"}},
  {3, {"UP"}},
  {7, {"UP", "X"}},
  {6, {"UP"}},
  {6, {}},
  {8, {"LEFT"}},
  {6, {}},
  {5, {"RIGHT"}},
  {21, {}},
  {1, {"B", "LEFT"}},
  {37, {"B", "LEFT", "A"}},
  {3, {"B", "A"}},
  {1, {}},
  {3, {"RIGHT"}},
  {55, {"RIGHT", "A"}},
  {1, {"A"}},
  {31, {}},
}

-- Top return handoff → floor. Morph+X = bombs on shelf.
M.TOP_MID_RLE = {
  {12, {}},
  {13, {"RIGHT"}},
  {13, {}},
  {2, {"LEFT"}},
  {23, {"B", "LEFT"}},
  {20, {"B", "LEFT", "A"}},
  {1, {"LEFT", "A"}},
  {18, {"LEFT"}},
  {8, {}},
  {5, {"DOWN"}},
  {5, {}},
  {6, {"DOWN"}},
  {16, {}},
  {49, {"LEFT"}},
  {9, {"LEFT", "X"}},
  {87, {"LEFT"}},
  {7, {"LEFT", "X"}},
  {105, {"LEFT"}},
  {80, {}},
  {7, {"X"}},
  {15, {}},
  {3, {"RIGHT"}},
  {3, {"UP", "RIGHT"}},
  {40, {}},
}

-- Solid top node4 → Super door lip — morph tunnel (bombs=X).
M.TOP_DOOR_APPROACH_RLE = {
  {4, {}},
  {5, {"X"}},
  {5, {}},
  {6, {"DOWN"}},
  {6, {}},
  {7, {"DOWN"}},
  {12, {}},
  {9, {"X"}},
  {2, {}},
  {131, {"RIGHT"}},
  {10, {"RIGHT", "X"}},
  {131, {"RIGHT"}},
  {5, {"UP", "RIGHT"}},
  {13, {"RIGHT"}},
  {34, {"RIGHT", "A"}},
  {5, {"RIGHT"}},
  {10, {"RIGHT", "A"}},
  {7, {"RIGHT"}},
  {14, {}},
  {40, {"RIGHT", "B"}},
}

return M
