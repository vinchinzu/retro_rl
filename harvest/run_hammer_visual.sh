#!/bin/bash
# Run bot visually with hammer-only clearing (rocks + stones)
# Press LB+RB+SELECT on controller or A+S+TAB on keyboard to toggle bot on/off
uv run python harvest_bot.py play \
  --state Y1_After_First_Rock_Smash \
  --priority rock,stone \
  --priority-only \
  --scale 4 \
  --autoplay
