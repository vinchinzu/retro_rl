from retro_harness.env import make_env
from zelda_i.paths import GAME, GAME_DIR
import numpy as np

env = make_env(GAME, "Level4GleeokEnter", GAME_DIR, render_mode="rgb_array")
env.reset()

# Step 100 frames to get into fight
for _ in range(100):
    env.step([0]*9)

# Collect RAM over 60 frames
rams = []
for _ in range(60):
    obs, r, term, trunc, info = env.step([0]*9)
    rams.append(env.get_ram().copy())

# Find bytes that change regularly
diffs = {}
for i in range(len(rams)-1):
    diff = np.where(rams[i] != rams[i+1])[0]
    for addr in diff:
        diffs[addr] = diffs.get(addr, 0) + 1

# Sort by frequency
frequent = sorted(diffs.items(), key=lambda x: x[1], reverse=True)
print("Most frequent changing RAM addresses:")
for addr, count in frequent[:30]:
    vals = [rams[k][addr] for k in range(10)]
    print(f"  0x{addr:04x}: changed {count} times, samples={vals}")

env.close()
