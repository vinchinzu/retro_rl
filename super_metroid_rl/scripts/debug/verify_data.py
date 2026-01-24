import numpy as np
import os

def check_npz(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    try:
        data = np.load(path)
        print(f"Keys: {list(data.keys())}")
        if 'obs' in data:
            print(f"Obs shape: {data['obs'].shape}")
        if 'acts' in data:
            print(f"Acts shape: {data['acts'].shape}")
    except Exception as e:
        print(f"Error loading {path}: {e}")

if __name__ == "__main__":
    check_npz("boss_data/boss_demos.npz")
    check_npz("boss_data/boss_demos_discrete.npz")
