import stable_retro as retro
import numpy as np

def scan_refined():
    env = retro.make(game="SuperMetroid-Snes", state="BossTorizo")
    env.reset()
    ram = env.get_ram()
    
    hp_addr = 3980
    slot_start = hp_addr - 18
    print(f"Slot Start (approx): {slot_start}")
    
    for i in range(slot_start, slot_start + 64, 2):
        val = ram[i] + (ram[i+1] << 8)
        print(f"  Addr {i} (0x{i:04X}): {val}")
        
    env.close()

if __name__ == "__main__":
    scan_refined()
