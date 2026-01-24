import stable_retro as retro
import numpy as np
import time

def scan_boss():
    env = retro.make(game="SuperMetroid-Snes", state="BossTorizo")
    env.reset()
    
    # Dump RAM
    ram = env.get_ram()
    
    # Search for 800 (0x0320) which is boss HP
    # Bomb Torizo HP is usually at $0F8E (3982) - Wait, data.json says 3980
    
    potential_slots = []
    for i in range(0x0F00, 0x1100, 2):
        val = ram[i] + (ram[i+1] << 8)
        if val == 800:
            print(f"Found HP 800 at address {i} (0x{i:04X})")
            potential_slots.append(i)
            
    for hp_addr in potential_slots:
        # X is usually HP_addr - 16 bytes?
        # Let's look at nearby values
        x_addr = hp_addr - 18
        y_addr = hp_addr - 14
        x_val = ram[x_addr] + (ram[x_addr+1] << 8)
        y_val = ram[y_addr] + (ram[y_addr+1] << 8)
        print(f"  Slot start? X:{x_val} Y:{y_val} at HP:{hp_addr}")

    env.close()

if __name__ == "__main__":
    scan_boss()
