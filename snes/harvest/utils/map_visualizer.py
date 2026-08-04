import numpy as np

def visualize_map(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()[1:] # Skip header
    
    grid = []
    for line in lines:
        parts = line.split(': ')[1].split()
        grid.append([int(p, 16) for p in parts])
    
    grid = np.array(grid)
    
    # Define categories
    ground = {0x00, 0x01, 0x02, 0x03, 0x06, 0x07, 0x08}
    debris = {0x04, 0x05, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10}
    pond = {0xa6, 0xf0, 0xf7, 0xf1}
    border = {0xa1, 0xa8, 0xff}
    
    # Print map
    print("Map Visualization:")
    print("  . = Ground, # = Obstacle, X = Debris, ~ = Pond, B = Border/Building")
    
    print("   " + "".join(f"{x%10}" for x in range(64)))
    for y in range(64):
        row_str = f"{y:02d} "
        for x in range(64):
            val = grid[y, x]
            if val in ground:
                row_str += "."
            elif val in debris:
                row_str += "X"
            elif val in pond:
                row_str += "~"
            elif val in border:
                row_str += "B"
            else:
                row_str += "#"
        print(row_str)

    # List all "unknown" tiles (Obstacles/Buildings)
    print("\nPotential Building/Obstacle Tiles (not ground, debris, pond, or basic border):")
    all_vals = set(grid.flatten())
    categorized = ground | debris | pond | border
    unknown = sorted(list(all_vals - categorized))
    for val in unknown:
        print(f"  0x{val:02x}")

if __name__ == "__main__":
    visualize_map("farm_map_map.txt")
