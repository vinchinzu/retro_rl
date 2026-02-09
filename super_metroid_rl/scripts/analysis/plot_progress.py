import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_DIR, "boss_data")
OUTPUT_FILE = os.path.join(PROJECT_DIR, "training_progress.png")

def plot_logs():
    # Find all monitor files
    files = glob.glob(os.path.join(DATA_DIR, "boss_monitor_*.monitor.csv"))
    if not files:
        print("No monitor files found.")
        return

    # Sort by creation time/filename (timestamp in name helps)
    # Filename format: boss_monitor_{timestamp}.csv.monitor.csv
    files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    print(f"Found {len(files)} log files.")
    
    dfs = []
    cumulative_steps = 0
    
    for f in files:
        print(f"Reading {os.path.basename(f)}...")
        try:
            # Skip first line if it's metadata (starts with #)
            with open(f, 'r') as fp:
                header = fp.readline()
                if not header.startswith('#'):
                     # If no metadata, reset seek
                     fp.seek(0)
                
                # Use pandas to read, handling potential header issues
                # Monitor CSVs usually have header row after metadata
                df = pd.read_csv(fp)
                
                # Add cumulative timesteps for continuous x-axis
                if 'l' in df.columns: # 'l' is usually episode length in steps
                    df['total_steps'] = df['l'].cumsum() + cumulative_steps
                    cumulative_steps = df['total_steps'].iloc[-1]
                
                dfs.append(df)
        except Exception as e:
            print(f"Skipping {f}: {e}")
            
    if not dfs:
        print("No data loaded.")
        return

    full_df = pd.concat(dfs, ignore_index=True)
    
    # Calculate Moving Average
    window_size = 50
    full_df['reward_ma'] = full_df['r'].rolling(window=window_size).mean()
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(full_df.index, full_df['r'], alpha=0.3, color='gray', label='Episode Reward')
    plt.plot(full_df.index, full_df['reward_ma'], color='blue', linewidth=2, label=f'{window_size}-Ep Moving Avg')
    
    plt.title(f"Boss Training Progress ({len(full_df)} Episodes)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add a horizontal line for "Win" threshold (approx)
    # Based on our checks: > 25000 is usually a win
    plt.axhline(y=25000, color='green', linestyle='--', alpha=0.5, label='Win Threshold (~25k)')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE)
    print(f"Plot saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    plot_logs()
