import os
import time
import numpy as np
import torch
import stable_retro as retro
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import shutil

# Import EXACT wrappers and constants from train_boss
from train_boss import (
    BossReward, ForceMissiles, StateAugmented, SanitizeAction, 
    DiscreteAction, DISCRETE_ACTIONS, ScriptedOpener, 
    ActionHoldRepeat, FrameStack, _load_hold_lengths
)

def make_eval_env(record_path=None):
    hold_lengths = _load_hold_lengths()
    def hold_sampler(action_idx):
        holds = hold_lengths.get(int(action_idx)) if hold_lengths else None
        if holds: return np.random.choice(holds)
        return np.random.randint(2, 5)

    def _init():
        env = retro.make(
            game="SuperMetroid-Snes",
            state="BossTorizo",
            use_restricted_actions=retro.Actions.ALL,
            record=record_path
        )
        env = SanitizeAction(env)
        env = ForceMissiles(env)
        env = StateAugmented(env)
        env = BossReward(env)
        env = DiscreteAction(env, DISCRETE_ACTIONS)
        env = ScriptedOpener(env, opener_steps=240)
        env = ActionHoldRepeat(env, hold_sampler)
        # Use the FrameStack from train_boss.py
        env = FrameStack(env, n_frames=4)
        env = Monitor(env)
        return env
    return _init

def evaluate(model_path, num_episodes=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    record_dir = "recordings/eval_temp"
    win_dir = "recordings/eval_wins"
    os.makedirs(record_dir, exist_ok=True)
    os.makedirs(win_dir, exist_ok=True)
    
    wins = 0
    total_damage = 0
    
    print(f"Loading model: {model_path}")
    
    for i in range(num_episodes):
        print(f"Starting Episode {i+1}/{num_episodes}...", end=" ", flush=True)
        env_fn = make_eval_env(record_path=record_dir)
        v_env = DummyVecEnv([env_fn])
        
        # Load model with correct env for space verification
        model = PPO.load(model_path, env=v_env, device=device)
        
        obs = v_env.reset()
        done = [False]
        episode_damage = 0
        
        while not done[0]:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, info = v_env.step(action)
            episode_damage = info[0].get('total_damage_dealt', 0)
            
            if done[0]:
                boss_hp = info[0].get('boss_hp', 1) or info[0].get('enemy0_hp', 1)
                is_win = info[0].get('health', 0) > 0 and boss_hp == 0
                
                v_env.close()
                
                # Check recordings
                files = [f for f in os.listdir(record_dir) if f.endswith(".bk2")]
                if files:
                    latest_file = max([os.path.join(record_dir, f) for f in files], key=os.path.getctime)
                    if is_win:
                        wins += 1
                        dest = os.path.join(win_dir, f"win_ep_{i+1}_{int(time.time())}.bk2")
                        shutil.move(latest_file, dest)
                        print(f"WIN! Damage: {episode_damage}")
                    else:
                        print(f"Loss. Damage: {episode_damage}")
                        os.remove(latest_file)
                else:
                    print("Done (no record found).")
                
                total_damage += episode_damage
    
    win_rate = (wins / num_episodes) * 100
    avg_damage = total_damage / num_episodes
    
    print("\n" + "="*30)
    print(f"EVALUATION COMPLETE")
    print(f"Wins: {wins}/{num_episodes} ({win_rate:.2f}%)")
    print(f"Avg Damage: {avg_damage:.2f}")
    print("="*30)

if __name__ == "__main__":
    MODEL_PATH = "models/boss_ppo_checkpoint_30000_steps.zip"
    if os.path.exists(MODEL_PATH):
        evaluate(MODEL_PATH, num_episodes=30)
    else:
        print(f"Model not found at {MODEL_PATH}")
