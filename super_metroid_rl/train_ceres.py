#!/usr/bin/env python3
"""
PPO-based Reinforcement Learning Agent for Super Metroid - Ceres Escape.
Uses stable-retro and stable-baselines3.

The Ceres escape sequence moves DOWN and LEFT (opposite typical RL rewards).
Uses room-aware reward shaping to guide the agent through room transitions.

Usage:
    python train_ceres.py --train --steps 1000000 --headless
    python train_ceres.py --play --load best_model.zip --render
    python train_ceres.py --test-ram  # Test RAM address reading
"""

import os
import sys
import argparse
import time
import numpy as np
import gymnasium as gym

# Add custom integrations path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INTEGRATION_PATH = os.path.join(SCRIPT_DIR, "custom_integrations")

try:
    import stable_retro as retro
    retro.data.Integrations.add_custom_path(INTEGRATION_PATH)
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.atari_wrappers import WarpFrame
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install: pip install stable-retro stable-baselines3")
    sys.exit(1)

os.environ['SDL_VIDEODRIVER'] = 'x11'


# Ceres room IDs and their order in the escape sequence
# These need to be verified by running the game and observing room_id values
CERES_ROOMS = {
    # Room ID -> (name, next_room_direction)
    # Direction: 'right', 'down', 'left' indicates which way leads to progress
}


class SNESDiscretizer(gym.ActionWrapper):
    """
    Discretize SNES MultiBinary(12) actions into useful combinations.
    SNES: [B, Y, Select, Start, Up, Down, Left, Right, A, X, L, R]
    """
    def __init__(self, env):
        super().__init__(env)
        # Define useful actions for Super Metroid escape
        self._actions = [
            [0,0,0,0,0,0,0,0,0,0,0,0],  # 0. NOOP
            [0,0,0,0,0,0,0,1,0,0,0,0],  # 1. Right
            [0,0,0,0,0,0,1,0,0,0,0,0],  # 2. Left
            [0,0,0,0,0,1,0,0,0,0,0,0],  # 3. Down
            [1,0,0,0,0,0,0,1,0,0,0,0],  # 4. Right + B (Run right)
            [1,0,0,0,0,0,1,0,0,0,0,0],  # 5. Left + B (Run left)
            [0,0,0,0,0,0,0,1,1,0,0,0],  # 6. Right + A (Jump right)
            [0,0,0,0,0,0,1,0,1,0,0,0],  # 7. Left + A (Jump left)
            [1,0,0,0,0,0,0,1,1,0,0,0],  # 8. Right + B + A (Running jump right)
            [1,0,0,0,0,0,1,0,1,0,0,0],  # 9. Left + B + A (Running jump left)
            [0,0,0,0,0,0,0,0,1,0,0,0],  # 10. A (Jump vertical)
            [0,1,0,0,0,0,0,0,0,0,0,0],  # 11. Y (Shoot)
            [0,0,0,0,0,1,0,0,1,0,0,0],  # 12. Down + A (Morph/Drop)
        ]
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, action):
        return self._actions[action]


class CeresReward(gym.Wrapper):
    """
    Custom reward wrapper for Ceres Station Escape.

    Rewards:
    - Room transitions (major progress)
    - Position changes within room (minor progress)
    - Time bonus (faster is better)
    - Stagnation penalty (penalize standing still)
    """
    def __init__(self, env, render_pygame=False):
        super().__init__(env)
        self.render_pygame = render_pygame
        self.screen = None
        self.clock = None

        # Tracking state
        self.prev_room_id = None
        self.rooms_visited = set()
        self.prev_x = 0
        self.prev_y = 0
        self.stuck_frames = 0
        self.total_frames = 0
        self.episode_reward = 0

        if self.render_pygame:
            import pygame
            pygame.init()
            self.screen = pygame.display.set_mode((512, 448))  # 2x SNES resolution
            pygame.display.set_caption("Super Metroid - Ceres Escape Training")
            self.clock = pygame.time.Clock()

    def reset(self, **kwargs):
        self.prev_room_id = None
        self.rooms_visited = set()
        self.prev_x = 0
        self.prev_y = 0
        self.stuck_frames = 0
        self.total_frames = 0
        self.episode_reward = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_frames += 1

        # Extract state from info (populated by data.json)
        x = info.get('samus_x', 0)
        y = info.get('samus_y', 0)
        room_id = info.get('room_id', 0)
        health = info.get('health', 0)
        game_state = info.get('game_state', 0)

        shaped_reward = 0.0

        # Room transition reward (major milestone)
        if self.prev_room_id is not None and room_id != self.prev_room_id:
            if room_id not in self.rooms_visited:
                shaped_reward += 100.0  # New room bonus
                self.rooms_visited.add(room_id)
            else:
                shaped_reward += 10.0  # Revisited room (smaller bonus)

        # Initialize room tracking
        if self.prev_room_id is None:
            self.prev_room_id = room_id
            self.rooms_visited.add(room_id)

        # Movement reward - encourage any movement
        dx = abs(x - self.prev_x)
        dy = abs(y - self.prev_y)
        movement = dx + dy

        if movement > 0:
            shaped_reward += 0.1  # Small reward for moving
            self.stuck_frames = 0
        else:
            self.stuck_frames += 1

        # Stagnation penalty
        if self.stuck_frames > 60:  # Stuck for 1 second
            shaped_reward -= 0.5
        if self.stuck_frames > 180:  # Stuck for 3 seconds
            shaped_reward -= 2.0

        # Health penalty (taking damage is bad)
        # Note: In Ceres there shouldn't be much damage, but track anyway

        # Update state
        self.prev_room_id = room_id
        self.prev_x = x
        self.prev_y = y

        # Add shaped reward to environment reward
        reward += shaped_reward
        self.episode_reward += reward

        # Add tracking info
        info['shaped_reward'] = shaped_reward
        info['rooms_visited'] = len(self.rooms_visited)
        info['stuck_frames'] = self.stuck_frames
        info['episode_reward'] = self.episode_reward

        # Handle pygame rendering
        if self.render_pygame:
            self._render_pygame(obs)

        return obs, reward, terminated, truncated, info

    def _render_pygame(self, obs):
        import pygame
        surf = pygame.surfarray.make_surface(obs.swapaxes(0, 1))
        self.screen.blit(pygame.transform.scale(surf, self.screen.get_size()), (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)


class StopOnRoomCountCallback(BaseCallback):
    """Stop training after visiting N unique rooms."""
    def __init__(self, n_rooms: int, verbose=1):
        super().__init__(verbose)
        self.n_rooms = n_rooms
        self.best_rooms = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            rooms = info.get("rooms_visited", 0)
            if rooms > self.best_rooms:
                self.best_rooms = rooms
                if self.verbose > 0:
                    print(f"New best: {rooms} rooms visited!")
            if rooms >= self.n_rooms:
                if self.verbose > 0:
                    print(f"Reached {self.n_rooms} rooms! Stopping training.")
                return False
        return True


class SaveOnBestRoomsCallback(BaseCallback):
    """Save model when agent visits more rooms than before."""
    def __init__(self, save_path: str, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.best_rooms = 0

    def _init_callback(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            rooms = info.get("rooms_visited", 0)
            if rooms > self.best_rooms:
                self.best_rooms = rooms
                path = os.path.join(self.save_path, f"best_model_{rooms}_rooms")
                self.model.save(path)
                if self.verbose > 0:
                    print(f"Saved new best model with {rooms} rooms to {path}")
        return True


def make_env(game, state, render_pygame=False, log_dir=None, record_dir=None):
    """Factory function for creating wrapped environments."""
    def _init():
        env = retro.make(
            game=game,
            state=state,
            render_mode='rgb_array',
            record=record_dir,
            inttype=retro.data.Integrations.ALL
        )
        env = CeresReward(env, render_pygame=render_pygame)
        env = SNESDiscretizer(env)
        if log_dir:
            env = Monitor(env, log_dir)
        else:
            env = Monitor(env)
        env = WarpFrame(env)  # Grayscale + resize to 84x84
        return env
    return _init


def linear_schedule(initial_value: float):
    """Linear learning rate schedule."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def test_ram_addresses():
    """Test that RAM addresses are being read correctly."""
    print("Testing RAM address reading...")
    print(f"Integration path: {INTEGRATION_PATH}")

    env = retro.make(
        game="SuperMetroid-Snes",
        state="Start",
        render_mode='rgb_array',
        inttype=retro.data.Integrations.ALL
    )

    obs, info = env.reset()
    print(f"\nInitial state info:")
    for key, value in sorted(info.items()):
        print(f"  {key}: {value}")

    print("\nRunning 120 frames (2 seconds)...")
    for i in range(120):
        # Just stand still
        action = [0] * 12
        obs, reward, terminated, truncated, info = env.step(action)

        if i % 30 == 0:
            print(f"\nFrame {i}:")
            print(f"  Position: ({info.get('samus_x', 'N/A')}, {info.get('samus_y', 'N/A')})")
            print(f"  Room ID: {info.get('room_id', 'N/A'):#06x}" if info.get('room_id') else "  Room ID: N/A")
            print(f"  Health: {info.get('health', 'N/A')}")
            print(f"  Game State: {info.get('game_state', 'N/A')}")
            print(f"  Velocity: ({info.get('velocity_x', 'N/A')}, {info.get('velocity_y', 'N/A')})")

    env.close()
    print("\nRAM test complete!")


def main():
    parser = argparse.ArgumentParser(description='Train PPO agent on Super Metroid Ceres Escape')
    parser.add_argument('--train', action='store_true', help='Train the agent')
    parser.add_argument('--play', action='store_true', help='Play using trained model')
    parser.add_argument('--test-ram', action='store_true', help='Test RAM address reading')
    parser.add_argument('--steps', type=int, default=2000000, help='Max training steps')
    parser.add_argument('--load', type=str, default=None, help='Model path to load')
    parser.add_argument('--render', action='store_true', help='Render to screen')
    parser.add_argument('--headless', action='store_true', help='Run without visualization')
    parser.add_argument('--record-dir', type=str, default=None, help='Directory to save recordings')

    args = parser.parse_args()

    if args.test_ram:
        test_ram_addresses()
        return

    game = "SuperMetroid-Snes"
    state = "Start"

    save_dir = os.path.join(SCRIPT_DIR, "models")
    log_dir = os.path.join(SCRIPT_DIR, "logs")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if args.train:
        print(f"Starting training on {game} - Ceres Escape...")
        print(f"Target: Navigate through all Ceres rooms and escape")

        render = args.render and not args.headless
        record = os.path.join(SCRIPT_DIR, "recordings") if not args.headless else None
        if record:
            os.makedirs(record, exist_ok=True)

        env = DummyVecEnv([make_env(
            game, state,
            render_pygame=render,
            log_dir=os.path.join(log_dir, "monitor"),
            record_dir=record
        )])

        # PPO Hyperparameters
        initial_lr = 2.5e-4
        initial_clip = 0.1
        ent_coef = 0.05  # Higher entropy for exploration

        if args.load and os.path.exists(args.load):
            print(f"Loading model from {args.load}...")
            model = PPO.load(args.load, env=env, tensorboard_log=log_dir)
            model.learning_rate = linear_schedule(initial_lr)
            model.clip_range = linear_schedule(initial_clip)
            model.ent_coef = ent_coef
        else:
            print("Creating new PPO model...")
            model = PPO(
                "CnnPolicy",
                env,
                verbose=1,
                tensorboard_log=log_dir,
                learning_rate=linear_schedule(initial_lr),
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=linear_schedule(initial_clip),
                ent_coef=ent_coef,
            )

        # Callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=100000,
            save_path=save_dir,
            name_prefix="ceres_ppo"
        )

        room_callback = StopOnRoomCountCallback(n_rooms=10)  # Ceres has ~5-7 rooms
        best_callback = SaveOnBestRoomsCallback(save_path=save_dir)

        callback_list = CallbackList([checkpoint_callback, room_callback, best_callback])

        try:
            model.learn(total_timesteps=args.steps, callback=callback_list, progress_bar=True)
            model.save(os.path.join(save_dir, "final_model"))
            print("Training finished!")
        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving model...")
            model.save(os.path.join(save_dir, "interrupted_model"))
        finally:
            env.close()

    elif args.play:
        if not args.load:
            # Find best model
            possible = [
                os.path.join(save_dir, "best_model.zip"),
                os.path.join(save_dir, "final_model.zip"),
            ]
            for p in possible:
                if os.path.exists(p):
                    args.load = p
                    break

        if not args.load or not os.path.exists(args.load):
            print("No model found to play!")
            return

        print(f"Playing with model: {args.load}")

        env = make_env(game, state, render_pygame=args.render, record_dir=args.record_dir)()
        model = PPO.load(args.load)

        obs, _ = env.reset()
        done = False
        total_reward = 0

        try:
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated

                if args.render:
                    time.sleep(0.016)  # ~60fps

            print(f"Episode finished. Total Reward: {total_reward:.2f}")
            print(f"Rooms visited: {info.get('rooms_visited', 0)}")
        except KeyboardInterrupt:
            pass
        finally:
            env.close()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
