import gymnasium as gym
import numpy as np

class MetroidProgressReward(gym.Wrapper):
    """
    Custom reward wrapper for Super Metroid.

    Rewards:
    - Depth: Reward for vertical progress (down during DESCENT, up during RETURN)
    - Morph Ball: Huge bonus for acquiring the Morph Ball item
    - Room transitions: Reward for entering new rooms
    - Stagnation: Penalty for staying still

    The wrapper automatically switches from DESCENT to RETURN phase
    when Morph Ball is acquired.
    """
    def __init__(self, env):
        super().__init__(env)
        self.max_y = 0
        self.min_y = float('inf')  # For tracking upward progress during RETURN
        self.has_morph_ball = False
        self.stuck_counter = 0
        self.prev_x = 0
        self.prev_y = 0
        self.prev_room = 0
        self.visited_rooms = set()
        self.episode_reward = 0

    def reset(self, **kwargs):
        self.max_y = 0
        self.min_y = float('inf')
        self.has_morph_ball = False
        self.stuck_counter = 0
        self.prev_x = 0
        self.prev_y = 0
        self.prev_room = 0
        self.visited_rooms = set()
        self.episode_reward = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # RAM values from data.json
        x = info.get('samus_x', 0)
        y = info.get('samus_y', 0)
        room_id = info.get('room_id', 0)
        items = info.get('items', 0)

        shaped_reward = 0.0

        # Check Morph Ball status (bit 0 of items)
        has_morph = (items & 0x1) != 0

        # === MORPH BALL BONUS (One-time) ===
        if has_morph and not self.has_morph_ball:
            shaped_reward += 1000.0
            self.has_morph_ball = True
            self.min_y = y  # Reset min_y for RETURN phase tracking
            print("REWARD: Morph Ball Acquired! (+1000)")

        # === VERTICAL PROGRESS ===
        if not self.has_morph_ball:
            # DESCENT phase: reward going DOWN (increasing Y)
            if y > self.max_y:
                shaped_reward += (y - self.max_y) * 0.1
                self.max_y = y
        else:
            # RETURN phase: reward going UP (decreasing Y)
            if y < self.min_y:
                shaped_reward += (self.min_y - y) * 0.1
                self.min_y = y

        # === ROOM TRANSITION BONUS ===
        if room_id != self.prev_room:
            if room_id not in self.visited_rooms:
                # First visit to this room
                shaped_reward += 50.0
                self.visited_rooms.add(room_id)
            else:
                # Revisiting during return journey
                if self.has_morph_ball:
                    shaped_reward += 10.0  # Smaller bonus for backtracking
            self.prev_room = room_id

        # === TORIZO ROOM BONUS ===
        TORIZO_ROOM = 0x9804
        if room_id == TORIZO_ROOM and self.has_morph_ball:
            shaped_reward += 500.0
            print("REWARD: Reached Torizo Room! (+500)")

        # === MOVEMENT / STAGNATION ===
        if x == self.prev_x and y == self.prev_y:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            shaped_reward += 0.01  # Breadcrumb reward

        if self.stuck_counter > 60:
            shaped_reward -= 0.1
        if self.stuck_counter > 300:
            shaped_reward -= 1.0

        self.prev_x = x
        self.prev_y = y

        reward += shaped_reward
        self.episode_reward += reward

        info['shaped_reward'] = shaped_reward
        info['total_shaped_reward'] = self.episode_reward
        info['has_morph_ball'] = self.has_morph_ball

        return obs, reward, terminated, truncated, info
