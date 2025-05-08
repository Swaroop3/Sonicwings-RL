import gym
import numpy as np
from gym import spaces
import cv2
import retro

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(84, 84, 3),  # Maintain channels-last format
            dtype=np.uint8  # Critical for image space recognition
        )

    def observation(self, obs):
        return cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)

class StrategicRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_hit_time = 0
        self.time_penalty_counter = 0
        self.screen_height = 224  # Typical SNES vertical resolution
        self.screen_width = 256   # Typical SNES horizontal resolution

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        custom_reward = 0.0

        # ===== POSITION STRATEGY ===== 
        
        # Position-based rewards (prevent corner camping)
        x_norm = info.get('screen_x', 0) / self.screen_width
        y_norm = info.get('screen_y', 0) / self.screen_height
        
        # Continuous position reward (parabolic curve for lower screen preference)
        vertical_reward = (1 - y_norm)**2  # Max reward at bottom
        horizontal_reward = 1 - abs(x_norm - 0.5)  # Center preference
        vh_balance = 0.9
        position_reward += vertical_reward * vh_balance + horizontal_reward * (1 - vh_balance)
        custom_reward += 1.0 * position_reward

        # Boundary enforcement (prevent edge hugging)
        # if x_norm > 0.9 or x_norm < 0.1 or y_norm > 0.66:
            # custom_reward -= 0.5
        
        # ===== COMBAT STRATEGY =====

        # Engagement incentives
        if info.get('enemy_hit', 0) > 0:
            custom_reward += 1.0 * info['enemy_hit']  # Prioritize hitting over killing
            self.last_hit_time = self.time_penalty_counter
            
        if info.get('enemy_killed', 0) > 0:
            custom_reward += 5.0 * info['enemy_killed']

        # Power-up collection (P items)
        if info.get('powerups_collected', 0) > 0:
            custom_reward += 5.0 * info['powerups_collected']

        # Progressive time penalty for inactivity
        time_since_last_hit = self.time_penalty_counter - self.last_hit_time
        custom_reward -= 0.01 * time_since_last_hit

        # ===== LIFE MANAGEMENT =====

        # Check for life loss (if available in info)
        if 'lives' in info and info['lives'] < self.last_lives:
            custom_reward -= 10.0
            self.last_lives = info['lives']
            
        # Survival time penalty (prevent infinite camping)
        custom_reward -= 0.01  # Small constant time penalty

        # Update trackers
        self.time_penalty_counter += 1
        self.last_x = info.get('screen_x', 0)

        return obs, custom_reward, done, info

    def reset(self, **kwargs):
        self.last_hit_time = 0
        self.time_penalty_counter = 0
        self.last_x = 0
        self.last_lives = 3  # Initial lives count
        return self.env.reset(**kwargs)
