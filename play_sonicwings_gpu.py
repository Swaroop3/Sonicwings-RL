# import retro
# from wrappers import ResizeObservation  # Must use gym-based wrapper!

# def make_env():
#     env = retro.make(
#         game='SonicWings-Snes',
#         use_restricted_actions=retro.Actions.DISCRETE
#     )
#     env = ResizeObservation(env)
#     return env

# env = make_env()

# # Manual rendering setup
# env.render()  # Initialize the rendering window

# obs = env.reset()
# while True:
#     action = env.action_space.sample()  # Replace with model prediction
#     obs, reward, done, info = env.step(action)
#     env.render()  # Explicit render call for old Gym versions
#     if done:
#         break
#         # obs = env.reset()

# play_sonicwings_old.py
import retro
import torch as th
import numpy as np
import cv2

class ResizeObservationWrapper:
    def __init__(self, env):
        self.env = env
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(84, 84, 3),
            dtype=np.uint8
        )
        
    def reset(self):
        obs = self.env.reset()
        return self._resize_obs(obs)
    
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return self._resize_obs(obs), rew, done, info
    
    def _resize_obs(self, obs):
        return cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)

# 1. Create environment
env = retro.make(
    game='SonicWings-Snes',
    use_restricted_actions=retro.Actions.DISCRETE
)
env = ResizeObservationWrapper(env)

# 2. Load TorchScript policy
policy = th.jit.load("./ckpt/ppo_sonicwings_policy.pt", map_location='cpu')

# 3. Manual prediction loop
obs = env.reset()
try:
    while True:
        # Convert HWC -> CHW and normalize
        obs_tensor = th.as_tensor(obs.transpose(2, 0, 1)[None]).float() / 255.0
        
        # Get action
        with th.no_grad():
            action_logits = policy(obs_tensor)
        action = action_logits.argmax().item()
        
        # Step environment
        obs, reward, done, info = env.step(action)
        env.render()
        
        if done:
            obs = env.reset()
except KeyboardInterrupt:
    print("Stopped by user")
finally:
    env.close()
