import retro
import numpy as np
from wrappers import ResizeObservation
from stable_baselines3 import PPO

# Create the environment
env = retro.make(
    game='SonicWings-Snes',
    use_restricted_actions=retro.Actions.DISCRETE
)
env = ResizeObservation(env)

# Load the trained model
model = PPO.load("./ckpt/ppo_sonicwings_1000000.zip")

obs = env.reset()
total_reward = 0

do = True

while do:
    # Expand observation to batch dimension (1, h, w, c)
    obs_input = obs[np.newaxis, ...]
    
    # Predict action and extract scalar
    action, _ = model.predict(obs_input, deterministic=True)
    action = int(action[0])  # Convert to integer for Gym Retro
    
    obs, reward, done, info = env.step(action)
    env.render()
    
    total_reward += reward
    if done:
        print(f"Episode finished. Total reward: {total_reward}")
        obs = env.reset()
        total_reward = 0
        do = False
