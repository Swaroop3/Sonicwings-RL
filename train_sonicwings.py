import retro
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from wrappers import ResizeObservation

def make_env():
    env = retro.make(
        game='SonicWings-Snes',
        use_restricted_actions=retro.Actions.DISCRETE
    )
    env = ResizeObservation(env)  # Output: (84, 84, 3) uint8
    return env

# ===== CONFIGURATION =====
LOAD_CHECKPOINT = True  # Set to True to resume training, False to start fresh
CHECKPOINT_PATH = "ppo_sonicwings_150000.zip"  # Update with your latest checkpoint
# =========================

# Create environment
env = DummyVecEnv([make_env])
env = VecTransposeImage(env)  # Now converts (H, W, C) to (C, H, W)

# Add this to your training script temporarily
print("Observation space:", env.observation_space)

# Initialize model
if LOAD_CHECKPOINT and os.path.exists(CHECKPOINT_PATH):
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    custom_objects = {
        "learning_rate": 0.001,
        "gamma": 0.99,
    }
    model = PPO.load(
        CHECKPOINT_PATH,
        env=env,
        custom_objects=custom_objects
    )
    model = PPO.load(CHECKPOINT_PATH, env=env)
    start_iteration = int(os.path.splitext(CHECKPOINT_PATH)[0].split("_")[-1]) // 50000
else:
    print("Starting new training session")
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        policy_kwargs={"normalize_images": True},
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        tensorboard_log="./sonicwings_tensorboard/"
    )
    start_iteration = 0

# Training loop
TIMESTEPS = 50000
for i in range(start_iteration, 20):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.save(f"ppo_sonicwings_{(i+1)*TIMESTEPS}")
    print(f"Saved checkpoint: ppo_sonicwings_{(i+1)*TIMESTEPS}.zip")

env.close()

