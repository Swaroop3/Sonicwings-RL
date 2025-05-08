# convert.py
import retro
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from wrappers import ResizeObservation

def make_env():
    env = retro.make(
        game='SonicWings-Snes',
        use_restricted_actions=retro.Actions.DISCRETE,
        render_mode="none"  # Disable rendering for conversion
    )
    env = ResizeObservation(env)
    return env

# 1. Verify checkpoint path
CHECKPOINT_PATH = "ckpt/ppo_sonicwings_42.5M"  # NO extra .zip
print(f"Loading checkpoint from: {CHECKPOINT_PATH}")

# 2. Load model with explicit device
model = PPO.load(
    CHECKPOINT_PATH,
    env=DummyVecEnv([make_env]),
    device="cpu"  # Force CPU for compatibility
)

# 3. Save policy weights only
model.policy.save("ckpt/ppo_sonicwings_policy.pth")
print("Conversion successful! Policy saved to ckpt/ppo_sonicwings_policy.pth")
