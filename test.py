import retro

env = retro.make(
    'Airstriker-Genesis',
    inttype=retro.data.Integrations.ALL  # Force custom integration
)
obs = env.reset()
obs, reward, done, info = env.step(env.action_space.sample())

print("\nAvailable Variables:", list(info.keys()))
# Should now include player_x, player_y, etc.
