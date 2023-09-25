from gymnasium.envs.registration import register

register(
    id='Linear-v0',
    entry_point='myenv.envs:LinearEnv',
    max_episode_steps=300,
)

register(
    id='MechArm-v0',
    entry_point='myenv.envs:MechArmEnv',
    max_episode_steps=300,
)
