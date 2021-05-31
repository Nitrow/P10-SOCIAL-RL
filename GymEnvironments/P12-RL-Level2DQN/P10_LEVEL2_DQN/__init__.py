from gym.envs.registration import register

register(
    id='P10_LEVEL2_DQN-v0',
    entry_point='P10_LEVEL2_DQN.envs:P10_LEVEL2_DQNEnv',
)

