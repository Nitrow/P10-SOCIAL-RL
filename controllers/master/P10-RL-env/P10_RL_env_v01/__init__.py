from gym.envs.registration import register

register(
    id='P10_RL-v0',
    entry_point='P10_RL_env_v01.envs:P10RLEnv',
)

