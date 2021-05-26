from gym.envs.registration import register

register(
    id='P10_DRL_GoToBucket-v0',
    entry_point='P10_DRL_GoToBucket.envs:P10_DRL_GoToBucketEnv',
)

