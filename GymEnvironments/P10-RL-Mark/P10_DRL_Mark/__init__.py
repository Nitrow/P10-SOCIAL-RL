from gym.envs.registration import register

register(
    id='P10-DRL-Mark-v0',
    entry_point='P10_DRL_Mark.envs:P10_DRL_Mark_Env',
)

