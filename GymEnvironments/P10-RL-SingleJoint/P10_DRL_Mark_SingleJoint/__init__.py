from gym.envs.registration import register

register(
    id='P10-DRL-Mark-SingleJoint-v0',
    entry_point='P10_DRL_Mark_SingleJoint.envs:P10_DRL_Mark_SingleJointEnv',
)

