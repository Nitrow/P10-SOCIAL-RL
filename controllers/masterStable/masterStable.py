#!/usr/bin/env python3.8
import gym
import numpy as np
#import P10_RL_env_v01
import P10_DRL_Mark
import torch as th
from sb3_contrib import TQC


from stable_baselines3 import PPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=[dict(pi=[256, 256], vf=[256, 256])])

env = gym.make('P10-DRL-Mark-v0')

# The noise objects for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.001 * np.ones(n_actions))

#model =  PPO("MlpPolicy", env, learning_rate=0.0003, verbose=2)  # , tensorboard_log="/home/asger/P10-XRL/controllers/masterStable/tensorboard")
model = TQC("MlpPolicy", env, top_quantiles_to_drop_per_net=2, verbose=1)
model.learn(total_timesteps=10000000, log_interval=1)
model.save("ppo_p10")
env = model.get_env()

obs = env.reset()
while True: #env.supervisor.step(env.TIME_STEP) != -1:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    
    
    
    
    #action_noise=action_noise,
    #batch_size=1024