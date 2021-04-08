#!/usr/bin/env python3.8
import gym
import numpy as np
import P10_DRL_Mark
import torch as th
from sb3_contrib import TQC

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from utils import plot_learning_curve


policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=[dict(pi=[256, 256], vf=[256, 256])])

env = gym.make('P10-DRL-Mark-v0')

# The noise objects for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.001 * np.ones(n_actions))

#modelA2C =  A2C("MlpPolicy", env, learning_rate=0.0003, verbose=0)  # , tensorboard_log="/home/asger/P10-XRL/controllers/masterStable/tensorboard")

#print("Training A2C")
#modelA2C.learn(total_timesteps=10000000, log_interval=1)
#print("...saving A2C")
#modelA2C.save("A2C_p10")
# Training TQC
print("Training TQC")
modelTQC = TQC("MlpPolicy", env, top_quantiles_to_drop_per_net=2, verbose=0)
modelTQC.learn(total_timesteps=10000000, log_interval=1)
print("...saving TQC")
modelTQC.save("tqc_p10")
print("Training PPO")
modelPPO =  PPO("MlpPolicy", env, learning_rate=0.0003, verbose=0)  # , tensorboard_log="/home/asger/P10-XRL/controllers/masterStable/tensorboard")
modelPPO.learn(total_timesteps=10000000, log_interval=1)
print("...saving PPO")
modelPPO.save("ppo_p10")

print("Training second TQC")
model = TQC("MlpPolicy", env, top_quantiles_to_drop_per_net=2, verbose=0)
model.learn(total_timesteps=10000000, log_interval=1)
print("...saving second TQC")
model.save("tqc_p10")
env = model.get_env()

plot_rewards = []
total_reward = 0
obs = env.reset()
while True: #env.supervisor.step(env.TIME_STEP) != -1:
    action, _states = model.predict(obs)
    obs, rewards, dones, _ = env.step(action)
    #print(info)
    env.render()
    total_reward += rewards
    print("Observation:{} \t Reward:{} \t Dones: {} ".format(obs, rewards, dones))
    if dones:
        print('In here')
        plot_rewards.append(total_reward)
        total_reward = 0
        x = [i+1 for i in range(len(plot_rewards))]
        plot_learning_curve(x, plot_rewards, 'plots/ppo-2021-04-07')
    
    
    
    #action_noise=action_noise,
    #batch_size=1024