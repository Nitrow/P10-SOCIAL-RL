#!/usr/bin/env python3.8

import gym
import numpy as np
from sac_torch import Agent
from utils import plot_learning_curve
import numpy as np
from datetime import datetime

from P10_DRL_Mark.envs import P10_DRL_Mark_Env
from P10_DRL_Mark_SingleJoint.envs import P10_DRL_Mark_SingleJointEnv
#from P10_DRL_GoToBucket.envs import P10_DRL_GoToBucketEnv
#from P10_DRL_Mark_SimpleEnv.envs import P10_DRL_Mark_SimpleEnv
#from P10_RL_env_v01.envs import P10RLEnv

if __name__ == '__main__':

    n_games = 5000
    dt = 32
    env = P10_DRL_Mark_SingleJointEnv()
    #env = P10_DRL_Mark_SimpleEnv()
    #env = P10_DRL_Mark_Env()
    # agent = Agent(alpha=0.0003, beta=0.0003, reward_scale=2, env_id=env.id, 
                # input_dims=env.observation_space.shape, tau=0.005,
                # env=env, batch_size=256, layer1_size=256, layer2_size=256,
                # n_actions=env.action_space.shape[0])
                
    agent = Agent(alpha=0.0003, beta=0.0003, reward_scale=2, env_id=env.id, 
                input_dims=env.observation_space.shape, tau=0.005,
                env=env, batch_size=512, layer1_size=512, layer2_size=512,
                n_actions=env.action_space.shape[0])
                
    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False
    steps = 0
    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    for i in range(n_games):
        # Start a new game
        observation = env.reset()
        done = False
        score = 0
        
        while not done:
            # Take an action
            action = agent.choose_action(observation)
            # Do a step
            observation_, reward, done, info = env.step(action)
            # Add to the memory
            agent.remember(observation, action, reward, observation_, done)
            
            if not load_checkpoint: agent.learn()
            # Bookkeeping scores
            score += reward
            observation = observation_
            steps += 1
            
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint: agent.save_models()

        print('Episode {}: score {} trailing 100 games avg {} steps {} {} scale {}'.format(i, score, avg_score, steps, env.id, agent.scale))
    # Run the plotting
    if not load_checkpoint: plot_learning_curve([i+1 for i in range(n_games)], score_history, 'plots/' + env.id)



