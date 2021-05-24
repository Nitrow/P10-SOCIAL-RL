#!/usr/bin/env python3.8

import gym
import numpy as np
from agent_dqn import Agent
from utils import plot_learning_curve
import numpy as np
from datetime import datetime
import math as m
import shutil

from P10_DRL_Lvl3_Grasping_Primitives.envs import P10_DRL_Lvl3_Grasping_Primitives
from P10_DRL_Mark_SingleJoint.envs import P10_DRL_Mark_SingleJointEnv
from P10_DRL_Mark_SimpleEnv.envs import P10_DRL_Mark_SimpleEnv
from P10_RL_env_v01.envs import P10RLEnv

if __name__ == '__main__':
    neurons = 256
    n_games = 1000
    itername=str(n_games) + "_games_OnlyRotation_NoDisplacement_" + str(neurons) + "neurons_1000NewtonGrasp"#_16actions" # alwaysreset
    
    env = P10_DRL_Lvl3_Grasping_Primitives(itername)
    shutil.copy(env.own_path, env.path)
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=env.action_shape, eps_end=0.01, input_dims=[env.state_shape], lr=0.003, chkpt_dir=env.path, fc1_dims=neurons, fc2_dims=neurons)
    scores, eps_history = [], []
                
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
            #print(action, m.degrees(observation[0]))
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
        eps_history.append(agent.epsilon)

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint: agent.save_models()

        print('Episode {}: score {} trailing 100 games avg {} steps {} {} scale {}'.format(i, score, avg_score, steps, env.id, 1))
    # Run the plotting
    #if not load_checkpoint: plot_learning_curve([i+1 for i in range(n_games)], score_history, 'plots/' + env.id)
    plot_learning_curve([i+1 for i in range(n_games)], score_history, env.path + env.id)



