#!/usr/bin/env python3.8
import gym
from gym import wrappers
import numpy as np
from dueling_ddqn_agent import DuelingDDQNAgent
from utils import plot_learning_curve, make_env
import shap
import torch as T
import matplotlib.pyplot as plt
#import sys


#print(sys.version)

from P10_DRL_D3QN.envs import P10_DRL_D3QNEnv

if __name__ == '__main__':
    
    episodeMemory = []
    env = P10_DRL_D3QNEnv()
    best_score = -np.inf
    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    load_checkpoint = False
    n_games = 15
    agent = DuelingDDQNAgent(gamma=0.99, epsilon=1, lr=0.0003,
                         input_dims=(env.observation_space.shape),
                         n_actions=env.action_space.n, mem_size=50000, eps_min=0.05,
                         batch_size=32, replace=10000, eps_dec=1e-4,
                         chkpt_dir='models/', algo='DuelingDDQNAgent',
                         env_name='P10_DRL_D3QNEnv')

    if load_checkpoint:
        agent.load_models()

    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) +'_' \
            + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'

    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    for i in range(n_games):
        done = False
        observation = env.reset()

        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            episodeMemory.append(observation_) 
            score += reward

            if not load_checkpoint:
                agent.store_transition(observation, action,
                                     reward, observation_, int(done))
                agent.learn()
               
            observation = observation_
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print('episode: ', i,'score: ', score,
             ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
            'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)
        if load_checkpoint and n_steps >= 18000:
            break


     
