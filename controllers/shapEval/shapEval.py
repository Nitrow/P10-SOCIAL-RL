#!/usr/bin/env python3.8
import gym
from gym import wrappers
import numpy as np
from dueling_ddqn_agent import DuelingDDQNAgent
from utils import plot_learning_curve, make_env
import shap
import torch as T
#import sys


#print(sys.version)

from P10_DRL_D3QN.envs import P10_DRL_D3QNEnv

if __name__ == '__main__':
    feature_names = ['can 1 x', 'can 1 y', 'can 1 z', 'can 1 rot x', 'can 1 roty', 'can 1 rotz', 'can 1 angle', 'can 1 color x', 'can 1 color y', 'can 1 color z',
                     'can 2 x', 'can 2 y', 'can 2 z', 'can 2 rot x', 'can 2 roty', 'can 2 rotz', 'can 2 angle', 'can 2 color x', 'can 2 color y', 'can 2 color z',      
                     'can 3 x', 'can 3 y', 'can 3 z', 'can 3 rot x', 'can 3 roty', 'can 3 rotz', 'can 3 angle', 'can 3 color x', 'can 3 color y', 'can 3 color z',      
                     'can 4 x', 'can 4 y', 'can 4 z', 'can 4 rot x', 'can 4 roty', 'can 4 rotz', 'can 4 angle', 'can 4 color x', 'can 4 color y', 'can 4 color z',      
                     'can 5 x', 'can 5 y', 'can 5 z', 'can 5 rot x', 'can 5 roty', 'can 5 rotz', 'can 5 angle', 'can 5 color x', 'can 5 color y', 'can 5 color z',      
                     'can 6 x', 'can 6 y', 'can 6 z', 'can 6 rot x', 'can 6 roty', 'can 6 rotz', 'can 6 angle', 'can 6 color x', 'can 6 color y', 'can 6 color z']
    episodeMemory = []
    env = P10_DRL_D3QNEnv()
    best_score = -np.inf
    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    load_checkpoint = False
    n_games = 5
    agent = DuelingDDQNAgent(gamma=0.99, epsilon=1.0, lr=0.0001,
                         input_dims=(env.observation_space.shape),
                         n_actions=env.action_space.n, mem_size=50000, eps_min=0.05,
                         batch_size=32, replace=10000, eps_dec=1e-5,
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


     
     
    
    obsTensor = T.tensor(episodeMemory) 
    testTensor = T.tensor([episodeMemory[-1]])
    e = shap.DeepExplainer(agent.q_next, obsTensor)
    shap_values = e.shap_values(testTensor)
    print(shap_values)
    
    
    
    
    
    
    
    
    