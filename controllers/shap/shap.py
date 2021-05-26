#!/usr/bin/env python3.8
import gym
from gym import wrappers
import numpy as np
from dueling_ddqn_agent import DuelingDDQNAgent
from utils import plot_learning_curve, make_env
import shap
import torch as T


from P10_DRL_D3QN.envs import P10_DRL_D3QNEnv

if __name__ == '__main__':
    feature_names=['Position\nin m', 'orientation\nin degree', 'color']

    env = P10_DRL_D3QNEnv()
    best_score = -np.inf
    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    load_checkpoint = True
    n_games = 1
    agent = DuelingDDQNAgent(gamma=0.99, epsilon=1.0, lr=0.0001,
                         input_dims=(env.observation_space.shape),
                         n_actions=env.action_space.n, mem_size=50000, eps_min=0.05,
                         batch_size=32, replace=10000, eps_dec=1e-5,
                         chkpt_dir='models/', algo='DuelingDDQNAgent',
                         env_name='P10_DRL_D3QNEnv')

    episodeMemory = []
    for i in range(n_games):
        # Start a new game
        observation = env.reset()
        done = False
        score = 0
        
        while not done:
            # Take an action
            #obsTensor = T.tensor([observation])
            #obsTensor.to(device)
            #agent.Q_eval.to(device)
            #output = agent.Q_eval(obsTensor)
            #print(output.shape)
            #e = shap.DeepExplainer(agent.Q_eval, obsTensor)
            #print(e)
            action = agent.choose_action(observation)
            #print(actions[0].tolist())
            #print(action, m.degrees(observation[0]))
            # Do a step
            observation_, reward, done, info = env.step(action)
            # Add to the memory
            #agent.remember(observation, action, reward, observation_, done)
            episodeMemory.append(observation_)
            #print(memory[0], len(memory), type(memory[0]))
            if not load_checkpoint: agent.learn()
            # Bookkeeping scores
            score += reward
            observation = observation_
           # steps += 1
        #score_history.append(score)
        #avg_score = np.mean(score_history[-100:])
        #eps_history.append(agent.epsilon) 
        
    obsTensor = T.tensor(episodeMemory) 
    testTensor = T.tensor([episodeMemory[-1]])
    e = shap.DeepExplainer(agent.Q_eval, obsTensor)
    shap_values = e.shap_values(testTensor)
    print(shap_values)
    
    
    
    
    
    
    
    
    