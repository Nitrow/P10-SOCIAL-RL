#!/usr/bin/env python3.8

import gym
import numpy as np
from agent_dqn import Agent
from utils import plot_learning_curve
import numpy as np
from datetime import datetime
import math as m
import shutil
import shap
import torch as T

from P10_DRL_SHAP_TEST.envs import P10_DRL_SHAP_TEST

if __name__ == '__main__':
    load_checkpoint = False
    chkpt_path = "/home/harumanager/P10-XRL/controllers/shap_test/data/DQN - 2021-05-26 16_43_04P10_DRL_Lvl3_Grasping_Primitives_Test"
    neurons = 128
    n_games = 1
    itername="Test"#str(n_games) + "_games_WidthWiseDisplacement_" + str(neurons) + "neurons_214NewtonGrasp_8actions_axisangle_090gamma" # alwaysreset
    #device = T.device('cuda:0')
    device = T.device('cpu')
    env = P10_DRL_SHAP_TEST(itername)
    shutil.copy(env.own_path, env.path)
    if load_checkpoint:
        agent = Agent(gamma=0.90, epsilon=0.01, batch_size=64, n_actions=env.action_shape, eps_end=0.01, input_dims=[env.state_shape], lr=0.003, chkpt_dir=chkpt_path, fc1_dims=neurons, fc2_dims=neurons)
    else:
        agent = Agent(gamma=0.90, epsilon=1.0, batch_size=64, n_actions=env.action_shape, eps_end=0.01, input_dims=[env.state_shape], lr=0.003, chkpt_dir=env.path, fc1_dims=neurons, fc2_dims=neurons)
    scores, eps_history = [], []
                
    best_score = env.reward_range[0]
    score_history = []
    feature_names = ['Green yaw','Green dist','Green color','Blue yaw','Blue dist','Blue color','Yellow yaw','Yellow dist','Yellow color','Red yaw','Red dist','Red color']
    
    steps = 0
    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

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
            action, actions = agent.choose_action(observation)
            #print(actions[0].tolist())
            #print(action, m.degrees(observation[0]))
            # Do a step
            observation_, reward, done, info = env.step(action)
            # Add to the memory
            agent.remember(observation, action, reward, observation_, done)
            episodeMemory.append(observation_)
            #print(memory[0], len(memory), type(memory[0]))
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

    print("Turning memory into tensor")
    obsTensor = T.tensor(episodeMemory)
    print(obsTensor.shape)
    testTensor = T.tensor([episodeMemory[-1]])
    print(testTensor.shape)
    print("Creating explainer...")
    e = shap.DeepExplainer(agent.Q_eval, obsTensor)
    print("...done...")
    shap_values = e.shap_values(testTensor)
    print(shap_values)



