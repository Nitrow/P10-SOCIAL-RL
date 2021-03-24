#!/usr/bin/env python3.8
from controller import Robot
from environment import Environment
from agent import Agent

# Create an environment
env = Environment()
agent = Agent()

timestep = env.TIME_STEP

total_rewards = 0
total_reward = 0
record = 0
counter = 0
actions = [0.1, 0, 0, 0, 0.1, 0.1]

# Helper functions
plot_rewards = []
plot_mean_rewards = []

while env.supervisor.step(timestep) != -1:
    # Observe the environment to get the state
    state = agent.observe(env)
    
    # Decide on a action
    action = agent.get_action(state)
    
    # Execute the action
    reward, done = env.play_step(action)
    
    # Observe the new environment
    new_state = agent.observe(env)
    
    # train short memory (for single step)
    agent.train_short_memory(state, action, reward, new_state, done)

    # remember
    agent.remember(state, action, reward, new_state, done)
    
    # Keep track of the total rewards
    total_reward += reward
    print(reward)
    
    
    if done:
        env.reset()
        agent.n_games += 1
        agent.train_long_memory()
        
        total_rewards += total_reward
        
        if total_reward > record:
            record = total_reward
            agent.model.save()
        
        plot_rewards.append(total_reward)
        mean_reward = total_reward / agent.n_games
        plot_mean_rewards.append(mean_reward)
        
        total_reward = 0
        print('Game', agent.n_games, 'Score: ', total_reward, 'Record: ', record)   
