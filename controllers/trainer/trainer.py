#!/usr/bin/env python3.8
from controller import Robot
from environment import Environment
from agent_dqn import DQN_Agent
from agent_sac import SAC_Agent
from agent_ppo import PPO_Agent
from utils import plot_learning_curve

# Create an environment
env = Environment()
#agent = DQN_Agent()
agent = PPO_Agent(input_dims=[18], action_range=[2], n_actions=6, batch_size=256)


timestep = env.TIME_STEP

total_rewards = 0
total_reward = 0
record = -9999
counter = 0

# Helper functions
plot_rewards = []
plot_mean_rewards = []
n_episode_step = 0
N = 256
learn_iters = 0

while env.supervisor.step(timestep) != -1:
    # Observe the environment to get the state
    state = agent.observe(env)
    
    # Decide on a action
    action, prob, val = agent.choose_action(state)
    
    # Execute the action
    reward = env.play_step(action)
    
    # Observe the new environment
    #new_state = agent.observe(env)
    
    # remember
    agent.remember(state, action, prob, val, reward, env.episode_over)
    if n_episode_step % N == 0:
        agent.learn()
        learn_iters += 1
    # train short memory (for single step)
    #agent.learn(state, action, reward, new_state, done)
    #agent.learn()

    # Keep track of the total rewards
    total_reward += reward
    #print("Episode {} \t Step {} \t Reward {}\t Total Reward: {}\t Collision: {}".format(agent.n_games, n_episode_step, reward, int(total_reward), env.collision))

    # Increase episode's step number
    n_episode_step += 1
    
    
    
    if env.episode_over:
        agent.n_games += 1
        n_episode_step = 0
        #agent.train_long_memory()
        
        total_rewards += total_reward
        
        if total_reward > record:
            record = total_reward
            #agent.model.save()
            agent.save_models()
            
        plot_rewards.append(total_reward)
        mean_reward = total_reward / agent.n_games
        plot_mean_rewards.append(mean_reward)   
        print('Episode', agent.n_games, env.episode_over,' \t Total reward: ', total_reward, 'Record: ', record)
        total_reward = 0
        env.reset()  
        x = [i+1 for i in range(len(plot_rewards))]
        plot_learning_curve(x, plot_rewards, 'plots/ppo-2021-04-06')