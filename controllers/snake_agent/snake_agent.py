#!/usr/bin/env python3
import torch
import random
import numpy as np
from collections import deque
from controller import Robot, Motor
from model import Linear_QNet, QTrainer
from pygame_dqn import SnakeGameAI


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

BLOCK_SIZE = 20

class Agent():

    def __init__(self):
        self.n_games = 0 # number of games
        self.epsilon = 0 # exploration rate
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # if exceeds memory it calls pop(left)
        self.model = Linear_QNet(11, 156, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        state = []
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEM is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff between exploration and exploitation
        # With probability epsilon take a random action
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()  # item converts tensor to a single value
            final_move[move] = 1

        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get action
        final_move = agent.get_action(state_old)

        # Do action
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Train the long memory (replay memory / experience replay)
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score: ', score, 'Record: ', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games




# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)
# get the motor devices
shoulder_lift_motor = robot.getDevice('shoulder_lift_joint')
shoulder_pan_motor = robot.getDevice('shoulder_pan_joint')
elbow_motor = robot.getDevice('elbow_joint')
shoulder_pan_motor = robot.getDevice('wrist_1_joint')
shoulder_pan_motor = robot.getDevice('wrist_2_joint')
shoulder_pan_motor = robot.getDevice('wrist_3_joint')
# set the target position of the motors
shoulder_lift_motor.setPosition(6.2) # Max rotation speed is 6.28319
shoulder_pan_motor.setPosition(6.2)


# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()

    # Process sensor data here.

    # Enter here functions to send actuator commands, like:
    #  motor.setPosition(10.0)
    pass

# Enter here exit cleanup code.
#if __name__ == '__main__':
#    train()