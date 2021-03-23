import torch
import random
import numpy as np
from collections import deque
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent():

    def __init__(self):
        self.n_games = 0 # number of games
        self.epsilon = 0 # exploration rate
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # if exceeds memory it calls pop(left)
        self.model = Linear_QNet(18, 156, 6)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def observe(self, environment):
        state = environment.getState()
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
        final_move = [0, 0, 0, 0, 0 ,0]
        if random.randint(0,200) < self.epsilon:
            for joint in range(len(final_move)):
                final_move[joint] = random.randint(0,10)/10
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            #move = torch.argmax(prediction).item()  # item converts tensor to a single value
            final_move = prediction.tolist()
        return final_move