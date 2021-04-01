import torch
import random
import numpy as np
from collections import deque
from models.dqn import DQN_Linear, DQN_Trainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class DQN_Agent():

    def __init__(self):
        self.n_games = 0 # number of games
        self.epsilon = 0 # exploration rate
        self.gamma = 0.9 # discount rate
        self.n_actions = 6 # number of actions
        self.memory = deque(maxlen=MAX_MEMORY) # if exceeds memory it calls pop(left)
        self.model = DQN_Linear(18, 156, self.n_actions)
        self.trainer = DQN_Trainer(self.model, lr=LR, gamma=self.gamma)

    def observe(self, environment):
        state = environment.getState()
        print(state.high)
        #return np.array(state, dtype=int)
        return np.array(state)

    def remember(self, state, action, reward, next_state, done):
        """
        Saves the episode in the memory
        """
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEM is reached

    def train_long_memory(self):
        """
        This is the experience replay
        """
        # If there isn't enough memory to get a full batch, use the whole memory
        if len(self.memory) > BATCH_SIZE:
            # Otherwise get a batch-size worth of samples
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def learn(self, state, action, reward, next_state, done):
        """
        Training the agent for only one step
        """
        self.trainer.train_step(state, action, reward, next_state, done)

    def choose_action(self, state):
        # random moves: tradeoff between exploration and exploitation
        # With probability epsilon take a random action
        self.epsilon = 80 - self.n_games
        final_move = [0]*self.n_actions
        if random.randint(0,200) < self.epsilon:
            for joint in range(len(final_move)):
                final_move[joint] = random.randint(0,10)/10
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            #print(prediction)
            #move = torch.argmax(prediction).item()  # item converts tensor to a single value
            final_move = prediction.tolist()
        return final_move
        
 