import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, states, actions, rewards, next_states, done):
        state = torch.tensor(states, dtype = torch.float)
        action = torch.tensor(actions, dtype=torch.long)
        next_state = torch.tensor(next_states, dtype=torch.float)
        reward = torch.tensor(rewards, dtype=torch.float)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            done = (done, )

        # 1) Predict Q values for current state
        pred = self.model(state) # Returns a list of three values (Q-values for actions)
        target = pred.clone()

        for idx in range(len(done)): # Loop through every timestep
            Q_new = reward[idx] # Get the reward gained
            if not done:
                Q_new = Q_new + self.gamma * torch.max(self.model(next_state)) # gamma * predictions for next state

            # The idx's timestep, update the action chosen with the new Q-value (reward gained in the next state)
            target[idx][torch.argmax(action).item()] = Q_new

        # 2) R + y*max(next_predicted Q value)
        self.optimizer.zero_grad() # empty the gradients (stg we have to remember)
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()





