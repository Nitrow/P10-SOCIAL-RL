import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    """
    This the network emulating the Q-function
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        """
        Save the model
        """
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    """
    Training and optimization
    """

    def __init__(self, model, lr, gamma):
        self.lr = lr # Learning rate
        self.gamma = gamma # discount factor
        self.model = model # The model we train
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss() # Loss function: mean squared error

    def train_step(self, states, actions, rewards, next_states, done):
        """
        Training function to handle memory training
        """
        # Convert inputs to PyTorch tensors
        state = torch.tensor(states, dtype = torch.float)
        action = torch.tensor(actions, dtype=torch.long)
        next_state = torch.tensor(next_states, dtype=torch.float)
        reward = torch.tensor(rewards, dtype=torch.float)

        # Handle multiple sizes
        if len(state.shape) == 1:
            # (1, x) -> Append one dimention for single inputs to put them into a batch
            state = torch.unsqueeze(state, 0) # Axis 0 means it appends one dimention
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            done = (done, ) # Convert the done into a tuple

        # 1) Predict Q values for current state (one value for each action)
        pred = self.model(state) # Returns a list of values (Q-values for actions)
        # 2) Predict Q value for the new state: R + y*max(next_predicted Q value) 
        #    The new Q-value is where the pred has the highest value
        target = pred.clone()
        
        #    Iterate over the tensors to apply the formula
        for idx in range(len(done)): # Loop through every timestep
            Q_new = reward[idx] # Get the reward gained in step idx
            if not done:        # Only update the q value if we are not done (no next state otherwise)!
                Q_new = Q_new + self.gamma * torch.max(self.model(next_state)) # gamma * predictions for next state

            # The idx's timestep, update the action chosen (argmax)
            # with the new Q-value  (reward gained in the next state)
            # Call it with .item() to treat it as a value and not as a tensor
            #print(torch.argmax(action).item())
            target[idx][torch.argmax(action[idx]).item()] = Q_new

       
        self.optimizer.zero_grad() # empty the gradients (stg we have to remember)
        loss = self.criterion(target, pred) # The target is updated with Q_new, prediction has only Q
        loss.backward() # Apply backpropagation

        self.optimizer.step() # 





