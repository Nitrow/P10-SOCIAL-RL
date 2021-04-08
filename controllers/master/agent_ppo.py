import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]
        return np.array(self.states), \
               np.array(self.actions), \
               np.array(self.probs), \
               np.array(self.vals), \
               np.array(self.rewards), \
               np.array(self.dones), \
               batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
                 fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo', std = 0.0):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),  # Unpack the input dimension with *
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions)
            #nn.Softmax(dim=-1)  # For Discrete implementation
            )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        #self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.device = T.device("cpu")
        self.to(self.device)
        
        self.log_std = nn.Parameter(T.ones(1, n_actions) * std)

    def forward(self, state):
        # For Discrete implementation
        # Pass the state through the network
        #dist = self.actor(state)
        # Use the dist to draw from a Categorical distribution
        #dist = Categorical(dist)
        
        mu = self.actor(state)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(loc=mu, scale=std)
        

        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, critic_discount=0.5, fc1_dims=256, fc2_dims=256,
                 chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        # Critic takes in a state, outputs a value
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )
        self.critic_discount = critic_discount  # The critic loss seems to be a lot bigger, this scales it down

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        #self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.device = T.device("cpu")
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class PPO_Agent:
    def __init__(self, n_actions, input_dims, action_range=[1],gamma=0.99, alpha=0.00003, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, n_epochs=10, N=2048):
        self.N = N  # The number of steps before an update is performed
        self.gamma = gamma  # Discount factor
        self.policy_clip = policy_clip  #
        self.n_games = 0
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda  # Generalized Advantage Estimation - Lambda parameter
        self.max_action = action_range

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def observe(self, environment):
        state = environment.getState()
        #return np.array(state, dtype=int)
        return np.array(state)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        # Put it into [] as a way of adding a batch-dimension
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        
        #print(state)
        
        dist = self.actor(state)
        
        value = self.critic(state)
<<<<<<< HEAD:controllers/master/agent_ppo.py
        action = dist.sample()
        #print("BEFORE: ", action)
=======
        action = dist.sample().to(self.actor.device)
>>>>>>> de35a1bb681ef65b76ed77c1af7d9e4dc6d5f588:controllers/trainer/agent_ppo.py
        action = T.tanh(action)*T.tensor(self.max_action).to(self.actor.device)
        #print("AFTER: ", action)
        # squeeze removes the batch dimension, item makes the tensor into an integer
        #print(dist.log_prob(action), dist.log_prob(action).sum(1, keepdim=True))
        probs = T.squeeze(dist.log_prob(action).sum(1, keepdim=True)).item()
        #action = T.squeeze(action).item()
        action = action.cpu().detach().numpy()[0]
        value = T.squeeze(value).item()

        return action, probs, value

    def learn(self):
        # Loop through every epoch (how many times to train - learns offline)
        for _ in range(self.n_epochs):
            # Generate a batch from the memory
            state_arr, action_arr, old_prob_arr, vals_arr, \
            reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            # Calculate the advantage
            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1  # No discount
                a_t = 0  # Advantage starts out as 0
                # Go through the rest of the episodes to calculate the advantage
                for k in range(t, len(reward_arr) - 1):
                    # Direct implementation from the paper
                    a_t += discount * (reward_arr[k] - values[k] +
                                       self.gamma * values[k + 1] * (1 - int(dones_arr[k])))
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            # Turn the advantage and values into a tensor
            advantage = T.tensor(advantage).to(self.actor.device)
            values = T.tensor(values).to(self.actor.device)
            # Go through the batches
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)  # Get the new predictions from the states
                # The following line is for Discrete implementation
                new_probs = dist.log_prob(actions).sum(1, keepdim=True)  # Get the probabilities based on the new distribution

                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)
                #print('New: ', new_probs.shape, 'Old: ', old_probs.shape)
                prob_ratio = new_probs.exp() / old_probs.exp()
                # prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip,
                                                 1 + self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                # The expected return (G) is the value + the advantage
                returns = advantage[batch] + values[batch]
                # The loss is the MSE between the the returns and what the critic predicted
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + self.critic.critic_discount * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()

