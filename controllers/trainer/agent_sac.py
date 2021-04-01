import os
import torch as T
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from buffer import ReplayBuffer
#from models.sac import ActorNetwork, CriticNetwork, ValueNetwork  # Implemented in the same class 


class SAC_Agent():
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[8], action_range=[2, 2, 2, 2, 2, 2],
                 gamma=0.99, n_actions=2, max_size=1000000, tau=0.005,
                 layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2):
        self.gamma = gamma
        self.n_games = 0
        self.tau = tau  # The Target value network will be "soft-copied", parameters are de-tuned by tau amount
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor = ActorNetwork(alpha, input_dims, n_actions=n_actions,
                                  name='actor', max_action=action_range)
        self.critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                                      name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                                      name='critic_2')
        self.value = ValueNetwork(beta, input_dims, name='value')
        self.target_value = ValueNetwork(beta, input_dims, name='target_value')

        self.scale = reward_scale
        self.update_network_parameters(tau=1)  # Set the values of the target network

    def choose_action(self, observation):
        # Turn the observation into a PyTorch tensor
        state = T.Tensor([observation]).to(self.actor.device)
        # Sample actions from, logprobs are not needed so it's left blank
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        #print(actions.shape, actions, probs.shape, probs)
        # It's a cuda tensor, we have to detach it, turn it into a numpy array and return the 0th element
        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        """
        Interface function between the memory and the agent
        """
        self.memory.store_transition(state, action, reward, new_state, done)

    def observe(self, environment):
        state = environment.getState()
        #return np.array(state, dtype=int)
        return np.array(state)

    def update_network_parameters(self, tau=None):
        """
        "Book-keeping function"
        """
        # At the beginning we want to copy the value network exactly, otherwise it's a soft copy
        if tau is None:
            tau = self.tau

        # Create a copy of the parameters
        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        # Every value from the value network is copied to the target_value
        for name in value_state_dict:
            value_state_dict[name] = tau * value_state_dict[name].clone() + \
                                     (1 - tau) * target_value_state_dict[name].clone()
        # Load the state dictionary
        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self):
        # See if we have enough experience in the memory
        if self.memory.mem_cntr < self.batch_size:
            return
        # Learn by sampling the buffer
        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        # Turn the memory sections into tensors
        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)

        # Value of the state and new state (view(-1)) clamps it along the batch dimension (predictions)
        value = self.value(state).flatten()
        value_ = self.target_value(state_).flatten()
        # In all the terminal states the values should be set to 0
        value_[done] = 0.0

        # Get the actions and log probabilites according to the new policy
        # (and not according to the old policy from the buffer)
        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.flatten()
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        # Get two Q-values from the two critics and take the lowest one
        # This is done to get over the "overestimation-bias"
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.flatten()

        # Calculate the loss and backpropagate
        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        # Retaining the graph preserves the data that is present due to the coupling of the neural networks
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        # Calculate the actor network loss
        # feed forward to get the actions and log probs
        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.flatten()
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.flatten()

        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        # Critic loss for both critics
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        # Scaling factor includes the entropy in the loss function (encourage exploration)
        q_hat = self.scale * reward + self.gamma * value_
        # Using the old buffer we calculate the q values
        q1_old_policy = self.critic_1.forward(state, action).flatten()
        q2_old_policy = self.critic_2.forward(state, action).flatten()
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)
        # Take the sum of the two critic losses and backpropagate
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()
        # Handle the update
        self.update_network_parameters()

class CriticNetwork(nn.Module):
    """
    Evaluates the quality of the action taken,
    Takes as input the state and action pairs
    """
    def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256,
            name='critic', chkpt_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims # Number of input dimensions from the environment
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name # For model checkpointing
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        # The critic evaluates the state-action pairs, so the actions are needed as an input
        self.fc1 = nn.Linear(self.input_dims[0]+n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # Returns a single q-value
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        # Take advantage of
        self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, state, action):
        # The input to the critic is the selected action and the state
        action_value = T.cat([state, action], dim=1)
        action_value = self.fc1(action_value)
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class ValueNetwork(nn.Module):
    """
    Estimates the value of a state or set of states,
    regardless of the action taken
    """
    def __init__(self, beta, input_dims, fc1_dims=256, fc2_dims=256,
            name='value', chkpt_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, fc2_dims)
        # Returns a single value for the state we ended up in
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)

        self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)

        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    """
    Outputs an action given the state
    """
    def __init__(self, alpha, input_dims, max_action, fc1_dims=256,
            fc2_dims=256, n_actions=2, name='actor', chkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.max_action = max_action # The output should be in a range of actions
        self.reparam_noise = 1e-6 # Safeguards for log of 0

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # MU: mean of the distribution for the policy for each action component
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        # SIGMA: Standard deviation
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, state):
        """
        Takes the states as input
        """
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        #print(prob.shape,state.shape, mu.shape, '\t', mu)
        sigma = self.sigma(prob)
        # The standard distribution is clamped between a min and max value,
        # here it's a tiny bit higher than 0, as 0 would give errors
        # Clamp is faster than sigmoid function
        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        """
        Need a normal distribution of the continuous action space
        """
        mu, sigma = self.forward(state)
        
        # Distribution of actions, defined by the mean and standard deviation
        probabilities = Normal(loc=mu, scale=sigma)
            
        if reparameterize:
            # Adding some noise (exploration factor) to the distribution
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()
        # Tanh gives an action between -1 and 1, which is scaled by the "max_axtion"
        action = T.tanh(actions)*T.tensor(self.max_action).to(self.device)
        # Log prob for the loss function
        log_probs = probabilities.log_prob(actions)  # Takes in "actions" that are sampled, not the scaled "action"
        log_probs -= T.log(1-action.pow(2)+self.reparam_noise)  # reparam noise is added to save from log0
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
