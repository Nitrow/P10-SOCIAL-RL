import os
import torch as T
import numpy as np
import torch.nn.functional as F
from buffer import ReplayBuffer
from models.sac import ActorNetwork, CriticNetwork, ValueNetwork


class SAC_Agent():
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[8], action_range=2,
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
