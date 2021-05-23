import numpy as np

class ReplayBuffer():
    """
    Max_size: Don't want the memory to be unbounded
    Input_shape: Matches the state (observation) coming from the environment
    N_actions: Number of actions (or number of components)
    """
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size # We don't want it to be unbounded
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool) # Done flags

    def store_transition(self, state, action, reward, state_, done):
        # Get the index of the first available memory
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        # How many memories are stored in the buffer
        max_mem = min(self.mem_cntr, self.mem_size)
        # Get batch_size amount of sequence of random integers between 0 and max_memory
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones