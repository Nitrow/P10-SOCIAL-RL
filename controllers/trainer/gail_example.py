import gym

from stable_baselines import GAIL, SAC, DQN
from stable_baselines.gail import ExpertDataset, generate_expert_traj

# Generate expert trajectories (train expert)
model = SAC('MlpPolicy', 'Pendulum-v0', verbose=0)
generate_expert_traj(model, 'expert_pendulum', n_timesteps=100, n_episodes=10)
#
# # Load the expert dataset
dataset = ExpertDataset(expert_path='expert_pendulum.npz', traj_limitation=10, verbose=1)

#model = DQN('MlpPolicy', 'CartPole-v1', verbose=1)
      # Train a DQN agent for 1e5 timesteps and generate 10 trajectories
      # data will be saved in a numpy archive named `expert_cartpole.npz`
#generate_expert_traj(model, 'expert_cartpole', n_timesteps=int(1e5), n_episodes=10)

#model = GAIL('MlpPolicy', 'CartPole-v1', dataset, verbose=1)
model = GAIL('MlpPolicy', 'Pendulum-v0', dataset, verbose=1)
# Note: in practice, you need to train for 1M steps to have a working policy
model.learn(total_timesteps=1000000)
model.save("gail_pendulum")
#model.save("gail_cartpole")
del model # remove to demonstrate saving and loading

model = GAIL.load("gail_pendulum")

env = gym.make('Pendulum-v0')
obs = env.reset()

for i in range(10000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()