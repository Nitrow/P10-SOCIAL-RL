import gym
from gym import wrappers
import numpy as np
from dueling_ddqn_agent import DuelingDDQNAgent
from utils import plot_learning_curve, make_env

from P10_DRL_D3QN.envs import P10_DRL_D3QNEnv


feature_names=['Position\nin m', 'orientation\nin degree', 'color']


def eval():
    env = P10_DRL_D3QNEnv()
    best_score = -np.inf
    load_checkpoint = True
    n_games = 100
    agent = DuelingDDQNAgent(gamma=0.99, epsilon=1.0, lr=0.0001,
                     input_dims=(env.observation_space.shape),
                     n_actions=env.action_space.n, mem_size=50000, eps_min=0.05,
                     batch_size=32, replace=10000, eps_dec=1e-5,
                     chkpt_dir='models/', algo='DuelingDDQNAgent',
                     env_name='P10_DRL_D3QNEnv')
                     
                     
    s = env.reset()
    done = False
    reward = 0
    while not(done):
        if video:
            env.render()
        a = agent.choose_action(observation)
        s_, r, done, info = env.step(action)
        s = s_
        reward +=reward
        
        action_log = np.append(action_log, a)
        if len(state_log)==0:
            state_log = np.expand_dims(s, axis=0)
        else:
            state_log = np.append(state_log, np.expand_dims(s, axis=0), axis=0)
    
    return state_log, action_log
    
state_log, action_log = eval()    



def explain(agent=agent, state_log=state_log, feature_names=feature_names, action_log=action_log, cmap='coolwarm', save_fig=False):

    # Build explainer
    model = ([ddpg.S], ddpg.a) # define inputs and outputs of actor
    explainer = shap.DeepExplainer(model, state_log, session=ddpg.sess) # build DeepExplainer
    shap_values = explainer.shap_values(state_log) # Calculate shap values
       
    state_log_re = state_log*(env.state_max - env.state_min) + env.state_min # rescale state log 
    
    norm = plt.Normalize(vmin=-1, vmax=1) # define color scala between -1 and +1 (like the agents action space)  
        
    fig = plt.figure(figsize=(15,12))
    gs = fig.add_gridspec(9, hspace=0)
    axs = gs.subplots(sharex=True, sharey=False)
    axs[0].plot(state_log_re[:,0])
    axs[0].plot(state_log_re[:,2])
    axs[0].set_ylabel('velocity\nin m/s')
    axs[0].legend(['vehicle', 'speed limit'])
    axs[1].scatter(range(0, len(action_log)), action_log, c=action_log-explainer.expected_value, cmap = cmap, norm=norm)
    axs[1].set_ylabel('action')
    axs[1].yaxis.set_label_position('right')
    axs[2].scatter(range(0,len(shap_values[0][:,0])), state_log_re[:,0], c=shap_values[0][:,0], cmap=cmap, norm=norm)
    axs[2].set_ylabel(feature_names[0])
    axs[3].scatter(range(0,len(shap_values[0][:,0])), state_log_re[:,1], c=shap_values[0][:,1], cmap=cmap, norm=norm)
    axs[3].set_ylabel(feature_names[1])
    axs[3].yaxis.set_label_position('right')
    axs[4].scatter(range(0,len(shap_values[0][:,0])), state_log_re[:,2], c=shap_values[0][:,2], cmap=cmap, norm=norm)
    axs[4].set_ylabel(feature_names[2])
    axs[5].scatter(range(0,len(shap_values[0][:,0])), state_log_re[:,3], c=shap_values[0][:,3], cmap=cmap, norm=norm)
    axs[5].set_ylabel(feature_names[3])   
    axs[5].yaxis.set_label_position('right')
    axs[6].scatter(range(0,len(shap_values[0][:,0])), state_log_re[:,5], c=shap_values[0][:,5], cmap=cmap, norm=norm)
    axs[6].set_ylabel(feature_names[5])    
    axs[7].scatter(range(0,len(shap_values[0][:,0])), state_log_re[:,4], c=shap_values[0][:,4], cmap=cmap, norm=norm)
    axs[7].set_ylabel(feature_names[4])   
    axs[7].yaxis.set_label_position('right')
    axs[8].scatter(range(0,len(shap_values[0][:,0])), state_log_re[:,6], c=shap_values[0][:,6], cmap=cmap, norm=norm)
    axs[8].set_ylabel(feature_names[6])    
    axs[8].set_xlabel('Distance in m')
    
    if save_fig:
        fig.savefig("Shap_RL_Example.pdf", bbox_inches='tight')

