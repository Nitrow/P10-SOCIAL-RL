# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import gym
from gym import spaces
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from datetime import datetime
import torch

import os
from scipy.spatial import distance

#import ikpy
#from ikpy.chain import Chain


from controller import Robot, Motor, Supervisor


class P10_DRL_D3QNEnv(gym.Env):

    def __init__(self):
        
        
        
        
        random.seed(1)
      
        self.supervisor = Supervisor()
        self.TIME_STEP = int(self.supervisor.getBasicTimeStep())
        self.robot_node = self.supervisor.getFromDef("UR3")

        
        self.timeout = 1500
        
                
        
     
        self.done = True
        self.prev_dist = 0
        
        self.plot_rewards = []
        self.total_rewards = 0

        self.epOutcome = ""

        self.successReward = 1000
        self.constPunishment = -0.01
        self.rewardstr = "success {}, collision {} distance {}".format(self.successReward, self.collisionReward, self.distanceDeltaReward)
        self.figure_file = self.path + "{} - Rewards {} - Timeout at {}".format(self.id, self.rewardstr, str(self.timeout))
        
         
            
            
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(42,), dtype=np.float32)
        
        self.documentation = "Action space: all joints State space: tcp pos (xyz), target pos (xyz), joint positions ."
        self.documentation += "{} - Rewards {} - Timeout at {}\n".format(self.id, self.rewardstr, str(self.timeout))
        self.saveEpisode(self.documentation)
        
        
        
        
      
        
        
    def reset(self):
        print('\n ------------------------------------ RESET ------------------------------------ \n')
        self.supervisor.simulationReset()
        self.counter = 0
        

        self.supervisor.step(self.TIME_STEP) 


        self.total_rewards = 0    
        self.done = False    

        state = self.getState()
        return np.asarray(state)


    def step(self, action):
        #print(self.counter)
        

        self.picked_Can = action 
        
        
        self.generateCans()
        
        self.supervisor.step(self.TIME_STEP)   
           
        state = self.getState()
        
       
               

        #print(self.distance)
        self.prevdisty = self.distancey
        self.prevdisty2 = self.distancey2
        #print("Distance: {}".format(np.linalg.norm(np.array(self.tcp.getPosition()) - self.robot_pos)))
       
        # Normalize the distance by the maximum robot reach so it's between 0 and 1
       
        #print(self.total_rewards)
        self.counter = self.counter + 1
              
              
              
       
        #print(self.done)
        self.total_rewards += reward
        if self.done:
            self.saveEpisode(str(round(self.total_rewards)) + ";")
            self.plot_learning_curve()
        #print(reward)
        return [state, float(reward), self.done, {}]


  
    
  

    

    def render(self, mode='human'):
        pass


    def saveEpisode(self, reward):
        with open(os.path.join(self.path, "documentation.txt"), 'a') as f:
            f.write(reward)


    def getState(self):
    
    
        self.distancey = distance.euclidean(self.tcpy.getPosition(), self.goaly.getPosition())
        self.distancey2 = distance.euclidean(self.tcpy2.getPosition(), self.goaly2.getPosition())
        return [self.sensors[i].getValue() for i in range(len(self.sensors))] + [self.motors[i].getVelocity() for i in range(len(self.motors))]  + self.tcpy.getPosition() + self.tcpy2.getPosition() + self.goaly.getPosition() + self.goaly2.getPosition() + [self.distancey, self.distancey2]
        #return self.tcp.getPosition() + self.target 


    def plot_learning_curve(self):
        self.plot_rewards.append(self.total_rewards)
        x = [i+1 for i in range(len(self.plot_rewards))]
        running_avg = np.zeros(len(self.plot_rewards))
        for i in range(len(running_avg)):
            running_avg[i] = np.mean(self.plot_rewards[max(0, i-100):(i+1)])
        plt.plot(x, running_avg)
        plt.title('Running average of previous 100 scores')
        plt.savefig(self.figure_file)
    
    
    def generateCans():
        global can_num, pos_choice
        can_num += 1
        can_distances = ["000", "999", "556", "479", "506", "490", "530"]
        can_distances.remove(pos_choice)
        can_colors = ["green", "yellow", "red"]
        pos_choice = random.choice(can_distances)

        can_color = random.choice(can_colors)
        colorCount[can_color] += 1
        can = "resources/" + can_color + "_can_" + pos_choice + ".wbo"
        root_children.importMFNode(-1, can)
       