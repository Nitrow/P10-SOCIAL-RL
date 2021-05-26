# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import gym
from gym import spaces
import numpy as np
import random
import math
import math as m
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

        
        self.conveyor = self.supervisor.getFromDef('CONVEYOR')
        
        self.cans = [0]*6
        self.position = [0]*6
        self.orientation = [0]*6
        
        
        for i in range(len(self.cans)):
            self.cans[i] = self.supervisor.getFromDef('can' + str(i+1))
        
        
        self.done = True
        self.cansDict = {}
        
        self.plot_rewards = []
        self.total_rewards = 0

        self.epOutcome = ""

        self.successReward = 1000
        self.constPunishment = -0.01
        self.counter = 0
        
          
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(60,), dtype=np.float32)
        
        
    def reset(self):
        print('\n ------------------------------------ RESET ------------------------------------ \n')
        self.supervisor.simulationReset()
        
        self.conveyor.restartController()
        self.counter = 0
        
            
        
        self.supervisor.step(self.TIME_STEP) 

           

        self.supervisor.step(self.TIME_STEP) 
        state = self.getCans()

        self.total_rewards = 0    
        self.done = False    

        state = [0]*60
        return state


    def step(self, action):
        
        
        self.supervisor.step(self.TIME_STEP)   
           
        state = self.getCans()
        #print (state)
        
        if action < 6 and self.cansDict["Can " + str(action+1)]["Position"][0] > 0:
            reward = 1 - self.cansDict["Can " + str(action+1)]["Position"][0]
            yaw =self._util_axisangle2euler(self.cansDict["Can " + str(action+1)]["Orientation"])
            reward += yaw*(math.pi/180)
            reward += self.cansDict["Can " + str(action+1)]["Color"][0]*1 + self.cansDict["Can " + str(action+1)]["Color"][1]*2 + self.cansDict["Can " + str(action+1)]["Color"][2]*3
            
            
        else:
            reward = 0
        #print(self.cansDict)
        #print(reward)
        
        
        
        
        self.counter = self.counter + 1
  
       
        for i in range(len(self.cans)):
            if self.cansDict["Can " + str(i+1)]["Position"][0] < -0.49:
                self.cans[i].getField("translation").setSFVec3f([1.79,0.83,0.66])
        
        counter =+ 1
        
        if self.counter == 4000:
            self.done = True
        
        
        
        
        return [state, float(reward), self.done, {}]


    def render(self, mode='human'):
        pass


    def saveEpisode(self, reward):
        with open(os.path.join(self.path, "documentation.txt"), 'a') as f:
            f.write(reward)


    def getState(self):
  
        return  
        #return self.tcp.getPosition() + self.target 

    def getCans(self):
        state = [0]*6
        for i in range(len(self.cans)):
            self.position = self.cans[i].getField("translation").getSFVec3f()
            self.orientation = self.cans[i].getField("rotation").getSFRotation()
            self.color = self.cans[i].getField("color").getSFColor()
            if self.position[0] < 1.7 and self.position[0] > -0.5 :

                state[i] = self.position + self.orientation + self.color
                canDict = {"Position": self.position,
                   "Orientation": self.orientation,
                   "Color" : self.color}
                   
                self.cansDict.update ({"Can " + str(i+1): canDict})
            else:
                state[i] = [0]*3 + [0]*4 + [0]*3
                canDict = {"Position": [0]*3,
                   "Orientation": [0]*4,
                   "Color" : [0]*3}   
                   
                self.cansDict.update ({"Can " + str(i+1): canDict})    
                
        return state[0] + state[1] + state[2] + state[3] + state[4] + state[5]   
    
    def _util_axisangle2euler(self, rotation):
        # YZX
        x,y,z,angle = rotation
        s = m.sin(angle)
        c = m.cos(angle)
        t = 1-c
        if ((x*y*t + z*s) > 0.998): # north pole singularity
            yaw = round(m.degrees(2*m.atan2(x*m.sin(angle/2), m.cos(angle/2))))

        elif ((x*y*t + z*s) < -0.998):
            yaw = round(m.degrees(-2*m.atan2(x*m.sin(angle/2), m.cos(angle/2))))

        else:
            yaw = round(m.degrees(m.atan2(y*s - x*z, 1 - (y*y + z*z) * t)))
        return yaw
