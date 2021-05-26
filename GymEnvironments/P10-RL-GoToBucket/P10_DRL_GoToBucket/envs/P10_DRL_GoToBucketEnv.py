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


class P10_DRL_GoToBucketEnv(gym.Env):

    def __init__(self):
        
        
        
        
        random.seed(1)
        self.id = "SAC - " + "2021-05-20 11_05_30" + '_P10_MarkEnv_SingleJoint - all joints' 
        #self.id = '2021-04-15 09_44_43_SAC_P10_MarkEnv_SingleJoint_' 
        self.path = "data/" + self.id + "/"
        os.makedirs(self.path, exist_ok=True)
        self.supervisor = Supervisor()
        self.TIME_STEP = int(self.supervisor.getBasicTimeStep())
        self.robot_node = self.supervisor.getFromDef("UR3")
        self.green_bin = self.supervisor.getFromDef("green_bin")
        self.blue_bin = self.supervisor.getFromDef("blue_bin")

        
        
        
        
        self.tcp = self.supervisor.getFromDef('TCP')
        self.robot_pos = np.array(self.robot_node.getPosition())
        
        self.timeout = 500
        
                
        self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        #self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        
        self.motors = [0] * len(self.joint_names)
        self.sensors = [0] * len(self.joint_names)
        
        self.touch_sensors = [0] * 7#(len(self.joint_names)+1)
     
        self.done = True
        self.prev_dist = 0
        
        self.plot_rewards = []
        self.total_rewards = 0

        self.epOutcome = ""

        self.collisionReward = -300 
        self.distanceReward = -0.1
        self.distanceDeltaReward = 100
        self.successReward = 1000
        self.constPunishment = -0.01
        self.rewardstr = "success {}, collision {} distance {}".format(self.successReward, self.collisionReward, self.distanceDeltaReward)
        self.figure_file = self.path + "{} - Rewards {} - Timeout at {}".format(self.id, self.rewardstr, str(self.timeout))
        
        #print(self.goal_node.getSFVec3f())
        self.oldDistance = 0
        self.distancex = 0
        self.distancey = 0
        self.distancez = 0
        self.tcp_pos_world = self.tcp.getPosition()
        self.counter = 0
         
            
            
        self.actionScale = 3
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.joint_names),), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(17,), dtype=np.float32)
        
        self.documentation = "Action space: all joints State space: tcp pos (xyz), target pos (xyz), joint positions ."
        self.documentation += "{} - Rewards {} - Timeout at {}\n".format(self.id, self.rewardstr, str(self.timeout))
        self.saveEpisode(self.documentation)
        

        
        
      
        
        
    def reset(self):
        print('\n ------------------------------------ RESET ------------------------------------ \n')
        self.supervisor.simulationReset()
        self.counter = 0
        
        self._getSensors()
        self._getMotors()
        self.supervisor.step(self.TIME_STEP) 
        self.supervisor.step(self.TIME_STEP) 
        
        self._setTarget()
        
        color_choice = random.choice([0,1])
        
        if color_choice == 0:
            self.blue_bin.getField("translation").setSFVec3f(self.target)
        
            self.supervisor.step(self.TIME_STEP) 

        
            self.Bucket_positions = self.blue_bin.getPosition()
            
            #print("BLUE!")
            self.goal_target = self.Bucket_positions   
            
            self.goal_target[1] = self.goal_target[1]+1
            
        if color_choice == 1:
            self.green_bin.getField("translation").setSFVec3f(self.target)
        
            self.supervisor.step(self.TIME_STEP) 

        
            self.Bucket_positions = self.green_bin.getPosition()
            
            #print("GREEN!")
            
            self.goal_target = self.Bucket_positions   
            
            
            self.goal_target[1] = self.goal_target[1]+1
        
        


        
        self.supervisor.step(self.TIME_STEP) 
        #self.supervisor.step(self.TIME_STEP) 
        #print(self.goal_target)
        
        
        self.supervisor.step(self.TIME_STEP)
        self.total_rewards = 0    
        self.done = False    
        self.prevdist = distance.euclidean(self.tcp.getPosition(), self.goal_target)
        state = self.getState()
        return np.asarray(state)


    def step(self, action):
        #print(self.counter)
        
        self._getMotors()
        for i in range(len(self.joint_names)):
            self.motors[i].setVelocity(float(action[i]))
        

        
        self.supervisor.step(self.TIME_STEP)   
           
        
        state = self.getState()
        
        distDifference = self.prevdist - self.Distance  
           
        self.prevdist = self.Distance  
               
        #print("Distance: {}".format(np.linalg.norm(np.array(self.tcp.getPosition()) - self.robot_pos)))
       
        # Normalize the distance by the maximum robot reach so it's between 0 and 1
        reward = (self.distanceDeltaReward * distDifference) #+ self.constPunishment
        #print(self.total_rewards)
        self.counter = self.counter + 1
              
        if self.counter >= self.timeout:
            self.epOutcome = "Timeout"
            print("Timeout")
            self.done = True
        if self.Distance  < 0.1:
            self.epOutcome = "Success"
            print("Success")
            self.done = True
            reward += self.successReward
        if self._isCollision():
            self.epOutcome = "Collision"
            print("Collision")
            self.done = True
            reward += self.collisionReward
        #print(self.done)
        self.total_rewards += reward
        if self.done:
            self.saveEpisode(str(round(self.total_rewards)) + ";")
            self.plot_learning_curve()
        #print(reward)
        return [state, float(reward), self.done, {}]


    def _isCollision(self):
        """
        Returns True if any of the touch sensors are activated
        """
        return any([self.touch_sensors[i].getValue() for i in range(len(self.touch_sensors))])


    def _getSensors(self):
        """
        Initializes the touch sensors
        """
        for i in range(len(self.touch_sensors)):
            self.touch_sensors[i] = self.supervisor.getDevice("touch_sensor"+str(i+1))
            self.touch_sensors[i].enable(self.TIME_STEP)


    def _getMotors(self):
        """
        Initializes the motors and their sensors
        """
        for i in range(len(self.joint_names)):
            # Get motors
            self.motors[i] = self.supervisor.getDevice(self.joint_names[i])
            self.motors[i].setPosition(float('inf'))
            self.motors[i].setVelocity(0)
            # Get sensors and enable them
            self.sensors[i] = self.supervisor.getDevice(self.joint_names[i]+'_sensor')
            self.sensors[i].enable(self.TIME_STEP)
            
    
    def _setTarget(self):
        # generate a point around the circle 0.75m far from the robot, making sure it's far away 

        
        
        x = random.choice([random.uniform(-0.45, -0.55), random.uniform(0.45, 0.55)])
        z = random.uniform(0.5, 0.35)
        y = 0
            
        self.target = list(np.array([x, y, z]))

            
    
    def render(self, mode='human'):
        pass


    def saveEpisode(self, reward):
        with open(os.path.join(self.path, "documentation.txt"), 'a') as f:
            f.write(reward)


    def getState(self):
    
    
        self.Distance = distance.euclidean(self.tcp.getPosition(), self.goal_target)
        
        return [self.sensors[i].getValue() for i in range(len(self.sensors)-1)] + [self.motors[i].getVelocity() for i in range(len(self.motors)-1)]  + self.tcp.getPosition() +  [self.Distance] + self.goal_target
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
    
